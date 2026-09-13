"""Run OfficeBench subtasks with ACON's agent, one container per subtask.

Host:
    python -m pair.benchmarks.officebench.run --tasks data/tasklists/officebench_train.jsonl \
        --out RUNS/officebench/p0_w2048_s1 --tag p0_w2048_s1 --method prompts/p0 --window 2048 --seed 1 --workers 20

For every (task, subtask) not yet in <out>/run_records.jsonl the host writes a job file and starts
`docker run --rm` with the ACON checkout mounted read-only, a private copy of its officebench
experiments directory shadow-mounted over its own path (where the working directory lives and dies),
the canonical task export read-only, this repository read-only and the job directory at /work. The
container runs this module with --job.
Outputs: <out>/run_records.jsonl, <out>/trajectories/<task>__<subtask>.json (steps and compaction
events, the input of boundary mining), <out>/containers/<task>__<subtask>.log.

Container:
    python -m pair.benchmarks.officebench.run --job /work/job.json --out /work/out
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PAIR_HOME = Path(__file__).resolve().parents[3]
CTR_SKELETON = ("experiment_config.py", "prompts", "utils", "configs", "data", "evaluation")
CTR_ENV_PASS = ("ACON_VLLM_BASE_URL", "ACON_VLLM_API_KEY", "ACON_VLLM_API_STYLE", "ACON_VLLM_REASONING_EFFORT",
                "ACON_VLLM_NO_THINKING_KWARG", "OPENAI_API_KEY", "PAIR_AGENT_MODEL", "ACON_ROOT", "OB_CANONICAL_TASKS",
                "PAIR_COMPRESSOR_MODEL", "PAIR_COMPRESSOR_BASE_URL", "PAIR_COMPRESSOR_API_KEY", "PAIR_COMPRESSOR_REASONING")
DEFAULT_IMAGE = "pair-officebench"


def empty_record(job: dict) -> dict:
    return {"task_id": f"{job['task']}__{job['subtask']}", "task": job["task"], "subtask": job["subtask"], "split": job["split"],
            "method": Path(job["method"]).name if job["method"] else "full", "window": job["window"], "seed": job["seed"],
            "success": False, "score": 0.0, "num_steps": 0, "n_compressions": 0, "peak_prompt_tokens": 0,
            "cumulative_input_tokens": 0, "termination_reason": None, "error": None}


# ------------------------------------------------------------------------------- container side --
def run_job(job: dict, out_dir: str) -> dict:
    from pair.benchmarks.officebench import agent as A
    from pair.compressor import Compressor, count_tokens
    task, subtask, seed, tag = job["task"], job["subtask"], job["seed"], job["tag"]
    rec = empty_record(job)
    t0, env, events, extra = time.time(), None, [], {}
    try:
        os.chdir(A.OB_ROOT)
        name = A.wd_name(tag, task, subtask, seed)
        wd = A.make_workdir(task, name)
        task_dir = os.path.join(wd, "tasks", task)
        task_config = A.load_task_config(task, subtask, task_dir)
        env = A.make_env(task, wd, task_config.get("task", task))
        agent = A.make_agent(env, job["model"], task_config)
        mm = agent.memory_manager
        comp = Compressor(job["method"], job["window"], seed) if job["method"] else None
        if comp is not None:
            mm.history_optimizer = comp
            mm.do_history_optimization = True
            mm.history_summary_rule = "reset"
            mm.preserve_last_k_turns = 1
            mm.baseline_strategy = "none"
            mm.history_summary_interval = -1
        instruction = task_config.get("task", "")
        first_prompt = agent.build_prompt(env)
        if comp is not None:
            comp.agent_prompt = agent.system_message + "\n\n" + first_prompt   # the agent's fixed prefix
        extra = {"system_message": agent.system_message, "first_prompt": first_prompt, "wd_name": name,
                 "task_config": task_config, "agent_config": A.exp_config_dict(agent)}
        peak = cum = 0
        rec["termination_reason"] = "max_iterations_reached"
        pending_call = None
        for step in range(1, A.MAX_STEPS + 1):
            user_prompt = first_prompt if step == 1 else agent.build_prompt(env)
            if pending_call:                     # observation + instruction block as the tool result
                mm.add_tool_response(pending_call, user_prompt)
            else:
                mm.add_user_prompt(user_prompt)
            if comp:
                n_before = len(comp.events)
                mm.optimize_history(task=instruction, opt_args={})
                if len(comp.events) > n_before:
                    ev = comp.events[-1]
                    events.append({"compression_before_step": step, "executed_actions_before": len(env.history_log),
                                   "kind": ev["kind"], "summary": ev["summary"], "summary_tokens": count_tokens(ev["summary"]),
                                   "input_system": ev["input_system"], "input_user": ev["input_user"],
                                   "agent_visible": ev["agent_visible"], "error": ev["error"]})
            before = agent.llm.total_input_tokens
            r = A.act(agent, mm, env)
            used = agent.llm.total_input_tokens - before
            peak, cum = max(peak, used), cum + used
            rec["num_steps"] = step
            pending_call = r["tool_call_id"]
            events.append({"step": step, "executed_actions_before": len(env.history_log) - (1 if r["executed_action"] is not None else 0),
                           **{k: r[k] for k in ("raw_response", "thinking", "reasoning_source", "api_reasoning_summary",
                                                "action", "executed_action", "observation", "action_mode", "tool_call_id", "extra_tool_calls")}})
            if r["done"]:
                rec["termination_reason"] = "task_finished" if r["observation"] == "Task finished" else "task_failed_got_stuck"
                break
        passed, eval_error = A.evaluate(task_config, task_dir)
        rec.update(success=passed, eval_error=eval_error, peak_prompt_tokens=peak, cumulative_input_tokens=cum,
                   n_compressions=len(comp.events) if comp else 0, instruction=instruction, score=1.0 if passed else 0.0)
    except Exception as e:  # noqa: BLE001
        import traceback
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["error_trace"] = traceback.format_exc()[-3000:]
        rec["termination_reason"] = "error"
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:  # noqa: BLE001
            pass
    rec["elapsed_s"] = round(time.time() - t0, 2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "trajectory.json").write_text(json.dumps({**rec, **extra, "events": events}))
    (Path(out_dir) / "run_record.json").write_text(json.dumps(rec))
    return rec


# ------------------------------------------------------------------------------------ host side --
def docker_cmd(job: dict, jobdir: str, image: str, module: str = "pair.benchmarks.officebench.run") -> list:
    """The container command; pair.benchmarks.officebench.rollouts reuses it for continuation jobs."""
    from pair.benchmarks.officebench import agent as A
    ob, acon = A.OB_ROOT, str(A.ACON_ROOT)
    priv = os.path.join(jobdir, "ob")
    os.makedirs(priv)
    os.makedirs(os.path.join(jobdir, "home"))
    os.makedirs(os.path.join(jobdir, "out"))
    for item in CTR_SKELETON:
        src = os.path.join(ob, item)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(priv, item))
        elif os.path.exists(src):
            shutil.copy2(src, os.path.join(priv, item))
    json.dump(job, open(os.path.join(jobdir, "job.json"), "w"))
    tz = open("/etc/timezone").read().strip() if os.path.exists("/etc/timezone") else "UTC"
    cmd = ["docker", "run", "--rm", "--network", "host", f"--user={os.getuid()}:{os.getgid()}",
           "-e", "HOME=/work/home", "-e", f"TZ={tz}", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "ACON_VLLM_STRIP_THINK=0",
           "-e", f"PYTHONPATH={acon}/src:{PAIR_HOME}",
           "-v", "/etc/passwd:/etc/passwd:ro", "-v", "/etc/group:/etc/group:ro",
           "-v", f"{acon}:{acon}:ro",
           "-v", f"{priv}:{ob}",
           "-v", f"{A.CANONICAL_TASKS}:{A.CANONICAL_TASKS}:ro",
           "-v", f"{PAIR_HOME}:{PAIR_HOME}:ro",
           "-v", f"{jobdir}:/work", "-w", ob]
    for k in CTR_ENV_PASS:
        if os.environ.get(k) is not None:
            cmd += ["-e", f"{k}={os.environ[k]}"]
    cmd += [image, "sh", "-c", f"umask 002; exec python3 -m {module} --job /work/job.json --out /work/out"]
    return cmd


def run_in_container(job: dict, out: Path, image: str, scratch: str, timeout: int) -> dict:
    tid = f"{job['task']}__{job['subtask']}"
    jobdir = os.path.join(scratch, f"{job['tag']}_{tid}_{uuid.uuid4().hex[:8]}")
    os.makedirs(jobdir)
    cmd = docker_cmd(job, jobdir, image)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log = f"$ {' '.join(cmd[:8])} ...\n--- stdout ---\n{p.stdout[-20000:]}\n--- stderr ---\n{p.stderr[-20000:]}\nrc={p.returncode}\n"
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        log = f"TIMEOUT after {timeout}s\n{(e.stdout or '')[-5000:]}\n{(e.stderr or '')[-5000:]}"
        rc = -9
    (out / "containers").mkdir(parents=True, exist_ok=True)
    (out / "containers" / f"{tid}.log").write_text(log)
    rp = os.path.join(jobdir, "out", "run_record.json")
    if os.path.exists(rp):
        rec = json.load(open(rp))
        (out / "trajectories").mkdir(parents=True, exist_ok=True)
        shutil.move(os.path.join(jobdir, "out", "trajectory.json"), out / "trajectories" / f"{tid}.json")
    else:
        rec = {**empty_record(job), "termination_reason": "container_error", "error": f"container rc={rc}, no run_record",
               "elapsed_s": round(time.time() - t0, 2)}
    rec["container_rc"] = rc
    shutil.rmtree(jobdir, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", help="(container) job json to run")
    ap.add_argument("--out", required=True, help="host: run directory; container: output directory")
    ap.add_argument("--tasks", help="jsonl of {task, subtask, split}")
    ap.add_argument("--tag", help="run tag, also the working-directory prefix")
    ap.add_argument("--method", default=None, help="compression prompt directory; omit for full context")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("PAIR_AGENT_MODEL"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10, help="concurrent containers")
    ap.add_argument("--image", default=os.environ.get("PAIR_OB_IMAGE", DEFAULT_IMAGE))
    ap.add_argument("--scratch", default=os.environ.get("PAIR_OB_SCRATCH", "/tmp/pair_officebench"))
    ap.add_argument("--timeout", type=int, default=5400, help="seconds per container")
    a = ap.parse_args()

    if a.job:
        rec = run_job(json.load(open(a.job)), a.out)
        print(json.dumps({k: rec.get(k) for k in ("task_id", "success", "num_steps", "n_compressions", "termination_reason", "error")}))
        return
    if not (a.tasks and a.tag):
        sys.exit("host mode needs --tasks and --tag")
    if bool(a.method) != bool(a.window):
        sys.exit("--method and --window go together")
    from pair.benchmarks.officebench import agent as A
    A.require_env()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.makedirs(a.scratch, exist_ok=True)
    records = out / "run_records.jsonl"
    done = {json.loads(l)["task_id"] for l in open(records) if l.strip()} if records.exists() else set()
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    method = str(Path(a.method).resolve()) if a.method else None
    jobs = [{"task": t["task"], "subtask": t["subtask"], "split": t.get("split", "test"), "tag": a.tag,
             "method": method, "window": a.window, "model": a.model, "seed": a.seed}
            for t in tasks if f"{t['task']}__{t['subtask']}" not in done]
    print(f"{out.name}: {len(tasks)} subtasks, {len(done)} done, {len(jobs)} to run, method={a.method or 'full'} "
          f"window={a.window} model={a.model} image={a.image} workers={a.workers}", flush=True)
    n_pass = 0
    with open(records, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_in_container, j, out, a.image, a.scratch, a.timeout) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            n_pass += bool(r["success"])
            print(f"  [{i}/{len(jobs)}] {r['task_id']} pass={r['success']} steps={r['num_steps']} compactions={r['n_compressions']} "
                  f"{r['termination_reason']} err={r['error']}", flush=True)
    print(f"{out.name}: {n_pass}/{len(jobs)} passed", flush=True)


if __name__ == "__main__":
    main()
