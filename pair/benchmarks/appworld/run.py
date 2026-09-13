"""Run AppWorld tasks with ACON's agent, with or without history compression.

    python -m pair.benchmarks.appworld.run --tasks data/tasklists/appworld_train.jsonl \
        --out RUNS/appworld/p0_w4096_s1 --method prompts/p0 --window 4096 --seed 1 --workers 20

Without --method the agent keeps its full history. Writes <out>/run_records.jsonl (one line per
task) and <out>/trajectories/<task_id>.json (steps and compaction events, the input of boundary
mining). Re-running skips tasks already recorded.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pair.benchmarks.appworld import agent as A


def run_task(job: dict) -> dict:
    from pair.compressor import Compressor, count_tokens
    task_id, split, seed = job["task_id"], job["split"], job["seed"]
    method = Path(job["method"]).name if job["method"] else "full"
    rec = {"task_id": task_id, "split": split, "method": method, "window": job["window"], "seed": seed,
           "success": False, "score": 0.0, "num_steps": 0, "n_compressions": 0,
           "peak_prompt_tokens": 0, "cumulative_input_tokens": 0, "termination_reason": None, "error": None}
    t0, env, events = time.time(), None, []
    try:
        env = A.make_env(task_id, split, f"pair_{method}_{job['window']}_s{seed}_{task_id}", A.MAX_STEPS + 2)
        agent = A.make_agent(env, job["model"], job["temperature"], seed)
        mm = agent.memory_manager
        comp = None
        if job["method"]:
            comp = Compressor(job["method"], job["window"], seed)
            mm.history_optimizer = comp
            mm.do_history_optimization = True
            mm.history_summary_rule = "reset"
            mm.preserve_last_k_turns = 1
            mm.baseline_strategy = "none"
            mm.history_summary_interval = -1
        instruction = env.task.instruction
        first_prompt = agent.build_prompt(env)
        if comp:
            comp.agent_prompt = first_prompt
        mm.add_user_prompt(first_prompt)
        peak = cum = 0
        rec["termination_reason"] = "max_iterations_reached"
        for step in range(1, A.MAX_STEPS + 1):
            if comp:
                n_before = len(comp.events)
                mm.optimize_history(task=instruction, opt_args={})
                if len(comp.events) > n_before:
                    ev = comp.events[-1]
                    events.append({"compression_before_step": step, "kind": ev["kind"], "summary": ev["summary"],
                                   "summary_tokens": count_tokens(ev["summary"]), "input_system": ev["input_system"],
                                   "input_user": ev["input_user"], "agent_visible": ev["agent_visible"], "error": ev["error"]})
            before = agent.llm.total_input_tokens
            code, obs, done, info, meta = A.act(agent, mm, env)
            used = agent.llm.total_input_tokens - before
            peak, cum = max(peak, used), cum + used
            rec["num_steps"] = step
            events.append({"step": step, "reasoning": meta.get("reasoning", ""), "code": code, "observation": obs})
            if done:
                rec["termination_reason"] = info.get("reason", "done")
                break
        ev = A.evaluate(env)
        rec.update(success=ev["success"], score=ev["score"], peak_prompt_tokens=peak, cumulative_input_tokens=cum,
                   n_compressions=len(comp.events) if comp else 0, instruction=instruction)
        tdir = Path(job["out"]) / "trajectories"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{task_id}.json").write_text(json.dumps({**rec, "first_prompt": first_prompt, "events": events}))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["termination_reason"] = "error"
    finally:
        if env is not None:
            A.close(env)
    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="jsonl of {task_id, split}")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--method", default=None, help="compression prompt directory; omit for full context")
    ap.add_argument("--window", type=int, default=0, help="context budget in tokens")
    ap.add_argument("--model", default=os.environ.get("PAIR_AGENT_MODEL"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if bool(a.method) != bool(a.window):
        sys.exit("--method and --window go together")
    A.require_env()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = out / "run_records.jsonl"
    done = {json.loads(l)["task_id"] for l in open(records) if l.strip()} if records.exists() else set()
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    jobs = [{"task_id": t["task_id"], "split": t.get("split", "train"), "out": str(out),
             "method": str(Path(a.method).resolve()) if a.method else None, "window": a.window,
             "model": a.model, "temperature": a.temperature, "seed": a.seed}
            for t in tasks if t["task_id"] not in done]
    print(f"{out.name}: {len(tasks)} tasks, {len(done)} done, {len(jobs)} to run, "
          f"method={a.method or 'full'} window={a.window} model={a.model}", flush=True)
    n_pass = 0
    with open(records, "a") as f, ProcessPoolExecutor(a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(run_task, j) for j in jobs]), 1):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            n_pass += r["success"]
            print(f"  [{i}/{len(jobs)}] {r['task_id']} pass={r['success']} steps={r['num_steps']} "
                  f"compactions={r['n_compressions']} err={r['error']}", flush=True)
    print(f"{out.name}: {n_pass}/{len(jobs)} passed", flush=True)


if __name__ == "__main__":
    main()
