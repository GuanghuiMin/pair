"""Steps 1-2 for OfficeBench: mine compaction boundaries and run PRE/POST continuations in containers.

    python -m pair.benchmarks.officebench.rollouts mine  --run RUNS/officebench/p0_w2048_s1 --out B
    python -m pair.benchmarks.officebench.rollouts audit --run RUNS/officebench/p0_w2048_s1 --out B [--workers 20]
    python -m pair.benchmarks.officebench.rollouts run   --boundaries B/boundaries.jsonl --out B --rounds 3 --workers 20

Same protocol as AppWorld: restore the environment by replaying the recorded actions before the
boundary, hand the agent the PRE context (raw prefix at t = 1, otherwise previous summary plus the raw
turns since the previous boundary) or the POST context (this summary plus the retained turn), run to
completion or to the remaining budget with no further compression, grade with the benchmark's checkers.
Continuations are allocated by successive halving (pair.halving): round d runs draw d of both arms for the
boundaries still active, every boundary in round 0 and the top half by current evidence of harm afterwards.
OfficeBench replay is checked, not assumed: every replayed observation is compared with the recorded
one (timestamps masked); a boundary whose prefix does not reproduce is recorded as `replay_mismatch`
and gets no continuation. `audit` replays every trajectory twice with no LLM call and reports replay
fidelity, determinism, checker agreement and whether the rebuilt history renders exactly what the
compressor was recorded to receive.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SEED_BASE = 1000
SUMMARY_OPEN, SUMMARY_CLOSE = "\n\n<HISTORY_SUMMARY>\n", "\n</HISTORY_SUMMARY>"
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?")
_LS_TIME_RE = re.compile(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}\b")
_HEX_RE = re.compile(r"0x[0-9a-fA-F]{6,}")
_LS_INODE_RE = re.compile(r"(?m)^\s*\d{5,}\s+(?=\d+\s+[-dl][rwx-]{9})")


def normalise(s: str) -> str:
    s = _TS_RE.sub("<TS>", s or "")
    s = _LS_TIME_RE.sub("<TS>", s)
    s = _HEX_RE.sub("<HEX>", s)
    s = _LS_INODE_RE.sub("<INODE> ", s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------------------------------ mining --
def compactions(events):
    return [(int(e["compression_before_step"]), e.get("summary") or "") for e in events if e.get("kind")]


def step_events(events):
    return {int(e["step"]): e for e in events if "step" in e}


def mine(run_dir: Path):
    rows, skipped = [], []
    for line in open(run_dir / "run_records.jsonl"):
        if not line.strip():
            continue
        rec = json.loads(line)
        path = run_dir / "trajectories" / f"{rec['task_id']}.json"
        if not path.exists():
            skipped.append((rec["task_id"], "no trajectory file"))
            continue
        tr = json.load(open(path))
        bnds, num = compactions(tr["events"]), step_events(tr["events"])
        for t, (step, summary) in enumerate(bnds, 1):
            lo = 1 if t == 1 else bnds[t - 2][0] - 1
            hi = step - 1
            if hi < 1 or any(i not in num for i in range(1, hi + 1)):
                skipped.append((rec["task_id"], f"t{t}: missing steps before {step}"))
                continue
            if not summary:
                skipped.append((rec["task_id"], f"t{t}: empty summary"))
                continue
            rows.append({"boundary_id": f"{run_dir.name}::{rec['task_id']}#t{t}", "trajectory": str(path),
                         "task_id": rec["task_id"], "task": tr["task"], "subtask": tr["subtask"], "split": rec["split"],
                         "t": t, "n_boundaries": len(bnds), "step": step, "raw_suffix": [lo, hi],
                         "trajectory_success": rec["success"], "trajectory_steps": rec["num_steps"], "wd_name": tr["wd_name"]})
    return rows, skipped


# ------------------------------------------------------------------------------------- container --
def _restore(job: dict):
    """Rebuild the recorded working directory, environment and agent."""
    from pair.benchmarks.officebench import agent as A
    tr = json.load(open(job["trajectory_path"]))
    os.chdir(A.OB_ROOT)
    task, subtask = tr["task"], tr["subtask"]
    wd = A.make_workdir(task, tr["wd_name"])
    task_dir = os.path.join(wd, "tasks", task)
    task_config = A.load_task_config(task, subtask, task_dir)
    env = A.make_env(task, wd, task_config.get("task", task))
    agent = A.make_agent(env, job["model"], task_config)
    first_prompt = agent.build_prompt(env)
    checks = {"first_prompt_match": first_prompt == tr["first_prompt"],
              "system_match": agent.system_message == tr["system_message"],
              "task_config_match": task_config == tr.get("task_config")}
    return A, tr, env, agent, task_config, task_dir, step_events(tr["events"]), checks, first_prompt


def _replay(env, agent, steps: dict, n: int):
    """Replay recorded steps 1..n; returns (per-step observations and next prompts, mismatching steps)."""
    got, mism = [], []
    for i in range(1, n + 1):
        e = steps[i]
        obs, _reward, done, _info = env.step(e["action"] if e.get("action") is not None else "")
        obs = obs if obs is not None else ""
        if normalise(obs) != normalise(e.get("observation") or ""):
            mism.append(i)
        got.append({"obs": obs, "next_prompt": None if done else agent.build_prompt(env), "done": bool(done)})
        if done and i < n:
            raise RuntimeError(f"task ended during replay at step {i} of {n}")
    return got, mism


def _tool_call(e: dict):
    act = e.get("action")
    args = json.dumps({"action": act}, separators=(",", ":")) if isinstance(act, str) else json.dumps({"action": act})
    return [{"id": e["tool_call_id"], "type": "function", "function": {"name": "execute_action", "arguments": args}}]


def _turn_messages(e: dict, next_prompt: str):
    """The two messages the runtime stored for step e: the assistant turn and its observation."""
    if e.get("action_mode") == "toolcall" and e.get("tool_call_id"):
        return [{"role": "assistant", "content": e.get("thinking") or "", "tool_calls": _tool_call(e)},
                {"role": "tool", "tool_call_id": e["tool_call_id"], "content": next_prompt}]
    return [{"role": "assistant", "content": e.get("raw_response") or ""},
            {"role": "user", "content": next_prompt}]


def _add_session(mm, first_text: str, steps: dict, got: list, lo: int, hi: int):
    mm.add_user_prompt(first_text)
    pending = None
    for i in range(lo, hi + 1):
        for m in _turn_messages(steps[i], got[i - 1]["next_prompt"]):
            if m["role"] == "assistant":
                mm.add_assistant_response(m["content"], tool_calls=m.get("tool_calls"))
                pending = (m.get("tool_calls") or [{}])[0].get("id")
            elif m["role"] == "tool":
                mm.add_tool_response(m["tool_call_id"], m["content"])
            else:
                mm.add_user_prompt(m["content"])
    return pending


def audit_job(job: dict) -> dict:
    """Replay one recorded trajectory twice without any LLM call and check it reproduces."""
    rec = {"task_id": None, "error": None}
    t0, env = time.time(), None
    try:
        A, tr, env, agent, task_config, task_dir, steps, checks, _ = _restore(job)
        rec.update(task_id=tr["task_id"], recorded_success=tr["success"], num_steps=tr["num_steps"], **checks)
        n = tr["num_steps"]
        root_before = set(os.listdir(A.OB_ROOT))
        got1, mism1 = _replay(env, agent, steps, n)
        env.close()
        env = None
        stray = sorted(set(os.listdir(A.OB_ROOT)) - root_before - {tr["wd_name"]})   # files written outside the workdir
        for name in stray:
            p = os.path.join(A.OB_ROOT, name)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        rec["stray_root_writes"] = stray
        _, _, env, agent2, _, task_dir, steps2, _, _ = _restore(job)
        got2, _ = _replay(env, agent2, steps2, n)
        det_diff = [i + 1 for i in range(n) if normalise(got1[i]["obs"]) != normalise(got2[i]["obs"])]
        passed, eval_error = A.evaluate(task_config, task_dir)
        rec.update(fidelity=round((n - len(mism1)) / n, 4) if n else None, n_mismatch=len(mism1), mismatch_steps=mism1[:20],
                   determinism_diff_steps=det_diff[:20], replay_eval_success=passed, replay_eval_error=eval_error,
                   eval_match=(passed == bool(tr["success"])))
        mm = agent2.memory_manager
        bnds = compactions(tr["events"])
        comp_events = [e for e in tr["events"] if e.get("kind")]
        conv = []
        for t, (b, _s) in enumerate(bnds, 1):
            lo = 1 if t == 1 else bnds[t - 2][0] - 1
            msgs = [{"role": "system", "content": tr["system_message"]}, {"role": "user", "content": tr["first_prompt"]}]
            for i in range(lo, b - 1):
                msgs += _turn_messages(steps[i], got2[i - 1]["next_prompt"])
            rendered = mm.convert_llm_history_to_text(msgs).strip()
            m = re.search(r"<conversation>(.*?)</conversation>", comp_events[t - 1].get("input_user") or "", re.S)
            recorded = m.group(1).strip() if m else None
            conv.append({"t": t, "step": b, "exact": recorded is not None and rendered == recorded,
                         "match": recorded is not None and normalise(rendered) == normalise(recorded)})
        rec["conversation_checks"] = conv
        rec["all_conversations_match"] = all(c["match"] for c in conv) if conv else None
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["error_trace"] = traceback.format_exc()[-2000:]
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:  # noqa: BLE001
            pass
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def rollout_job(job: dict) -> dict:
    """One PRE/POST continuation from a boundary, run to completion and graded."""
    b = job["boundary"]
    rec = {k: b[k] for k in ("boundary_id", "task_id", "task", "subtask", "split", "t", "step")}
    rec.update(arm=job["arm"], draw=job["draw"], seed=SEED_BASE + job["draw"], model=job["model"], steps=[],
               termination_reason=None, error=None, success=False, eval_error=None, peak_prompt_tokens=0, cumulative_input_tokens=0)
    t0, env = time.time(), None
    try:
        A, tr, env, agent, task_config, task_dir, steps, checks, first_prompt = _restore(job)
        rec.update(checks)
        if not (checks["first_prompt_match"] and checks["system_match"]):
            raise RuntimeError("first prompt / system message do not reproduce")
        step, t = b["step"], b["t"]
        lo, hi = b["raw_suffix"]
        bnds = compactions(tr["events"])
        got, mism = _replay(env, agent, steps, hi)
        rec.update(replay_n=hi, replay_mismatch_steps=mism[:20], replay_fidelity=round((hi - len(mism)) / hi, 4) if hi else None)
        if mism:
            rec["termination_reason"] = "replay_mismatch"
            rec["error"] = f"replay mismatch at steps {mism[:10]}"
            return rec
        if job["arm"] == "PRE":
            first_text = first_prompt if t == 1 else first_prompt + SUMMARY_OPEN + bnds[t - 2][1] + SUMMARY_CLOSE
            s_lo = lo
        else:
            first_text = first_prompt + SUMMARY_OPEN + bnds[t - 1][1] + SUMMARY_CLOSE
            s_lo = hi
        mm = agent.memory_manager
        pending = _add_session(mm, first_text, steps, got, s_lo, hi)
        budget = A.MAX_STEPS - step + 1
        rec["budget"] = budget
        rec["termination_reason"] = "budget_exhausted"
        peak = cum = 0
        for k in range(budget):
            if k > 0:
                user_prompt = agent.build_prompt(env)
                if pending:
                    mm.add_tool_response(pending, user_prompt)
                else:
                    mm.add_user_prompt(user_prompt)
            before = agent.llm.total_input_tokens
            r = A.act(agent, mm, env)
            used = agent.llm.total_input_tokens - before
            peak, cum = max(peak, used), cum + used
            pending = r["tool_call_id"]
            rec["steps"].append({"action": r["action"], "executed_action": r["executed_action"],
                                 "observation": r["observation"], "reasoning": r["thinking"], "action_mode": r["action_mode"]})
            if r["done"]:
                rec["termination_reason"] = "task_completed" if r["observation"] == "Task finished" else "task_failed_got_stuck"
                break
        passed, eval_error = A.evaluate(task_config, task_dir)
        rec.update(success=passed, eval_error=eval_error, peak_prompt_tokens=peak, cumulative_input_tokens=cum)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["error_trace"] = traceback.format_exc()[-2000:]
        rec["termination_reason"] = "error"
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:  # noqa: BLE001
            pass
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


# ------------------------------------------------------------------------------------------- host --
def run_ctr(job: dict, out: Path, image: str, scratch: str, timeout: int, name: str) -> dict:
    from pair.benchmarks.officebench.run import docker_cmd
    jobdir = os.path.join(scratch, f"rb_{name}_{uuid.uuid4().hex[:8]}")
    os.makedirs(jobdir)
    shutil.copy2(job["trajectory_host_path"], os.path.join(jobdir, "trajectory.json"))
    job = {**job, "trajectory_path": "/work/trajectory.json"}
    cmd = docker_cmd(job, jobdir, image, module="pair.benchmarks.officebench.rollouts")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log = f"$ {' '.join(cmd[:8])} ...\n--- stdout ---\n{p.stdout[-20000:]}\n--- stderr ---\n{p.stderr[-20000:]}\nrc={p.returncode}\n"
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        log = f"TIMEOUT after {timeout}s\n{(e.stdout or '')[-5000:]}\n{(e.stderr or '')[-5000:]}"
        rc = -9
    (out / "containers").mkdir(parents=True, exist_ok=True)
    (out / "containers" / f"{name}.log").write_text(log)
    rp = os.path.join(jobdir, "out", "result.json")
    if os.path.exists(rp):
        rec = json.load(open(rp))
    else:
        rec = {"error": f"container rc={rc}, no result", "termination_reason": "container_error", "success": False,
               "elapsed_s": round(time.time() - t0, 2)}
        if job["mode"] == "rollout":
            b = job["boundary"]
            rec.update({k: b[k] for k in ("boundary_id", "task_id", "task", "subtask", "split", "t", "step")},
                       arm=job["arm"], draw=job["draw"], seed=SEED_BASE + job["draw"], model=job["model"], steps=[])
        else:
            rec.update(task_id=job.get("task_id"))
    rec["container_rc"] = rc
    shutil.rmtree(jobdir, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["mine", "audit", "run"])
    ap.add_argument("--job", help="(container) job json")
    ap.add_argument("--run", help="mine/audit: run directory")
    ap.add_argument("--boundaries", help="run: boundaries.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3, help="successive-halving rounds (one PRE/POST pair each)")
    ap.add_argument("--keep", type=float, default=0.5, help="fraction of a round's boundaries kept for the next")
    ap.add_argument("--seed", type=int, default=2027, help="tie-break order of the ranking")
    ap.add_argument("--boundary-ids", nargs="*", default=None)
    ap.add_argument("--task-ids", nargs="*", default=None, help="audit: only these task ids")
    ap.add_argument("--model", default=os.environ.get("PAIR_AGENT_MODEL"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--image", default=os.environ.get("PAIR_OB_IMAGE", "pair-officebench"))
    ap.add_argument("--scratch", default=os.environ.get("PAIR_OB_SCRATCH", "/tmp/pair_officebench"))
    ap.add_argument("--timeout", type=int, default=5400)
    a = ap.parse_args()

    if a.job:
        job = json.load(open(a.job))
        rec = audit_job(job) if job["mode"] == "audit" else rollout_job(job)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "result.json").write_text(json.dumps(rec))
        print(json.dumps({k: rec.get(k) for k in ("task_id", "boundary_id", "arm", "success", "termination_reason", "error")}))
        return

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "mine":
        rows, skipped = mine(Path(a.run).resolve())
        with open(out / "boundaries.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{len(rows)} boundaries from {len({r['trajectory'] for r in rows})} trajectories; skipped {len(skipped)}")
        for s in skipped[:10]:
            print("  skipped", *s)
        return

    from pair.benchmarks.officebench import agent as A
    A.require_env()
    os.makedirs(a.scratch, exist_ok=True)
    if a.mode == "audit":
        run = Path(a.run).resolve()
        path = out / "audit.jsonl"
        have = {json.loads(l)["task_id"] for l in open(path) if l.strip()} if path.exists() else set()
        recs = [json.loads(l) for l in open(run / "run_records.jsonl") if l.strip()]
        jobs = [{"mode": "audit", "task_id": r["task_id"], "model": a.model,
                 "trajectory_host_path": str(run / "trajectories" / f"{r['task_id']}.json")}
                for r in recs if r["task_id"] not in have and (not a.task_ids or r["task_id"] in a.task_ids)]
        print(f"audit {run.name}: {len(recs)} trajectories, {len(have)} done, {len(jobs)} to run, {a.workers} containers", flush=True)
        with open(path, "a") as f, ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(run_ctr, j, out, a.image, a.scratch, a.timeout, f"audit_{j['task_id']}") for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"  [{i}/{len(jobs)}] {r.get('task_id')} fidelity={r.get('fidelity')} determinism_diff={r.get('determinism_diff_steps')} "
                      f"eval_match={r.get('eval_match')} conversations_match={r.get('all_conversations_match')} err={r.get('error')}", flush=True)
        return

    if a.mode == "run":
        from pair import halving
        path = out / "rollouts.jsonl"
        bounds = [json.loads(l) for l in open(a.boundaries) if l.strip()]
        if a.boundary_ids:
            bounds = [b for b in bounds if b["boundary_id"] in set(a.boundary_ids)]
        for rnd in range(a.rounds):
            draws = halving.load_draws(path)
            keep = halving.active([b["boundary_id"] for b in bounds], draws, rnd, a.keep, seed=a.seed)
            have = {(bid, arm, d) for bid, arms in draws.items() for arm in arms for d in arms[arm]}
            jobs = [{"mode": "rollout", "boundary": b, "arm": arm, "draw": rnd, "model": a.model, "trajectory_host_path": b["trajectory"]}
                    for b in bounds if b["boundary_id"] in keep for arm in ("PRE", "POST") if (b["boundary_id"], arm, rnd) not in have]
            print(f"round {rnd + 1}/{a.rounds}: {len(keep)} of {len(bounds)} boundaries active, {len(jobs)} to run, "
                  f"{a.workers} containers", flush=True)
            t0 = time.time()
            with open(path, "a") as f, ThreadPoolExecutor(a.workers) as ex:
                futs = [ex.submit(run_ctr, j, out, a.image, a.scratch, a.timeout,
                                  f"{j['boundary']['task_id']}_t{j['boundary']['t']}_{j['arm']}{j['draw']}") for j in jobs]
                for i, fut in enumerate(as_completed(futs), 1):
                    r = fut.result()
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print(f"  [{i}/{len(jobs)}] {r.get('boundary_id')} {r.get('arm')}{r.get('draw')} steps={len(r.get('steps') or [])} "
                          f"pass={r.get('success')} {r.get('termination_reason')} err={(r.get('error') or '')[:80]} {(time.time() - t0) / 60:.1f}min", flush=True)
        return
    ap.error("mode required (mine | audit | run) or --job")


if __name__ == "__main__":
    main()
