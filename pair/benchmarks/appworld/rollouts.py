"""Step 2 (continuations): PRE/POST rollouts from AppWorld compaction boundaries.

    python -m pair.benchmarks.appworld.rollouts --boundaries B/boundaries.jsonl --out B --draws 3 --workers 20

Both arms restore the environment by replaying the recorded actions before the boundary into a fresh
AppWorld instance and then hand the frozen agent one of two contexts:
    PRE   the context without this compaction: the raw prefix at t = 1, otherwise the previous summary
          plus the raw steps since the previous boundary
    POST  the context the agent actually saw: this compaction's summary plus the retained raw turn
The agent runs with no further compression until it finishes or exhausts the remaining step budget
(50 minus the boundary step); the end state is graded by AppWorld's evaluator. Draw d uses seed
1000 + d. Re-running fills in missing (boundary, arm, draw) triples only.
"""
import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pair.benchmarks.appworld import agent as A
from pair.benchmarks.appworld.boundaries import compactions, steps

SEED_BASE = 1000


def turn(e):
    return {"code": e.get("code") or "", "observation": e.get("observation") or "", "reasoning": e.get("reasoning") or ""}


def contexts(b: dict):
    """(prefix codes to replay, {arm: (summary, turns to place in the history)})."""
    ev = json.load(open(b["trajectory"]))["events"]
    bnds, num = compactions(ev), steps(ev)
    lo, hi = b["raw_suffix"]
    t = b["t"]
    prefix_codes = [num[i]["code"] for i in sorted(num) if i < b["step"]]
    pre = (None, [turn(num[i]) for i in range(1, hi + 1)]) if t == 1 else \
          (bnds[t - 2][1], [turn(num[i]) for i in range(lo, hi + 1)])
    post = (bnds[t - 1][1], [turn(num[hi])])
    return prefix_codes, {"PRE": pre, "POST": post}


def run_one(job: dict) -> dict:
    rec = {k: job[k] for k in ("boundary_id", "task_id", "split", "t", "step", "arm", "draw", "seed")}
    rec.update(model=job["model"], steps=[], termination_reason=None, error=None)
    t0, env = time.time(), None
    budget = A.MAX_STEPS - job["step"] + 1
    try:
        env = A.make_env(job["task_id"], job["split"],
                         f"pair_bd_{job['task_id']}_t{job['t']}_{job['arm']}{job['draw']}_{os.getpid()}",
                         job["step"] + budget + 2)
        agent = A.make_agent(env, job["model"], job["temperature"], job["seed"])
        first_prompt = agent.build_prompt(env)
        for i, code in enumerate(job["prefix_codes"]):
            _, _, done, _ = env.step(code)
            if done:
                raise RuntimeError(f"task ended during replay at step {i + 1}")
        mm = agent.memory_manager
        summary, turns = job["summary"], job["turns"]
        mm.add_user_prompt(first_prompt if summary is None else first_prompt + A.SUMMARY_OPEN + summary + A.SUMMARY_CLOSE)
        for i, tn in enumerate(turns):
            A.add_turn(mm, tn, f"replay_{i}")
        rec["termination_reason"] = "budget_exhausted"
        for _ in range(budget):
            code, obs, done, info, meta = A.act(agent, mm, env)
            rec["steps"].append({"code": code, "observation": obs, "reasoning": meta.get("reasoning", "")})
            if done:
                rec["termination_reason"] = info.get("reason", "done")
                break
        rec.update(A.evaluate(env))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
        rec["termination_reason"] = "error"
    finally:
        if env is not None:
            A.close(env)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundaries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=3, help="independent continuations per arm")
    ap.add_argument("--model", default=os.environ.get("PAIR_AGENT_MODEL"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--boundary-ids", nargs="*", default=None)
    a = ap.parse_args()
    A.require_env()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rollouts.jsonl"
    have = {(r["boundary_id"], r["arm"], r["draw"]) for r in map(json.loads, open(path)) if not r["error"]} if path.exists() else set()
    bounds = [json.loads(l) for l in open(a.boundaries) if l.strip()]
    if a.boundary_ids:
        bounds = [b for b in bounds if b["boundary_id"] in set(a.boundary_ids)]
    jobs = []
    for b in bounds:
        prefix_codes, ctx = contexts(b)
        for arm in ("PRE", "POST"):
            for d in range(a.draws):
                if (b["boundary_id"], arm, d) in have:
                    continue
                summary, turns = ctx[arm]
                jobs.append({**{k: b[k] for k in ("boundary_id", "task_id", "split", "t", "step")},
                             "arm": arm, "draw": d, "seed": SEED_BASE + d, "prefix_codes": prefix_codes,
                             "summary": summary, "turns": turns, "model": a.model, "temperature": a.temperature})
    print(f"{len(bounds)} boundaries, {len(have)} rollouts present, {len(jobs)} to run", flush=True)
    t0 = time.time()
    with open(path, "a") as f, ProcessPoolExecutor(a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  [{i}/{len(jobs)}] {r['boundary_id']} {r['arm']}{r['draw']} steps={len(r['steps'])} "
                  f"pass={r.get('success')} {r['termination_reason']} {(time.time() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
