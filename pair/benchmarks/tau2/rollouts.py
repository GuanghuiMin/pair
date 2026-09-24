"""Steps 1-2 for tau2-bench: mine compaction boundaries and run PRE/POST continuations.

    (tau2 env) python -m pair.benchmarks.tau2.rollouts mine --run RUNS/tau2/p0_w2048_s1 --domain retail --out B
    (tau2 env) python -m pair.benchmarks.tau2.rollouts run  --boundaries B/boundaries.jsonl --out B --rounds 3 --workers 24

A boundary is one recorded compaction: (task, compaction index t, n_seen = messages consumed when the
compressor fired). The conversation prefix messages[:n_seen] restores the environment (tau2 replays the
state-changing tool calls), the user simulator and the agent; no container is needed.
    PRE   the context without this compaction: the raw prefix at t = 1, otherwise the previous summary plus the
          raw messages since the previous boundary (what the agent held just before this compaction), compaction off
    POST  the agent continues from the recorded summary plus the raw tail (the most recent turn)
Budget = the remaining orchestrator steps of the recorded run (100 minus the prefix), at least 10.
Success is tau2's deterministic reward on the completed conversation; steps are the agent's new turns.
Continuations are allocated by successive halving (pair.halving): round d runs draw d of both arms for the
boundaries still active, every boundary in round 0 and the top half by current evidence of harm afterwards.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

PAIR_HOME = str(Path(__file__).resolve().parents[3])
if PAIR_HOME not in sys.path:
    sys.path.insert(0, PAIR_HOME)
from pair import halving  # noqa: E402
from pair.benchmarks.tau2.convert import deterministic_success  # noqa: E402

MAX_STEPS_TOTAL = 100
MIN_BUDGET = 10
SEED_BASE = 1000
_LOCK = threading.Lock()
_RESULTS: dict = {}


def mine(run_dir: str, domain: str, out: Path):
    res = json.load(open(os.path.join(run_dir, f"tau2_{domain}.json")))
    sims = {str(s["task_id"]): s for s in res["simulations"]}
    tasks = {str(t["id"]): t for t in res.get("tasks", [])}
    tag = os.path.basename(os.path.normpath(run_dir))
    comps = [json.loads(l) for l in open(os.path.join(run_dir, f"compactions_{domain}.jsonl")) if l.strip()]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in sorted(comps, key=lambda c: (str(c["task_id"]), c["n_compaction"])):
        tid = str(c["task_id"])
        s = sims.get(tid)
        if s is None or tid not in tasks:
            continue
        msgs = s["messages"]
        n_seen, n_tail, n_win = int(c["n_seen"]), int(c.get("n_tail_msgs", 0)), int(c.get("n_window_msgs", 0))
        cut = n_seen - n_tail                     # where the raw tail starts in the agent's message list
        rows.append({"boundary_id": f"{tag}::{domain}-{tid}#t{c['n_compaction']}",
                     "trajectory": os.path.abspath(os.path.join(run_dir, f"tau2_{domain}.json")),
                     "domain": domain, "task_id": f"{domain}-{tid}", "tau2_task_id": tid, "t": int(c["n_compaction"]),
                     "n_seen": n_seen, "cut": cut, "raw_suffix": [max(0, cut - n_win), cut], "n_tail_msgs": n_tail,
                     "step": sum(1 for m in msgs[:n_seen] if m["role"] == "assistant"),
                     "budget": max(MIN_BUDGET, MAX_STEPS_TOTAL - n_seen),
                     "summary": c["summary"], "prev_summary": c.get("prev_summary"), "window_tokens": c.get("window_tokens"),
                     "summary_tokens": c.get("summary_tokens"),
                     "recorded_success": bool(deterministic_success(s.get("reward_info") or {})), "recorded_n_messages": len(msgs)})
    with open(out / "boundaries.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} boundaries from {len({r['task_id'] for r in rows})} tasks -> {out / 'boundaries.jsonl'}")


def load_task_and_prefix(b: dict):
    from tau2.data_model.simulation import SimulationRun
    from tau2.data_model.tasks import Task
    res = _RESULTS.setdefault(b["trajectory"], json.load(open(b["trajectory"])))
    sim = next(s for s in res["simulations"] if str(s["task_id"]) == b["tau2_task_id"])
    task = Task.model_validate(next(t for t in res["tasks"] if str(t["id"]) == b["tau2_task_id"]))
    return task, list(SimulationRun.model_validate(sim).messages or [])[:b["n_seen"]]


def rollout_job(b: dict, arm: str, draw: int, user_llm: str, user_temp: float, agent_llm: str) -> dict:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.data_model.tasks import InitialState
    from tau2.data_model.message import ToolMessage
    from tau2.runner.build import build_text_orchestrator
    from tau2.runner.simulation import run_simulation
    from tau2.evaluator.evaluator import EvaluationType
    from pair.benchmarks.tau2 import agent
    agent.register()
    t0 = time.time()
    rec = {"boundary_id": b["boundary_id"], "arm": arm, "draw": draw, "seed": SEED_BASE + draw, "task_id": b["task_id"],
           "t": b["t"], "step": b["step"], "budget": b["budget"], "steps": [], "success": False, "termination_reason": None, "error": None}
    try:
        task, prefix = load_task_and_prefix(b)
        init = task.initial_state
        task2 = task.model_copy(update={"initial_state": InitialState(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            message_history=deepcopy(prefix))})
        os.environ["PAIR_TAU2_NO_COMPACT"] = "1"
        os.environ.pop("PAIR_TAU2_METHOD", None)
        config = TextRunConfig(domain=b["domain"], agent="pair_agent", user="user_simulator",
                               llm_agent=agent_llm, llm_args_agent={"temperature": 1.0, "reasoning_effort": "medium"},
                               llm_user=user_llm, llm_args_user={"temperature": user_temp},
                               max_steps=b["budget"], seed=SEED_BASE + draw)
        orch = build_text_orchestrator(config, task2, seed=SEED_BASE + draw)
        if arm == "POST":
            starts = [i for i, m in enumerate(prefix) if not isinstance(m, ToolMessage)]
            orch.agent.preset = (b["summary"], deepcopy(prefix[starts[-1] if starts else 0:]))
        elif int(b["t"]) >= 2:
            if not b.get("prev_summary"):
                raise ValueError(f"{b['boundary_id']}: no previous summary for the PRE context")
            orch.agent.preset = (b["prev_summary"], deepcopy(prefix[int(b["raw_suffix"][0]):]))
        sim = run_simulation(orch, evaluation_type=EvaluationType.ALL)
        new = (sim.messages or [])[len(prefix):]
        steps = []
        for m in new:
            if getattr(m, "role", "") == "assistant":
                calls = [{"name": tc.name, "arguments": tc.arguments} for tc in (getattr(m, "tool_calls", None) or [])]
                action = json.dumps(calls, ensure_ascii=False) if calls else (m.content or "")
                steps.append({"action": action, "executed_action": action, "observation": ""})
            elif steps:
                steps[-1]["observation"] = (steps[-1]["observation"] + f"\n[{m.role}] {(m.content or '')}").strip()
        ri = sim.reward_info.model_dump(mode="json") if sim.reward_info else {}
        term = str(sim.termination_reason)
        rec.update(steps=steps, success=deterministic_success(ri), official_reward=ri.get("reward"), reward_breakdown=ri.get("reward_breakdown"),
                   termination_reason="budget_exhausted" if term.endswith("MAX_STEPS") else term, n_new_messages=len(new), n_prefix=len(prefix))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["error_trace"] = traceback.format_exc()[-2000:]
        rec["termination_reason"] = "error"
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def run(a):
    bounds = [json.loads(l) for l in open(a.boundaries) if l.strip()]
    if a.boundary_ids:
        bounds = [b for b in bounds if b["boundary_id"] in set(a.boundary_ids)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rollouts.jsonl"
    for rnd in range(a.rounds):
        draws = halving.load_draws(path)
        keep = halving.active([b["boundary_id"] for b in bounds], draws, rnd, a.keep, seed=a.seed)
        have = {(bid, arm, d) for bid, arms in draws.items() for arm in arms for d in arms[arm]}
        jobs = [(b, arm, rnd) for b in bounds if b["boundary_id"] in keep for arm in ("PRE", "POST")
                if (b["boundary_id"], arm, rnd) not in have]
        print(f"round {rnd + 1}/{a.rounds}: {len(keep)} of {len(bounds)} boundaries active, {len(jobs)} to run, "
              f"{a.workers} workers", flush=True)
        t0, done = time.time(), 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(rollout_job, b, arm, d, a.user_llm, a.user_temperature, a.agent_llm) for b, arm, d in jobs]
            for fut in as_completed(futs):
                rec = fut.result()
                done += 1
                with _LOCK, open(path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                print(f"  [{done}/{len(jobs)}] {rec['boundary_id']} {rec['arm']}{rec['draw']} steps={len(rec['steps'])} pass={rec['success']} "
                      f"{rec['termination_reason']} err={rec['error']} {(time.time() - t0) / 60:.1f}min", flush=True)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--run", required=True)
    m.add_argument("--domain", default="retail")
    m.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--boundaries", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--rounds", type=int, default=halving.ROUNDS, help="successive-halving rounds (one PRE/POST pair each)")
    r.add_argument("--keep", type=float, default=halving.KEEP, help="fraction of a round's boundaries kept for the next")
    r.add_argument("--seed", type=int, default=halving.SEED, help="tie-break order of the ranking")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--boundary-ids", nargs="*", default=None)
    r.add_argument("--user-llm", default="gpt-4.1-2025-04-14")
    r.add_argument("--user-temperature", type=float, default=0.0)
    r.add_argument("--agent-llm", default=os.environ.get("PAIR_TAU2_AGENT_LLM", "openai/responses/gpt-5.6-luna"))
    a = ap.parse_args()
    if a.mode == "mine":
        mine(a.run, a.domain, Path(a.out))
    else:
        import litellm
        litellm.suppress_debug_info = True
        run(a)


if __name__ == "__main__":
    main()
