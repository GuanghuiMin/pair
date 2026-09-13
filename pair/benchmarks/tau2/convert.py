"""Convert a tau2 results file (plus the compaction log) into the shared run layout.

    python -m pair.benchmarks.tau2.convert --results RUN/tau2_retail.json --domain retail \
        [--compactions RUN/compactions_retail.jsonl] --out RUN --seed 1 --split test --method p0 --window 2048

Writes <out>/run_records.jsonl (one line per task) and <out>/trajectories/<domain>-<task>.json in the
same shape as the AppWorld and OfficeBench runners. `success` is tau2's deterministic reward: every
component of the task's reward basis except NL_ASSERTION equals 1 (no LLM judge); `score` is the
official reward. Steps are agent turns. Tasks already recorded are skipped.
"""
import argparse
import json
import os
from pathlib import Path


def deterministic_success(reward_info: dict) -> bool:
    basis = [b for b in (reward_info.get("reward_basis") or []) if b != "NL_ASSERTION"]
    breakdown = reward_info.get("reward_breakdown") or {}
    if not basis:
        return float(reward_info.get("reward") or 0.0) == 1.0
    return all(float(breakdown.get(b, 0.0) or 0.0) == 1.0 for b in basis)


def step_events(messages: list) -> list:
    """One step per assistant message; the tool results and user reply that follow are its observation."""
    events, step, i = [], 0, 0
    while i < len(messages):
        m = messages[i]
        if m["role"] != "assistant":
            i += 1
            continue
        step += 1
        calls = m.get("tool_calls") or []
        action = json.dumps([{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls], ensure_ascii=False) if calls else (m.get("content") or "")
        obs, j = [], i + 1
        while j < len(messages) and messages[j]["role"] != "assistant":
            obs.append(f"[{messages[j]['role']}] {messages[j].get('content') or ''}")
            j += 1
        usage = m.get("usage") or {}
        events.append({"step": step, "turn_idx": m.get("turn_idx"), "raw_response": m.get("content") or "", "action": action,
                       "action_mode": "toolcall" if calls else "text", "n_tool_calls": len(calls), "observation": "\n".join(obs),
                       "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})
        i = j
    return events


def convert(results_path: str, domain: str, out: Path, seed: int, split: str, compactions_path=None, method="full", window=0) -> int:
    r = json.load(open(results_path))
    info = r.get("info", {})
    comps = {}
    if compactions_path and os.path.exists(compactions_path):
        for l in open(compactions_path):
            if l.strip():
                c = json.loads(l)
                comps.setdefault((str(c.get("task_id")), int(c.get("trial", 0) or 0)), []).append(c)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectories").mkdir(exist_ok=True)
    records = out / "run_records.jsonl"
    done = {json.loads(l)["task_id"] for l in open(records) if l.strip()} if records.exists() else set()
    n_new = 0
    for s in r["simulations"]:
        tid = f"{domain}-{s['task_id']}"
        if tid in done:
            continue
        msgs = s.get("messages") or []
        ev = step_events(msgs)
        ptoks = [e["prompt_tokens"] for e in ev if e.get("prompt_tokens")]
        ri = s.get("reward_info") or {}
        clog = comps.get((str(s["task_id"]), int(s.get("trial", 0) or 0)), [])
        cevents = []
        for c in clog:
            n_seen = int(c.get("n_seen", 0) or 0)          # messages consumed when the compaction fired
            before = sum(1 for m in msgs[:n_seen] if m["role"] == "assistant") + 1
            cevents.append({"compression_before_step": before, "kind": "update" if c.get("prev_summary") else "first",
                            "summary": c.get("summary"), "summary_tokens": c.get("summary_tokens"), "input_system": c.get("input_system"),
                            "input_user": c.get("input_user") or c.get("window_text"), "prev_summary": c.get("prev_summary"),
                            "agent_visible": c.get("agent_visible", False), "error": c.get("error"), "window_tokens": c.get("window_tokens")})
        events = sorted(ev + cevents, key=lambda e: (e.get("compression_before_step", e.get("step", 0)), 0 if "compression_before_step" in e else 1))
        rec = {"task_id": tid, "task": domain, "subtask": str(s["task_id"]), "split": split, "method": method, "window": window, "seed": seed,
               "success": deterministic_success(ri), "score": float(ri.get("reward") or 0.0),
               "reward_basis": ri.get("reward_basis"), "reward_breakdown": ri.get("reward_breakdown"),
               "num_steps": len(ev), "n_messages": len(msgs), "n_compressions": len(clog),
               "peak_prompt_tokens": max(ptoks) if ptoks else 0, "cumulative_input_tokens": sum(ptoks) if ptoks else 0,
               "termination_reason": s.get("termination_reason"), "error": None, "elapsed_s": s.get("duration"),
               "tau2_sim_id": s.get("id"), "tau2_trial": s.get("trial"), "tau2_seed": s.get("seed"),
               "agent_cost": s.get("agent_cost"), "user_cost": s.get("user_cost")}
        traj = {**rec, "system_message": s.get("policy"), "first_prompt": next((m.get("content") for m in msgs if m["role"] == "user"), None),
                "agent_config": info.get("agent_info"), "events": events}
        (out / "trajectories" / f"{tid}.json").write_text(json.dumps(traj, ensure_ascii=False))
        with open(records, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_new += 1
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--compactions", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--method", default="full")
    ap.add_argument("--window", type=int, default=0)
    a = ap.parse_args()
    n = convert(a.results, a.domain, Path(a.out), a.seed, a.split, a.compactions, a.method, a.window)
    print(f"{n} new records -> {a.out}")


if __name__ == "__main__":
    main()
