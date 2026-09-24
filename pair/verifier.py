"""Step 2: verify adverse compaction boundaries from PRE/POST continuations.

    python -m pair.verifier --rollouts B/rollouts.jsonl --out B/effects.jsonl [--hazard-min 0.5] [--burden-min 5]
                            [--rounds 3] [--keep 0.5] [--seed 2027]

Continuations are allocated by successive halving (pair.halving): every boundary has one PRE/POST pair, and only
the boundaries that reached the last round have `--rounds` pairs. For every boundary with continuations from both arms:
    hazard  = pass rate(PRE) - pass rate(POST)          outcome hazard H_t
    burden  = mean steps(POST) - mean steps(PRE)        interaction burden B_t
computed on its draws 0..rounds-1. A boundary is retained if it completed every round and hazard >= --hazard-min or
burden >= --burden-min. Retained boundaries are labelled `error` (hazard condition) or `burden` (burden condition
only); the label selects the evidence channel shown to the optimizer.
"""
import argparse
import json
import statistics as st
from pathlib import Path

from pair import halving


def effects(draws: dict, hazard_min: float, burden_min: float, rounds: int = halving.ROUNDS,
            keep: float = halving.KEEP, seed: int = halving.SEED):
    final = halving.active(list(draws), draws, rounds - 1, keep, hazard_min, burden_min, seed)
    rows = []
    for bid, arms in sorted(draws.items()):
        pre = [arms["PRE"][d] for d in sorted(arms["PRE"]) if d < rounds]
        post = [arms["POST"][d] for d in sorted(arms["POST"]) if d < rounds]
        if not pre or not post:
            continue
        pass_pre = st.mean(bool(r["success"]) for r in pre)
        pass_post = st.mean(bool(r["success"]) for r in post)
        steps_pre = [len(r["steps"]) for r in pre]
        steps_post = [len(r["steps"]) for r in post]
        hazard = round(pass_pre - pass_post, 4)
        burden = round(st.mean(steps_post) - st.mean(steps_pre), 2)
        complete = bid in final and len(pre) == rounds and len(post) == rounds
        channel = None
        if complete:
            channel = "error" if hazard >= hazard_min else "burden" if burden >= burden_min else None
        rows.append({"boundary_id": bid, "task_id": pre[0]["task_id"], "t": pre[0]["t"], "step": pre[0].get("step"),
                     "n_pre": len(pre), "n_post": len(post), "completed_rounds": complete,
                     "pass_pre": round(pass_pre, 4), "pass_post": round(pass_post, 4),
                     "hazard": hazard, "burden": burden, "steps_pre": steps_pre, "steps_post": steps_post,
                     "capped_pre": sum(r["termination_reason"] == "budget_exhausted" for r in pre),
                     "capped_post": sum(r["termination_reason"] == "budget_exhausted" for r in post),
                     "retained": channel is not None, "channel": channel})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hazard-min", type=float, default=0.5)
    ap.add_argument("--burden-min", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=halving.ROUNDS)
    ap.add_argument("--keep", type=float, default=halving.KEEP)
    ap.add_argument("--seed", type=int, default=halving.SEED)
    a = ap.parse_args()
    rows = effects(halving.load_draws(a.rollouts), a.hazard_min, a.burden_min, a.rounds, a.keep, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for e in rows:
            f.write(json.dumps(e) + "\n")
    n_err = sum(e["channel"] == "error" for e in rows)
    n_bur = sum(e["channel"] == "burden" for e in rows)
    n_full = sum(e["completed_rounds"] for e in rows)
    print(f"{len(rows)} boundaries, {n_full} completed {a.rounds} rounds; retained {n_err + n_bur} "
          f"({n_err} error, {n_bur} burden) -> {a.out}")


if __name__ == "__main__":
    main()
