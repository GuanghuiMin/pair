"""Step 1 (boundaries): list every compaction of a recorded AppWorld run.

    python -m pair.benchmarks.appworld.boundaries --run RUNS/appworld/p0_w4096_s1 --out B

A boundary is one compaction event of one trajectory. With one retained raw turn, a compaction that
fires before step N summarises the steps since the previous boundary up to N-2 and keeps step N-1
raw; `raw_suffix` records that range. Every compaction of every trajectory is kept, whatever the
trajectory's final outcome.
"""
import argparse
import json
from pathlib import Path


def compactions(events):
    return [(int(e["compression_before_step"]), e["summary"]) for e in events
            if "compression_before_step" in e and e.get("summary")]


def steps(events):
    return {int(e["step"]): e for e in events if "step" in e and e.get("code")}


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
        ev = json.load(open(path))["events"]
        bnds, num = compactions(ev), steps(ev)
        for t, (step, _) in enumerate(bnds, 1):
            lo = 1 if t == 1 else bnds[t - 2][0] - 1     # first raw step after the previous boundary
            hi = step - 1                                # the turn this compaction keeps raw
            if any(i not in num for i in range(1, hi + 1)):
                skipped.append((rec["task_id"], f"t{t}: missing steps before {step}"))
                continue
            rows.append({"boundary_id": f"{run_dir.name}::{rec['task_id']}#t{t}", "trajectory": str(path),
                         "task_id": rec["task_id"], "split": rec["split"], "t": t, "n_boundaries": len(bnds),
                         "step": step, "raw_suffix": [lo, hi],
                         "trajectory_success": rec["success"], "trajectory_steps": rec["num_steps"]})
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="run directory (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for r in a.run:
        x, y = mine(Path(r).resolve())
        rows += x
        skipped += y
    with open(out / "boundaries.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} boundaries from {len({r['trajectory'] for r in rows})} trajectories; skipped {len(skipped)}")
    for s in skipped[:10]:
        print("  skipped", *s)


if __name__ == "__main__":
    main()
