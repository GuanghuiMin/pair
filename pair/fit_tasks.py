"""Step 4 (fit set): the training tasks with the longest compressed trajectories under the incumbent.

    python -m pair.fit_tasks --run RUNS/appworld/p0_w4096_s1 --n 12 --out data/tasklists/appworld_fit12.jsonl

Ranks the run's records by num_steps (ties by task id) and writes the top n as a task list in the
same shape the runners read ({task_id, split} plus task/subtask when the benchmark has them).
"""
import argparse
import json
from pathlib import Path

KEYS = ("task_id", "task", "subtask", "split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory with run_records.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(Path(a.run) / "run_records.jsonl") if l.strip()]
    recs.sort(key=lambda r: (-r["num_steps"], r["task_id"]))
    top = recs[:a.n]
    with open(a.out, "w") as f:
        for r in top:
            f.write(json.dumps({k: r[k] for k in KEYS if k in r}) + "\n")
    print(f"{len(top)} tasks -> {a.out}: " + ", ".join(f"{r['task_id']}({r['num_steps']})" for r in top))


if __name__ == "__main__":
    main()
