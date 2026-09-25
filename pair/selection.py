"""Step 4 (selection): pick the candidate that passes the most fit tasks, ties by fewer steps.

    python -m pair.selection --candidate c1=RUNS/fit_c1_s1 --candidate c2=RUNS/fit_c2_s1 ...

The pipeline runs every candidate once (seed 1) on the fit tasks; the criterion is the number of fit tasks it
passes, ties broken by the lower mean number of interaction steps. A candidate may also name several runs
(NAME=RUN_DIR,RUN_DIR,...); the count is then Pass^k, the tasks that pass in every run.
"""
import argparse
import json
from pathlib import Path


def load(run_dir: str) -> dict:
    return {r["task_id"]: r for r in map(json.loads, (l for l in open(Path(run_dir) / "run_records.jsonl") if l.strip()))}


def score(run_dirs: list[str]) -> dict:
    runs = [load(d) for d in run_dirs]
    tasks = sorted(set.intersection(*[set(r) for r in runs]))
    passes = [[bool(r[t]["success"]) for r in runs] for t in tasks]
    n_runs = len(tasks) * len(runs)
    return {"n_tasks": len(tasks), "k": len(runs),
            "pass_k": sum(all(p) for p in passes),
            "pass_rate": sum(map(sum, passes)) / n_runs,
            "steps": sum(r[t]["num_steps"] for r in runs for t in tasks) / n_runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True, metavar="NAME=RUN_DIR[,RUN_DIR...]")
    a = ap.parse_args()
    rows = []
    for spec in a.candidate:
        name, dirs = spec.split("=", 1)
        rows.append((name, score(dirs.split(","))))
    rows.sort(key=lambda x: (-x[1]["pass_k"], x[1]["steps"]))
    k = rows[0][1]["k"]
    print(f"{'candidate':<24} {'Pass^' + str(k):>8} {'pass_rate':>10} {'steps':>7}  tasks")
    for name, s in rows:
        print(f"{name:<24} {s['pass_k']:>8} {100 * s['pass_rate']:>9.1f}% {s['steps']:>7.2f}  {s['n_tasks']}")
    print(f"selected: {rows[0][0]}")


if __name__ == "__main__":
    main()
