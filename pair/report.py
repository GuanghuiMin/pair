"""Pass rate, Pass^k and Pass@k of evaluation runs.

    python -m pair.report --arm full=RUNS/full_s1,RUNS/full_s2,RUNS/full_s3 --arm pair=RUNS/pair_s1,... [--out summary.md]

Each arm names its k independent runs (directories with run_records.jsonl). Only tasks present in
every run of an arm are counted. Pass^k = tasks passing in all runs, Pass@k = in at least one;
pass_rate, steps and compactions are means over all task-run pairs.
"""
import argparse
import json
from pathlib import Path


def load(run_dir: str) -> dict:
    return {r["task_id"]: r for r in map(json.loads, (l for l in open(Path(run_dir) / "run_records.jsonl") if l.strip()))}


def row(name: str, run_dirs: list[str]) -> dict:
    runs = [load(d) for d in run_dirs]
    tasks = sorted(set.intersection(*[set(r) for r in runs]))
    passes = [[bool(r[t]["success"]) for r in runs] for t in tasks]
    n = len(tasks) * len(runs)
    mean = lambda key: sum(float(r[t].get(key) or 0) for r in runs for t in tasks) / n
    return {"arm": name, "n": len(tasks), "k": len(runs),
            "pass_rate": 100 * sum(map(sum, passes)) / n,
            "pass_k": 100 * sum(all(p) for p in passes) / len(tasks),
            "pass_at_k": 100 * sum(any(p) for p in passes) / len(tasks),
            "steps": mean("num_steps"), "compactions": mean("n_compressions")}


def table(rows: list[dict]) -> str:
    k = rows[0]["k"]
    head = f"| arm | tasks | runs | pass_rate | Pass^{k} | Pass@{k} | steps | compactions |\n|---|---|---|---|---|---|---|---|\n"
    return head + "".join(f"| {r['arm']} | {r['n']} | {r['k']} | {r['pass_rate']:.1f}% | {r['pass_k']:.1f}% | "
                          f"{r['pass_at_k']:.1f}% | {r['steps']:.1f} | {r['compactions']:.2f} |\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=RUN_DIR[,RUN_DIR...]")
    ap.add_argument("--out", default=None, help="write the table as markdown")
    a = ap.parse_args()
    rows = [row(name, dirs.split(",")) for name, dirs in (s.split("=", 1) for s in a.arm)]
    doc = table(rows)
    print(doc)
    if a.out:
        Path(a.out).write_text(doc)


if __name__ == "__main__":
    main()
