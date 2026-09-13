"""Run tau2-bench tasks with or without history compression, then convert into the shared run layout.

    python -m pair.benchmarks.tau2.run --domain retail --split train --out RUNS/tau2/p0_w2048 \
        --method prompts/p0 --window 2048 --seeds 1 --concurrency 12 [--tasks LIST.jsonl]

One run directory per seed, <out>_s<seed>/, holding tau2's native results (tau2_<domain>.json), the
compaction log and the converted run_records.jsonl + trajectories/. The driver runs in the tau2
environment (TAU2_PY, TAU2_HOME); seed k maps to tau2 seed 300 + k - 1.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PAIR_HOME = Path(__file__).resolve().parents[3]
TAU2_PY = os.environ.get("TAU2_PY", sys.executable)
TAU2_HOME = os.environ.get("TAU2_HOME")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--split", default="test", help="tau2 task split: train or test")
    ap.add_argument("--out", required=True, help="run directory prefix; _s<seed> is appended")
    ap.add_argument("--method", default=None, help="compression prompt directory; omit for full context")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--tasks", default=None, help="jsonl task list; its `subtask` field is the tau2 task id")
    ap.add_argument("--task-ids", nargs="*", default=None)
    a = ap.parse_args()
    if not TAU2_HOME:
        sys.exit("TAU2_HOME is unset; source env.sh")
    if a.method and not os.path.isdir(a.method):
        sys.exit(f"prompt directory not found: {a.method}")
    if bool(a.method) != bool(a.window):
        sys.exit("--method and --window go together")
    task_ids = a.task_ids
    if a.tasks:
        task_ids = [json.loads(l)["subtask"] for l in open(a.tasks) if l.strip()]
    for seed in [int(s) for s in a.seeds.split(",")]:
        out = Path(f"{a.out}_s{seed}").resolve()
        out.mkdir(parents=True, exist_ok=True)
        save_to = str(out / f"tau2_{a.domain}.json")
        clog = str(out / f"compactions_{a.domain}.jsonl")
        env = dict(os.environ, PYTHONPATH=f"{PAIR_HOME}:{os.environ.get('PYTHONPATH', '')}")
        if a.method:
            env.update(PAIR_TAU2_METHOD=os.path.abspath(a.method), PAIR_TAU2_WINDOW=str(a.window),
                       PAIR_TAU2_SEED=str(seed), PAIR_TAU2_COMPACTION_LOG=clog)
        cmd = [TAU2_PY, "-m", "pair.benchmarks.tau2.driver", "--domain", a.domain, "--split", a.split,
               "--agent", "pair_agent" if a.method else "llm_agent", "--trials", "1", "--concurrency", str(a.concurrency),
               "--max-steps", str(a.max_steps), "--seed", str(300 + seed - 1), "--save-to", save_to]
        if task_ids:
            cmd += ["--task-ids", *task_ids]
        print(f"[{out.name}] {' '.join(cmd[3:])}", flush=True)
        with open(out / f"{a.domain}.runlog", "a") as lf:
            rc = subprocess.call(cmd, cwd=TAU2_HOME, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if rc != 0:
            sys.exit(f"[{out.name}] tau2 exited with {rc}, see {out / (a.domain + '.runlog')}")
        conv = [sys.executable, "-m", "pair.benchmarks.tau2.convert", "--results", save_to, "--domain", a.domain, "--out", str(out),
                "--seed", str(seed), "--split", a.split, "--method", (Path(a.method).name if a.method else "full"), "--window", str(a.window)]
        if a.method:
            conv += ["--compactions", clog]
        subprocess.check_call(conv, cwd=PAIR_HOME, env=env)


if __name__ == "__main__":
    main()
