import json
from pathlib import Path

from pair.benchmarks.officebench.rollouts import compactions, step_events


PREFIX_NOTE = ("At run time the jinja variable {{ agent_prompt }} in both templates is replaced by the exact prefix "
               "the downstream agent received for that task: its system message (operating instructions, date, user, "
               "apps, testbed path) followed by its first user message (the task and available apps). Each counterexample "
               "carries that prefix for its own task as agent_prompt_seen_by_compressor; the date, user, testbed path "
               "and task differ from task to task, so any rule you write must hold for every task.")


def _trajectory(b: dict, cache: dict):
    if b["trajectory"] not in cache:
        cache[b["trajectory"]] = json.load(open(b["trajectory"]))
    return cache[b["trajectory"]]


def summaries(b: dict, cache: dict):
    bnds = compactions(_trajectory(b, cache)["events"])
    return (bnds[b["t"] - 2][1] if b["t"] > 1 else None), bnds[b["t"] - 1][1]


def segment(b: dict, cache: dict, obs_chars):
    from pair.evidence import cut
    num = step_events(_trajectory(b, cache)["events"])
    lo, hi = b["raw_suffix"]
    return [{"step": i, "reasoning": num[i].get("thinking") or "", "action": num[i].get("executed_action") or num[i].get("action") or "",
             "observation": cut(num[i].get("observation"), obs_chars)} for i in range(lo, hi)]


def agent_prefix(run_dir: Path, task_id: str) -> str:
    t = json.load(open(run_dir / "trajectories" / f"{task_id}.json"))
    return t["system_message"] + "\n\n" + t["first_prompt"]


def leak_probe(b: dict, cache: dict) -> str:
    return _trajectory(b, cache)["system_message"][200:500]


def step(s: dict) -> dict:
    return {"reasoning": s.get("reasoning") or "", "action": s.get("executed_action") or s.get("action") or "",
            "observation": s.get("observation")}
