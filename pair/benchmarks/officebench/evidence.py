"""OfficeBench-specific pieces of a counterexample (see pair.evidence)."""
import json
import re
from pathlib import Path

from pair.benchmarks.officebench.rollouts import compactions, step_events

READ_OR_LIST = re.compile(r'"action"\s*:\s*"(read_file|list_files|read_email|list_emails|read_event|list_events)"|\b(ls|find|cat|head|tail)\b')

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


def terminal(r: dict) -> str:
    return r["termination_reason"]


def facts(r: dict) -> dict:
    """Re-work in a continuation: app switches, read/list actions, rejected actions; the closing finish_task."""
    redo, subs = {}, []
    for s in r["steps"]:
        act = s.get("executed_action") or s.get("action") or ""
        obs = s.get("observation") or ""
        try:
            j = json.loads(act)
        except ValueError:
            j = {}
        if isinstance(j, dict) and j.get("action") == "switch_app":
            redo["switch_app"] = redo.get("switch_app", 0) + 1
        elif READ_OR_LIST.search(act):
            redo["read_or_list"] = redo.get("read_or_list", 0) + 1
        if obs.startswith("Error") or obs.startswith("Command failed"):
            redo["rejected_actions"] = redo.get("rejected_actions", 0) + 1
        if isinstance(j, dict) and j.get("action") == "finish_task":
            subs.append(act[:120])
    return {"final_submission": subs[-1] if subs else None, "redo_counts": redo}


def agent_prefix(run_dir: Path, task_id: str) -> str:
    t = json.load(open(run_dir / "trajectories" / f"{task_id}.json"))
    return t["system_message"] + "\n\n" + t["first_prompt"]


def leak_probe(b: dict, cache: dict) -> str:
    return _trajectory(b, cache)["system_message"][200:500]
