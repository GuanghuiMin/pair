import json
from pathlib import Path

from pair.benchmarks.appworld.agent import APPWORLD_DIR
from pair.benchmarks.appworld.boundaries import compactions, steps


PREFIX_NOTE = ("At run time the jinja variable {{ agent_prompt }} in both templates is replaced by the exact first "
               "user message the downstream agent received for that task (its operating instructions, the "
               "supervisor's identity, and the task). Each counterexample carries that message for its own task as "
               "agent_prompt_seen_by_compressor; the identity and task fields differ from task to task, so any rule "
               "you write must hold for every task.")


def _trajectory(b: dict, cache: dict):
    if b["trajectory"] not in cache:
        ev = json.load(open(b["trajectory"]))["events"]
        cache[b["trajectory"]] = (compactions(ev), steps(ev))
    return cache[b["trajectory"]]


def summaries(b: dict, cache: dict):
    bnds, _ = _trajectory(b, cache)
    return (bnds[b["t"] - 2][1] if b["t"] > 1 else None), bnds[b["t"] - 1][1]


def segment(b: dict, cache: dict, obs_chars):
    from pair.evidence import cut
    _, num = _trajectory(b, cache)
    lo, hi = b["raw_suffix"]
    return [{"step": i, "reasoning": num[i].get("reasoning") or "", "code": num[i]["code"],
             "observation": cut(num[i]["observation"], obs_chars)} for i in range(lo, hi)]


def agent_prefix(run_dir: Path, task_id: str) -> str:
    return json.load(open(run_dir / "trajectories" / f"{task_id}.json"))["first_prompt"]


def leak_probe(b: dict, cache: dict) -> str:
    return (APPWORLD_DIR / "prompts" / "prompt_v1.jinja").read_text()[2000:2300]


def step(s: dict) -> dict:
    return {"reasoning": s.get("reasoning") or "", "code": s.get("code") or "", "observation": s.get("observation")}
