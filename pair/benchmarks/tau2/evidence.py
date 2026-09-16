"""tau2-specific pieces of a counterexample (see pair.evidence). The raw segment is a run of dialogue
messages rendered exactly as the compressor saw them."""
import json
from pathlib import Path

LOOKUP_PREFIXES = ("find_", "get_", "list_", "calculate")

PREFIX_NOTE = ("At run time the jinja variable {{ agent_prompt }} in both templates is replaced by the exact fixed prefix "
               "the downstream agent received: its system prompt (the customer-service domain policy and tool rules). "
               "In this benchmark the agent receives no task text — the customer's dialogue is the task and is part of "
               "the history you summarise, never of the prefix. Each counterexample carries that prefix as "
               "agent_prompt_seen_by_compressor; it is the same for every task in the domain, so any rule you write "
               "must hold for every conversation.")


def render_message(m: dict):
    role = m.get("role")
    if role == "assistant":
        parts = [m["content"]] if m.get("content") else []
        parts += [json.dumps({"tool": tc["name"], "arguments": tc["arguments"]}, ensure_ascii=False) for tc in (m.get("tool_calls") or [])]
        return {"speaker": "ASSISTANT", "text": "\n".join(parts)}
    if role == "user":
        return {"speaker": "USER", "text": m.get("content") or ""}
    if role == "tool":
        return {"speaker": "TOOL", "text": f"[tool result {m.get('id')}] " + (m.get("content") or "")}
    return None


def _simulation(b: dict, cache: dict) -> dict:
    res = cache.setdefault(b["trajectory"], json.load(open(b["trajectory"])))
    return next(s for s in res["simulations"] if str(s["task_id"]) == b["tau2_task_id"])


def summaries(b: dict, cache: dict):
    return b.get("prev_summary"), b["summary"]


def segment(b: dict, cache: dict, obs_chars):
    from pair.evidence import cut
    sim = _simulation(b, cache)
    lo, hi = b["raw_suffix"]
    rows = []
    for i in range(lo, hi):
        rm = render_message(sim["messages"][i])
        if rm is not None:
            rows.append({"msg": i, "speaker": rm["speaker"], "text": cut(rm["text"], obs_chars) if rm["speaker"] == "TOOL" else rm["text"]})
    return rows


def terminal(r: dict) -> str:
    return str(r["termination_reason"]).split(".")[-1].lower()


def facts(r: dict) -> dict:
    """Re-work in a continuation: lookups, repeated identical tool calls, tool errors, user-facing turns."""
    redo, seen, last_msg = {}, set(), None
    for s in r["steps"]:
        act = s.get("executed_action") or s.get("action") or ""
        obs = s.get("observation") or ""
        try:
            calls = json.loads(act)
            calls = calls if isinstance(calls, list) else None
        except ValueError:
            calls = None
        if calls is None:
            redo["user_turns"] = redo.get("user_turns", 0) + 1
            last_msg = act
            continue
        for c in calls:
            key = json.dumps(c, sort_keys=True)
            if key in seen:
                redo["repeated_calls"] = redo.get("repeated_calls", 0) + 1
            seen.add(key)
            if str(c.get("name", "")).startswith(LOOKUP_PREFIXES):
                redo["lookups"] = redo.get("lookups", 0) + 1
        if "[tool] Error" in obs or obs.startswith("Error"):
            redo["tool_errors"] = redo.get("tool_errors", 0) + 1
    from pair.evidence import cut
    return {"final_submission": cut(last_msg, 300) if last_msg else None, "redo_counts": redo}


def agent_prefix(run_dir: Path, task_id: str) -> str:
    return json.load(open(run_dir / "trajectories" / f"{task_id}.json"))["system_message"]


def leak_probe(b: dict, cache: dict) -> str:
    return (_simulation(b, cache).get("policy") or "")[200:500]
