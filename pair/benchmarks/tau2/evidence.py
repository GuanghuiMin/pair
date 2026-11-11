import json
from pathlib import Path


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


def agent_prefix(run_dir: Path, task_id: str) -> str:
    return json.load(open(run_dir / "trajectories" / f"{task_id}.json"))["system_message"]


def leak_probe(b: dict, cache: dict) -> str:
    return (_simulation(b, cache).get("policy") or "")[200:500]


def step(s: dict) -> dict:
    return {"action": s.get("executed_action") or s.get("action") or "", "observation": s.get("observation")}
