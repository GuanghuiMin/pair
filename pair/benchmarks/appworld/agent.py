"""ACON's AppWorld agent and environment in the single configuration PAIR uses.

The agent sees ACON's system message, the AppWorld task prompt, the summary under <HISTORY_SUMMARY>
when compression is on, and one tool `execute_python(code)` declared without a description. The
patched ACON checkout is located through ACON_ROOT (see third_party/README.md).
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ACON_ROOT = Path(os.environ.get("ACON_ROOT", Path(__file__).resolve().parents[4] / "acon"))
APPWORLD_DIR = ACON_ROOT / "experiments" / "appworld"
sys.path.insert(0, str(ACON_ROOT / "src"))

os.environ["ACON_APPWORLD_ACTION_MODE"] = "toolcall"
os.environ["ACON_APPWORLD_TOOLDESC_NEUTRAL"] = "empty"
os.environ["ACON_APPWORLD_TOOLCALL_SUFFIX"] = "0"

MAX_STEPS = 50
AGENT_MAX_TOKENS = 2048
SUMMARY_OPEN, SUMMARY_CLOSE = "\n\n<HISTORY_SUMMARY>\n", "\n</HISTORY_SUMMARY>"


def require_env():
    for k in ("PAIR_AGENT_MODEL", "ACON_VLLM_BASE_URL", "ACON_VLLM_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"{k} is unset; source env.sh")


def make_env(task_id: str, split: str, experiment: str, max_interactions: int):
    os.chdir(APPWORLD_DIR)
    from productive_agents.env.appworld import AppWorldEnv, AppWorldEnvConfig
    env = AppWorldEnv(config=AppWorldEnvConfig(
        experiment_name=experiment, max_interactions=max_interactions,
        dataset_split=split, verbose=False, debug_mode=False))
    env.reset(task_id=task_id)
    return env


def make_agent(env, model: str, temperature: float, seed: int):
    from productive_agents.agents.appworld import AppWorldAgent, AppWorldAgentConfig
    now = datetime.now()
    return AppWorldAgent(
        model_name=model, key="", env=env, debug_mode=False,
        task_config={"task_id": env.task_id, "split": env.config.dataset_split,
                     "experiment_name": env.experiment_name, "username": "user",
                     "date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A"),
                     "time": now.strftime("%H:%M:%S")},
        exp_config=AppWorldAgentConfig(
            model_name=model, temperature=temperature, max_tokens=AGENT_MAX_TOKENS,
            use_thinking=True, prompt_file=None, co_config=None, verbose=False,
            debug_mode=False, extra_config={"seed": seed}))


def add_turn(mm, turn: dict, turn_id: str):
    """Replay one recorded step into the agent's history as a tool call and its result."""
    call = {"id": turn_id, "type": "function", "function": {
        "name": "execute_python", "arguments": json.dumps({"code": turn.get("code") or ""})}}
    mm.add_assistant_response((turn.get("reasoning") or "").strip(), tool_calls=[call])
    mm.add_tool_response(turn_id, turn.get("observation") or "")


def act(agent, mm, env):
    """One agent step: (code, observation, done, info, meta)."""
    out = agent.forward(mm.get_conversation_history(exclude_system=True))
    meta = out.metadata or {}
    code = out.action or ""
    mm.add_assistant_response(meta.get("reasoning") or "", tool_calls=meta.get("tool_calls"))
    obs, _, done, info = env.step(code)
    if not done:
        if meta.get("tool_call_id"):
            mm.add_tool_response(meta["tool_call_id"], obs or "")
        else:
            mm.add_user_prompt(obs or "")
    return code, obs or "", done, info or {}, meta


def evaluate(env) -> dict:
    """AppWorld's evaluator on the live environment; every test is listed."""
    t = env.world.evaluate(suppress_errors=True)
    req = lambda x: (x.get("requirement") if isinstance(x, dict) else getattr(x, "requirement", "")) or ""
    tests = ([{"status": "pass", "requirement": req(p).strip()} for p in (getattr(t, "passes", None) or [])]
             + [{"status": "fail", "requirement": req(f).strip()} for f in (getattr(t, "failures", None) or [])])
    n = int(getattr(t, "num_tests", 0) or 0)
    pc = int(getattr(t, "pass_count", 0) or 0)
    return {"success": bool(getattr(t, "success", False)), "pass_count": pc, "num_tests": n,
            "score": pc / n if n else float(bool(getattr(t, "success", False))), "tests": tests}


def close(env):
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
