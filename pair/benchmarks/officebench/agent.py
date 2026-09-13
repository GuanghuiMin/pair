"""ACON's OfficeBench agent and environment in the single configuration PAIR uses.

The agent sees ACON's system message (prompts_v3.json), the first user prompt (task and apps), then
one observation plus instruction block per turn, and the summary under <HISTORY_SUMMARY> after the
first prompt when compression is on. Actions are native tool calls to `execute_action(action)`; the
assistant turn stored in history is the model's reasoning plus the call, and the next observation is
fed back as the tool result.

Every run happens inside a container (pair.benchmarks.officebench.run). The working directory
`wd_<tag>__<task>__<subtask>_s<seed>` is rebuilt from a read-only export of the benchmark's task tree
(OB_CANONICAL_TASKS) so a recorded trajectory can be replayed later for continuations.
"""
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ACON_ROOT = Path(os.environ.get("ACON_ROOT", Path(__file__).resolve().parents[4] / "acon"))
OB_ROOT = str(ACON_ROOT / "experiments" / "officebench")
CANONICAL_TASKS = os.environ.get("OB_CANONICAL_TASKS", "")
sys.path.insert(0, str(ACON_ROOT / "src"))
if OB_ROOT not in sys.path:
    sys.path.insert(0, OB_ROOT)          # experiment_config.py lives in the experiments directory

MAX_STEPS = 50
SUMMARY_OPEN, SUMMARY_CLOSE = "\n\n<HISTORY_SUMMARY>\n", "\n</HISTORY_SUMMARY>"
_THINK_ACTION = re.compile(r"<think>(.*?)</think>\s*<action>(.*)</action>", re.DOTALL)
_THINK_BLOCK = re.compile(r"<\s*(?:[\w.-]+:)?think[^>]*>.*?<\s*/\s*(?:[\w.-]+:)?think\s*>", re.DOTALL | re.IGNORECASE)

OB_TOOL = [{"type": "function", "function": {
    "name": "execute_action",
    "parameters": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]},
}}]


def require_env():
    for k in ("PAIR_AGENT_MODEL", "ACON_VLLM_BASE_URL", "ACON_VLLM_API_KEY", "OB_CANONICAL_TASKS"):
        if not os.environ.get(k):
            sys.exit(f"{k} is unset; source env.sh")
    if os.environ.get("ACON_VLLM_STRIP_THINK", "0") != "0":
        sys.exit("ACON_VLLM_STRIP_THINK must be 0 for OfficeBench (the agent's reasoning stays in history)")


def wd_name(tag: str, task: str, subtask: str, seed: int) -> str:
    return f"wd_{tag}__{task}__{subtask}_s{seed}"


def make_workdir(task: str, name: str) -> str:
    """Pristine copy of tasks/<task> from the canonical export, group-writable, fresh mtimes."""
    wd = os.path.join(OB_ROOT, name)
    shutil.rmtree(wd, ignore_errors=True)
    src = os.path.join(CANONICAL_TASKS, task)
    if not os.path.isdir(src):
        raise RuntimeError(f"task {task} missing under OB_CANONICAL_TASKS={CANONICAL_TASKS}")
    dst = os.path.join(wd, "tasks", task)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    now = time.time()
    for root, _dirs, files in os.walk(dst):
        os.chmod(root, 0o775)
        os.utime(root, (now, now))
        for n in files:
            p = os.path.join(root, n)
            os.chmod(p, 0o664)
            os.utime(p, (now, now))
    return f"./{name}"          # relative to OB_ROOT, the working directory ACON's runner uses


def load_task_config(task: str, subtask: str, task_dir: str) -> dict:
    cfg = json.load(open(os.path.join(task_dir, "subtasks", f"{subtask}.json")))
    cfg["testbed_data_path"] = f"{task_dir}/testbed/data"
    cfg["task_dir"] = task_dir
    return cfg


def make_env(task: str, wd: str, task_text: str):
    from productive_agents.env.officebench import OfficeBenchEnv, OfficeBenchEnvConfig
    import productive_agents
    cfg = OfficeBenchEnvConfig(
        local_workdir=wd, task=task_text, task_dir=os.path.join(wd, "tasks", task),
        prompt_file=os.path.join(OB_ROOT, "prompts/prompts_v3.json"),
        app_root_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(productive_agents.__file__))),
                                  "productive_agents/env/officebench/apps"))
    env = OfficeBenchEnv(config=cfg)
    env.reset()
    return env


def make_agent(env, model: str, task_config: dict):
    """The construction of ACON's experiments/officebench/run.py with its default experiment config."""
    from experiment_config import OfficeBenchExperimentConfig
    from productive_agents.agents.officebench import OfficeBenchAgent
    exp = OfficeBenchExperimentConfig()
    exp.prompt_file = os.path.join(OB_ROOT, "prompts/prompts_v3.json")
    exp.model_name = model
    exp.task = env.task
    exp.task_dir = env.task_dir
    exp.local_workdir = env.workdir
    exp.debug_mode = False
    agent = OfficeBenchAgent(model_name=model, key="", env=env, task_config=task_config,
                             llm_cache=None, debug_mode=False, exp_config=exp.to_agent_config(), lora_name=None)
    agent.pair_exp_config = exp
    return agent


def exp_config_dict(agent) -> dict:
    e = agent.pair_exp_config
    return {"exp_id": e.exp_id, "model_name": e.model_name, "max_iter": e.max_iter, "prompt_file": e.prompt_file,
            "use_workflow_memory": e.use_workflow_memory, "use_thinking_tokens": e.use_thinking_tokens,
            "co_config": None, "task": e.task, "task_dir": e.task_dir, "local_workdir": e.local_workdir}


def split_response(text: str):
    """Text-mode fallback when no tool call was emitted: (thinking, action json)."""
    text = text or ""
    m = _THINK_ACTION.match(text.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    return "", (m.group(0).strip() if m else None)


def action_from_arguments(arguments: str):
    """execute_action arguments -> the JSON action string the environment parses."""
    try:
        d = json.loads(arguments or "")
    except ValueError:
        return arguments or ""
    if isinstance(d, dict) and "action" in d and len(d) == 1:
        return d["action"] if isinstance(d["action"], str) else json.dumps(d["action"])
    return json.dumps(d) if isinstance(d, dict) else str(d)


def act(agent, mm, env):
    """One agent turn: model call with the tool, store the assistant turn, step the environment."""
    prompt = mm.get_conversation_history(exclude_system=True)
    message = agent.llm.generate(prompt, tools=OB_TOOL, tool_choice="auto", seed=getattr(agent, "seed", None))
    content = (getattr(message, "content", "") or "") if not isinstance(message, str) else message
    api_summary = ((getattr(message, "reasoning", None) or "") if not isinstance(message, str) else "").strip()
    inband = " ".join(m.strip() for m in _THINK_BLOCK.findall(content)) if content else ""
    inband = re.sub(r"<\s*/?\s*(?:[\w.-]+:)?think[^>]*>", "", inband).strip()
    tool_calls = getattr(message, "tool_calls", None) if not isinstance(message, str) else None
    if tool_calls:
        tc = tool_calls[0]
        arguments = tc.function.arguments or ""
        action = action_from_arguments(arguments)
        serialized = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": arguments}}]
        reasoning = api_summary or inband or _THINK_BLOCK.sub("", content).strip()
        source = "api_summary" if api_summary else ("inband" if inband else ("content" if reasoning else "none"))
        mm.add_assistant_response(reasoning, tool_calls=serialized)
        mode, call_id, n_extra = "toolcall", tc.id, len(tool_calls) - 1
    else:
        thinking, action = split_response(content)
        reasoning = api_summary or thinking
        source = "api_summary" if api_summary else ("inband" if thinking else "none")
        mm.add_assistant_response(content)
        mode, call_id, n_extra = "text_fallback", None, 0
    n_before = len(env.history_log)
    obs, reward, done, _info = env.step(action if action is not None else "")
    executed = env.history_log[n_before][0] if len(env.history_log) > n_before else None
    return {"raw_response": content, "thinking": reasoning, "reasoning_source": source, "api_reasoning_summary": api_summary,
            "action": action, "executed_action": executed, "observation": obs if obs is not None else "",
            "done": bool(done), "reward": reward, "action_mode": mode, "tool_call_id": call_id, "extra_tool_calls": n_extra}


def evaluate(task_config: dict, task_dir: str):
    """The benchmark's own checkers on this run's testbed: every evaluation item must pass. A checker
    exception counts as a failure and is returned as eval_error. Returns (passed, eval_error)."""
    from productive_agents.env.officebench import evaluate as EV
    testbed = os.path.join(task_dir, "testbed")
    # Diff-style checkers compare against a pristine copy at <task_dir>/cache/testbed; materialise it
    # from the canonical export at evaluation time so the agent never sees it.
    cache = os.path.join(task_dir, "cache", "testbed")
    src = os.path.join(CANONICAL_TASKS, os.path.basename(os.path.normpath(task_dir)), "testbed")
    if os.path.isdir(src):
        shutil.rmtree(cache, ignore_errors=True)
        shutil.copytree(src, cache)
    try:
        for item in task_config.get("evaluation", []):
            fn = getattr(EV, item["function"])
            if not fn(testbed, item["args"]):
                return False, None
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
