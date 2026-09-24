"""tau2-bench's LLMAgent with PAIR's compressor.

The agent is tau2's own (system prompt = the domain policy, native tool calls, user simulator on the
other side). tau2 gives the agent no task text: the dialogue is the task, so the compressor receives
task="" and a prefix-conditioned prompt gets {{ agent_prompt }} = the agent's system prompt. Before every
model call the rendered non-system history is measured; when it exceeds the window, everything but
the most recent turn is summarised and the summary is placed as a second system message under
<HISTORY_SUMMARY>, replacing any previous summary.

Configuration is passed through the environment because tau2 instantiates the agent itself:
    PAIR_TAU2_METHOD          compression prompt directory; unset = no compaction
    PAIR_TAU2_WINDOW          context budget in tokens
    PAIR_TAU2_SEED            compressor seed
    PAIR_TAU2_COMPACTION_LOG  jsonl log with one record per compaction (the input of boundary mining)
    PAIR_TAU2_NO_COMPACT=1    continuations: no further compaction
"""
from __future__ import annotations
import json
import os
import threading
import time

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage, ToolMessage, UserMessage
from tau2.utils.llm_utils import generate

from pair.compressor import Compressor, count_tokens

_LOCK = threading.Lock()
_SUMMARY_OPEN, _SUMMARY_CLOSE = "<HISTORY_SUMMARY>\n", "\n</HISTORY_SUMMARY>"


def render_history(messages) -> str:
    """The history as the compressor sees it: ASSISTANT / USER / tool-result lines."""
    out = []
    for m in messages:
        if isinstance(m, AssistantMessage):
            parts = [m.content] if m.content else []
            parts += [json.dumps({"tool": tc.name, "arguments": tc.arguments}, ensure_ascii=False) for tc in (m.tool_calls or [])]
            out.append("ASSISTANT:\n" + "\n".join(parts) + "\n\n")
        elif isinstance(m, UserMessage):
            out.append("USER:\n" + (m.content or "") + "\n\n")
        elif isinstance(m, ToolMessage):
            out.append("USER:\n" + f"[tool result {m.id}] " + (m.content or "") + "\n\n")
    return "".join(out).strip()


def turn_starts(messages) -> list:
    return [i for i, m in enumerate(messages) if not isinstance(m, ToolMessage)]


class PairLLMAgent(LLMAgent):
    def __init__(self, tools, domain_policy, llm=None, llm_args=None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        method = os.environ.get("PAIR_TAU2_METHOD")
        self.window = int(os.environ.get("PAIR_TAU2_WINDOW", "2048"))
        self.seed = int(os.environ.get("PAIR_TAU2_SEED", "1"))
        self.comp = Compressor(method, self.window, self.seed) if method else None
        self.log_path = os.environ.get("PAIR_TAU2_COMPACTION_LOG")
        self.compaction_enabled = os.environ.get("PAIR_TAU2_NO_COMPACT") != "1"
        self.n_compactions = 0
        self.n_seen = 0                 # non-system messages consumed so far: the boundary index
        self.prev_summary = None
        self.task_id = None
        self.trial = None
        self.preset = None              # (summary, tail messages): start of a POST or t >= 2 PRE continuation

    def _log(self, rec: dict):
        if self.log_path:
            with _LOCK, open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _compact(self, state: LLMAgentState) -> None:
        if not self.compaction_enabled or self.comp is None or len(state.messages) < 2:
            return
        msgs = state.messages
        text = render_history(msgs)
        if not self.comp.check_summarization_needed(text, self.prev_summary):
            return
        starts = turn_starts(msgs)
        if len(starts) < 2:
            return
        cut = starts[-1]                # keep the most recent turn and its tool results raw
        window_msgs, tail = msgs[:cut], msgs[cut:]
        window_text = render_history(window_msgs)
        if not self.comp.agent_prompt:
            self.comp.agent_prompt = state.system_messages[0].content or ""
        t0, prev = time.time(), self.prev_summary
        summary = self.comp.process(task="", history=window_text, prev_history_summary=prev)
        ev = self.comp.events[-1] if self.comp.events else {}
        state.system_messages = [state.system_messages[0], SystemMessage(role="system", content=_SUMMARY_OPEN + summary + _SUMMARY_CLOSE)]
        state.messages = list(tail)
        self.prev_summary = summary
        self.n_compactions += 1
        self._log({"elapsed": round(time.time() - t0, 2), "n_compaction": self.n_compactions, "n_seen": self.n_seen,
                   "task_id": self.task_id, "trial": self.trial, "kind": ev.get("kind"), "summary": summary,
                   "summary_tokens": count_tokens(summary), "input_system": ev.get("input_system"), "input_user": ev.get("input_user"),
                   "agent_visible": ev.get("agent_visible", False), "error": ev.get("error"), "window_tokens": count_tokens(text),
                   "window_text": window_text, "prev_summary": prev, "n_window_msgs": len(window_msgs), "n_tail_msgs": len(tail)})

    def get_init_state(self, message_history=None):
        state = super().get_init_state(message_history)
        self.n_seen = len(state.messages)
        if self.preset is not None:
            summary, tail = self.preset
            state.system_messages = [state.system_messages[0], SystemMessage(role="system", content=_SUMMARY_OPEN + summary + _SUMMARY_CLOSE)]
            state.messages = list(tail)
            self.prev_summary = summary
        return state

    def generate_next_message(self, message, state):
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            self.n_seen += len(message.tool_messages)
        else:
            state.messages.append(message)
            self.n_seen += 1
        self._compact(state)
        assistant_message = generate(model=self.llm, tools=self.tools, messages=state.system_messages + state.messages,
                                     call_name="agent_response", **self.llm_args)
        state.messages.append(assistant_message)
        self.n_seen += 1
        return assistant_message, state


def create_pair_agent(tools, domain_policy, **kwargs):
    ag = PairLLMAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))
    t = kwargs.get("task")
    ag.task_id = getattr(t, "id", None) if t is not None else None
    ag.trial = kwargs.get("trial")
    return ag


def register():
    from tau2.registry import registry
    if "pair_agent" not in getattr(registry, "_agent_factories", {}):
        registry.register_agent_factory(create_pair_agent, "pair_agent")
