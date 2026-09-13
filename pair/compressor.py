"""Summary compressor driven by a prompt directory.

A compression prompt is a directory with three Jinja templates:
    system_prompt.jinja    the compressor's system message
    first_summary.jinja    used when no summary exists yet ({{ history }}, {{ task }})
    update_summary.jinja   used afterwards ({{ history }}, {{ prev_summary }}, {{ task }})
A prefix-conditioned prompt additionally references {{ agent_prompt }}, which is filled with the fixed
prefix the agent itself received (its system prompt and first user message); such a prompt gets no
separate task preamble. The compressor exposes the interface ACON's MemoryManager expects
(check_summarization_needed / process) so it plugs into the AppWorld and OfficeBench agents unchanged.
"""
from pathlib import Path

import jinja2
import tiktoken

from pair import llm

_ENC = tiktoken.encoding_for_model("gpt-4o")

TASK_PREAMBLE = (
    "The user's ORIGINAL TASK is given verbatim below. Treat it as the definitive goal: "
    "do NOT guess, infer, paraphrase, narrow, or broaden it, and preserve any scope "
    "qualifiers exactly. (In real deployment the summarizer sees this task; it is "
    "provided here for the same reason.)\n<TASK>\n{task}\n</TASK>\n\n"
)
SUMMARY_MARKER = "# History Summary"
MAX_SUMMARY_TOKENS = 8192


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text or "", disallowed_special=()))


class Compressor:
    def __init__(self, prompt_dir: str, window: int, seed: int):
        d = Path(prompt_dir)
        self.name = d.name
        self.window = window
        self.seed = seed
        self.system = jinja2.Template((d / "system_prompt.jinja").read_text()).render()
        first_src, update_src = (d / "first_summary.jinja").read_text(), (d / "update_summary.jinja").read_text()
        self.first = jinja2.Template(first_src)
        self.update = jinja2.Template(update_src)
        self.agent_visible = "agent_prompt" in first_src or "agent_prompt" in update_src
        self.agent_prompt = None          # set by the runner for prefix-conditioned prompts
        self.events: list[dict] = []      # one record per compaction (summary, error, exact input)
        self.history: list = []           # MemoryManager compatibility

    def check_summarization_needed(self, history_text: str, prev_history_summary=None) -> bool:
        text = f"{prev_history_summary}\n{history_text}" if prev_history_summary else history_text
        return count_tokens(text) > self.window

    def process(self, task, history, prev_history_summary=None, **_) -> str:
        prev = prev_history_summary or ""
        kw = {"history": history, "prev_summary": prev, "task": task}
        if self.agent_visible:
            assert self.agent_prompt, "prefix-conditioned prompt: set compressor.agent_prompt first"
            kw["agent_prompt"] = self.agent_prompt
        user = self.update.render(**kw) if prev else self.first.render(**kw)
        if task and not self.agent_visible:
            user = TASK_PREAMBLE.format(task=task) + user
        res = llm.chat(self.system, user, MAX_SUMMARY_TOKENS, self.seed)
        text = res.text
        summary = text.split(SUMMARY_MARKER, 1)[1].strip() if SUMMARY_MARKER in text else text.strip()
        if not summary:                   # failed call: keep the agent running on what it had
            summary = prev or history
        self.events.append({"kind": "update" if prev else "first", "summary": summary, "error": res.error,
                            "elapsed_s": round(res.elapsed_s, 2), "input_system": self.system, "input_user": user,
                            "agent_visible": self.agent_visible})
        return summary

    def dump_history(self, output_dir):
        pass
