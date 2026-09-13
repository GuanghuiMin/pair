"""Single chat-completion call used by the compressor and the optimizer.

Endpoint, key and model come from PAIR_COMPRESSOR_BASE_URL / PAIR_COMPRESSOR_API_KEY /
PAIR_COMPRESSOR_MODEL (see env.example.sh).
"""
import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI

_THINK = re.compile(r"<think>.*?</think>", re.S)
_client = None


def client(timeout: int = 600) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=os.environ["PAIR_COMPRESSOR_BASE_URL"],
                         api_key=os.environ["PAIR_COMPRESSOR_API_KEY"], timeout=timeout)
    return _client


def model_name() -> str:
    return os.environ["PAIR_COMPRESSOR_MODEL"]


def completion_kwargs(max_tokens: int) -> dict:
    """GPT reasoning models take max_completion_tokens and an optional reasoning effort; others take max_tokens."""
    if model_name().startswith("gpt-"):
        kw = {"max_completion_tokens": max_tokens}
        if os.environ.get("PAIR_COMPRESSOR_REASONING"):
            kw["reasoning_effort"] = os.environ["PAIR_COMPRESSOR_REASONING"]
        return kw
    return {"max_tokens": max_tokens}


@dataclass
class ChatResult:
    text: str
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def chat(system: str, user: str, max_tokens: int, seed: int) -> ChatResult:
    kw = {"seed": seed, **completion_kwargs(max_tokens)}
    if "max_tokens" in kw:
        kw["temperature"] = 0.0
    t0 = time.time()
    try:
        r = client().chat.completions.create(
            model=model_name(), messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        raw = r.choices[0].message.content or ""
        return ChatResult(_THINK.sub("", raw).strip(), time.time() - t0,
                          getattr(r.usage, "prompt_tokens", 0), getattr(r.usage, "completion_tokens", 0))
    except Exception as e:  # noqa: BLE001
        return ChatResult("", time.time() - t0, error=str(e))
