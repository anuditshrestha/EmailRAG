"""Thin LLM interface, Anthropic implementation, and a tolerant JSON extractor."""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Protocol


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int = 2000) -> str: ...


class AnthropicLLM:
    """Anthropic Messages API client. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("Install the SDK first: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)  # reads ANTHROPIC_API_KEY if None
        self.model = model or os.environ.get("EMAILPIPE_MODEL", "claude-sonnet-5-5")
        self.calls = 0

    def complete(self, system: str, user: str, *, max_tokens: int = 2000) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.calls += 1
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def make_llm(kind: str = "auto", model: Optional[str] = None) -> LLM:
    """kind: 'anthropic' | 'mock' | 'auto' (anthropic if ANTHROPIC_API_KEY is set, else mock)."""
    if kind == "mock" or (kind == "auto" and not os.environ.get("ANTHROPIC_API_KEY")):
        from .mock_llm import MockLLM

        return MockLLM()
    return AnthropicLLM(model=model)


def extract_json(text: str):
    """Pull the first JSON value out of a model reply (tolerates ``` fences and chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise LLMError("no JSON found in model output")
