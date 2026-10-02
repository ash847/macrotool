"""Provider-neutral tool-calling seam.

``agent_flow`` talks only to the ``ToolLLM`` interface and to opaque,
adapter-owned message dicts — it never branches on provider. Each adapter owns:
  (a) tool-schema translation,
  (b) parsing the model turn into a normalized ``LLMTurn`` (+ ``ToolCall`` list),
  (c) building the provider-native assistant / tool-result / user messages.

Seam invariant: a ``ToolCall`` always carries an ``id``. Anthropic and OpenAI
supply one; the future Gemini adapter must manufacture one and map it back to the
function name (Gemini matches calls/results by name, not id).

Adapters: ``AnthropicToolLLM`` (built + tested first) and ``FakeToolLLM`` (scripted,
for no-API-burn tests). OpenAI is the next drop-in; Gemini after.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 9000   # raised from 2048 alongside _TOP_N=5 (agentic/render.py) — 5
                    # structures at the PM's usual table+prose+advantages/drawbacks
                    # depth was pushing observed replies close to the old 2048 cap


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMTurn:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    raw: Any = None        # provider-native assistant content, for append-back
    metrics: dict | None = field(default=None, kw_only=True)


class ToolLLM(Protocol):
    def create(self, messages: list[dict], system: str, tools: list[dict]) -> LLMTurn: ...
    def format_user(self, text: str) -> dict: ...
    def format_assistant(self, turn: LLMTurn) -> dict: ...
    def format_text_reply(self, text: str) -> dict: ...
    def format_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> dict: ...


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicToolLLM:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, *, cache_enabled: bool = True):
        anthropic = importlib.import_module("anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cache_enabled = cache_enabled

    def create(self, messages: list[dict], system: str, tools: list[dict]) -> LLMTurn:
        caching = getattr(self, "cache_enabled", True)
        request_system = system
        options = {}
        if caching:
            options["cache_control"] = {"type": "ephemeral"}
            if system:
                request_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        started = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=request_system,
            tools=tools,                       # our schema dicts == Anthropic's shape
            messages=messages,
            **options,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, args=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        usage = getattr(resp, "usage", None)
        metrics = {
            "model": getattr(resp, "model", self.model),
            "request_id": getattr(resp, "_request_id", None),
            "stop_reason": resp.stop_reason,
            "latency_ms": latency_ms,
            "cache_enabled": caching,
            "cache_ttl": "5m" if caching else None,
        }
        for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"):
            metrics[name] = getattr(usage, name, None)
        return LLMTurn(text=text, tool_calls=calls, stop_reason=resp.stop_reason, raw=resp.content, metrics=metrics)

    def format_user(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def format_assistant(self, turn: LLMTurn) -> dict:
        return {"role": "assistant", "content": turn.raw}

    def format_text_reply(self, text: str) -> dict:
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    def format_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> dict:
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": content,
                "is_error": is_error,
            }
            for call, content, is_error in results
        ]
        return {"role": "user", "content": blocks}


# ---------------------------------------------------------------------------
# Fake adapter — scripted, for deterministic tests (no API)
# ---------------------------------------------------------------------------

@dataclass
class FakeToolLLM:
    """Returns a queued script of LLMTurns. ``create`` records each call's inputs
    in ``seen`` so tests can assert on prompts/messages/tools.
    """

    script: list[LLMTurn]
    seen: list[dict] = field(default_factory=list)
    model: str = "fake"

    def create(self, messages: list[dict], system: str, tools: list[dict]) -> LLMTurn:
        self.seen.append({"messages": list(messages), "system": system, "tools": tools})
        if not self.script:
            raise AssertionError("FakeToolLLM script exhausted")
        return self.script.pop(0)

    def format_user(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def format_assistant(self, turn: LLMTurn) -> dict:
        return {"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls}

    def format_text_reply(self, text: str) -> dict:
        return {"role": "assistant", "content": text}

    def format_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> dict:
        return {
            "role": "tool",
            "content": [
                {"tool_use_id": c.id, "content": text, "is_error": err}
                for c, text, err in results
            ],
        }
