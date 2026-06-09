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
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-sonnet-4-6"   # workhorse; Opus 4.8 one flag away
MAX_TOKENS = 2048


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


class ToolLLM(Protocol):
    def create(self, messages: list[dict], system: str, tools: list[dict]) -> LLMTurn: ...
    def format_user(self, text: str) -> dict: ...
    def format_assistant(self, turn: LLMTurn) -> dict: ...
    def format_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> dict: ...


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicToolLLM:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        anthropic = importlib.import_module("anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def create(self, messages: list[dict], system: str, tools: list[dict]) -> LLMTurn:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools,                       # our schema dicts == Anthropic's shape
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, args=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        return LLMTurn(text=text, tool_calls=calls, stop_reason=resp.stop_reason, raw=resp.content)

    def format_user(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def format_assistant(self, turn: LLMTurn) -> dict:
        return {"role": "assistant", "content": turn.raw}

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

    def format_tool_results(self, results: list[tuple[ToolCall, str, bool]]) -> dict:
        return {
            "role": "tool",
            "content": [
                {"tool_use_id": c.id, "content": text, "is_error": err}
                for c, text, err in results
            ],
        }
