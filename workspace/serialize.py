"""Provider message history <-> JSON.

``AgentSession.messages`` holds adapter-owned messages. The Anthropic adapter appends
the SDK's content-block objects (``turn.raw``) as-is; the fake adapter embeds
``ToolCall`` dataclasses. Both must round-trip through a JSON column so a resumed
conversation hands the model exactly the history it had.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agentic.agent_llm import ToolCall

_TOOLCALL_TAG = "__toolcall__"


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, ToolCall):
        return {_TOOLCALL_TAG: True, "id": obj.id, "name": obj.name,
                "args": to_jsonable(obj.args)}
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):   # pydantic (Anthropic SDK content blocks)
        # exclude_none: the API rejects some null-valued optional fields on input.
        return to_jsonable(model_dump(mode="json", exclude_none=True))
    if dataclasses.is_dataclass(obj):
        return to_jsonable(dataclasses.asdict(obj))
    return str(obj)


def from_jsonable(obj: Any) -> Any:
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        if obj.get(_TOOLCALL_TAG):
            return ToolCall(id=obj["id"], name=obj["name"], args=obj.get("args") or {})
        return {k: from_jsonable(v) for k, v in obj.items()}
    return obj


def messages_to_json(messages: list) -> list:
    return [to_jsonable(m) for m in messages]


def messages_from_json(data: list) -> list:
    return [from_jsonable(m) for m in data or []]
