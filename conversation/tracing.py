"""
Langfuse tracing wrapper — no-op safe if keys are not configured.

One SessionTrace per ConversationFlow. Each LLM call becomes a generation
span. The trace is enriched with pair/mode/direction once the view is parsed.
"""

from __future__ import annotations

import os
from typing import Any

_langfuse = None


def _init_client():
    global _langfuse
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if pk and sk:
        try:
            from langfuse import Langfuse
            _langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
        except Exception:
            pass


_init_client()


class _NoOpGeneration:
    def end(self, output: str) -> None:
        pass


class _NoOpTrace:
    def update(self, **kwargs) -> None:
        pass

    def generation(self, **kwargs) -> _NoOpGeneration:
        return _NoOpGeneration()


class SessionTrace:
    """One trace per conversation session."""

    def __init__(self, session_id: str | None = None):
        self._trace: Any = _NoOpTrace()
        if _langfuse is not None:
            try:
                self._trace = _langfuse.trace(
                    name="macrotool-session",
                    session_id=session_id,
                )
            except Exception:
                pass

    def tag_view(self, pair: str, mode: str, direction: str) -> None:
        try:
            self._trace.update(
                tags=[pair, mode],
                metadata={"pair": pair, "mode": mode, "direction": direction},
            )
        except Exception:
            pass

    def generation(self, name: str, model: str, input: Any) -> Any:
        try:
            return self._trace.generation(name=name, model=model, input=input)
        except Exception:
            return _NoOpGeneration()
