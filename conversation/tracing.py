"""
Langfuse tracing wrapper — no-op safe if keys are not configured.

One SessionTrace per ConversationFlow. Each LLM call becomes a generation
span. The trace is enriched with pair/mode/direction once the view is parsed.
"""

from __future__ import annotations

import os
from typing import Any

_langfuse = None
_init_error: str | None = None


def _init_client() -> None:
    global _langfuse, _init_error
    _init_error = None
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not pk or not sk:
        _init_error = "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set"
        return
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
        _init_error = None
    except Exception as e:
        _langfuse = None
        _init_error = str(e)


def init_status() -> tuple[bool, str | None]:
    """Return (connected, error_message). For display in the UI."""
    return (_langfuse is not None, _init_error)


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

    def flush(self) -> None:
        """Force-flush buffered events to Langfuse. Call after each LLM response."""
        if _langfuse is not None:
            try:
                _langfuse.flush()
            except Exception:
                pass
