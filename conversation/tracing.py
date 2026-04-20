"""
Langfuse tracing — SDK v3, no-op safe if keys are not configured.

One trace per ConversationFlow session. Each LLM call is a generation span
nested inside the session trace. Flush is called after each generation to
ensure events are sent in Streamlit's short-lived request model.

Required env vars (or Streamlit secrets):
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  LANGFUSE_BASE_URL  (default: https://cloud.langfuse.com)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

_connected: bool = False
_init_error: str | None = None


def _init_client() -> None:
    global _connected, _init_error
    _connected = False
    _init_error = None

    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk:
        _init_error = "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set"
        return

    try:
        from langfuse import get_client
        lf = get_client()
        if lf.auth_check():
            _connected = True
        else:
            _init_error = "auth_check() failed — check your keys"
    except Exception as e:
        _init_error = str(e)


def init_status() -> tuple[bool, str | None]:
    """Return (connected, error_message) for sidebar display."""
    return (_connected, _init_error)


_init_client()


# ---------------------------------------------------------------------------
# No-op fallbacks
# ---------------------------------------------------------------------------

class _NoOpSpan:
    def update(self, **kwargs) -> None:
        pass

    def end(self, **kwargs) -> None:
        pass


@contextmanager
def _noop_ctx(**kwargs):
    yield _NoOpSpan()


# ---------------------------------------------------------------------------
# Public helpers used by ConversationFlow
# ---------------------------------------------------------------------------

def new_trace(name: str, session_id: str | None = None):
    """Start a new top-level trace. Returns a span object (or no-op)."""
    if not _connected:
        return _NoOpSpan()
    try:
        from langfuse import get_client
        lf = get_client()
        trace = lf.start_as_current_observation(
            as_type="span",
            name=name,
            session_id=session_id,
        )
        return trace
    except Exception:
        return _NoOpSpan()


@contextmanager
def generation_span(
    name: str,
    model: str,
    input: Any,
) -> Generator[_NoOpSpan, None, None]:
    """Context manager for a single LLM generation. Flushes on exit."""
    if not _connected:
        yield _NoOpSpan()
        return
    try:
        from langfuse import get_client
        lf = get_client()
        with lf.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input,
        ) as gen:
            yield gen
        lf.flush()
    except Exception:
        yield _NoOpSpan()


def flush() -> None:
    if not _connected:
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass
