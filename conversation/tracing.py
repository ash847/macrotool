"""
Langfuse tracing — SDK v4, no-op safe if keys are not configured.

Uses start_observation() (non-context-manager) so spans can be held across
generator yield boundaries without losing OTel context.

Required env vars:
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  LANGFUSE_BASE_URL  (default: https://cloud.langfuse.com)
"""

from __future__ import annotations

import os
from typing import Any

_lf = None
_init_error: str | None = None


def _init_client() -> None:
    global _lf, _init_error
    _lf = None
    _init_error = None

    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk:
        _init_error = "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set"
        return
    try:
        from langfuse import Langfuse
        _lf = Langfuse(
            public_key=pk,
            secret_key=sk,
            host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
        _init_error = None
    except Exception as e:
        _lf = None
        _init_error = str(e)


def init_status() -> tuple[bool, str | None]:
    return (_lf is not None, _init_error)


_init_client()


# ---------------------------------------------------------------------------
# No-op span
# ---------------------------------------------------------------------------

class _NoOpSpan:
    def update(self, **kwargs) -> None:
        pass

    def end(self, **kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def new_session_span(name: str = "macrotool-session") -> Any:
    """Create a top-level span for the session. Returns a span or no-op."""
    if _lf is None:
        return _NoOpSpan()
    try:
        return _lf.start_observation(as_type="span", name=name)
    except Exception:
        return _NoOpSpan()


def new_generation(
    name: str,
    model: str,
    input: Any,
    session_span: Any = None,
) -> Any:
    """
    Create a generation span. Returns the span object — caller must call
    span.update(output=...) and span.end() when done, then flush().
    """
    if _lf is None:
        return _NoOpSpan()
    try:
        kwargs: dict = dict(as_type="generation", name=name, model=model, input=input)
        if session_span is not None and hasattr(session_span, "trace_id"):
            from langfuse.types import TraceContext
            kwargs["trace_context"] = TraceContext(
                trace_id=session_span.trace_id,
                parent_observation_id=getattr(session_span, "id", None),
            )
        return _lf.start_observation(**kwargs)
    except Exception:
        return _NoOpSpan()


def flush() -> None:
    if _lf is not None:
        try:
            _lf.flush()
        except Exception:
            pass
