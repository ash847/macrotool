"""
Session debug logger — writes structured entries to logs/session.log.
Each entry is a JSON line so it's both human-readable and grep-friendly.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent / "logs" / "session.log"

_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(message)s"))

_logger = logging.getLogger("macrotool.debug")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)
_logger.propagate = False


def _entry(event: str, **kwargs) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
    _logger.debug(json.dumps(record, default=str))


def log_prompt(prompt: str) -> None:
    _entry("user_prompt", prompt=prompt)


def log_view_extracted(view_dict: dict) -> None:
    _entry("view_extracted", view=view_dict)


def log_view_failed(llm_response: str) -> None:
    _entry("view_extraction_failed", llm_response=llm_response[:500])


def log_market_state(ms) -> None:
    _entry(
        "market_state",
        spot=ms.spot,
        fwd=ms.fwd,
        vol=ms.vol,
        T=ms.T,
        r_d=ms.r_d,
        r_f=ms.r_f,
        c=ms.c,
        carry_regime=ms.carry_regime,
        target_z=ms.target_z,
        atmfsratio=ms.atmfsratio,
        put_call=ms.put_call,
    )


def log_scorer_result(result) -> None:
    shortlist = [
        {"rank": s.rank, "id": s.structure_id, "is_overlay": s.is_exotic}
        for s in result.shortlist
    ]
    _entry("scorer_result", shortlist=shortlist, rules_fired=result.rules_fired)


def log_error(context: str, exc: Exception) -> None:
    _entry("error", context=context, error=str(exc))
    _persist_error(context, exc)


def _persist_error(context: str, exc: Exception) -> None:
    """Best-effort mirror of the error to Supabase so failures testers hit are
    visible remotely (the local file above is ephemeral on Streamlit Cloud). Reads
    the session id / user email from Streamlit session state when available. Never
    raises — logging must not break the app."""
    try:
        import traceback as _tb

        from interface.supabase_logger import log_app_error

        session_id = user_email = None
        try:
            import streamlit as st
            session_id = st.session_state.get("session_id")
            user_email = st.session_state.get("current_user_email")
        except Exception:
            pass
        log_app_error(
            context=context,
            error_type=type(exc).__name__,
            message=str(exc),
            traceback="".join(_tb.format_exception(type(exc), exc, exc.__traceback__)),
            user_email=user_email,
            session_id=session_id,
        )
    except Exception:
        pass


def read_recent(n: int = 50) -> list[dict]:
    """Return the last n log entries as parsed dicts."""
    if not _LOG_PATH.exists():
        return []
    lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"raw": line})
    return entries
