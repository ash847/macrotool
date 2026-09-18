"""
Supabase client surfaces.

- service client: all app writes plus server-side engine/admin operations
- anon client: optional, retained only for external RLS smoke testing
"""

from __future__ import annotations

import os

_anon_client = None
_service_client = None
_init_error: str | None = None


def _init() -> None:
    global _anon_client, _service_client, _init_error
    _anon_client = None
    _service_client = None
    _init_error = None

    url = os.environ.get("SUPABASE_URL", "")
    anon_key = os.environ.get("SUPABASE_ANON_KEY", "") or os.environ.get("SUPABASE_KEY", "")
    service_key = os.environ.get("SUPABASE_SERVICE_KEY", "")

    if not url:
        _init_error = "SUPABASE_URL not set"
        return

    try:
        from supabase import create_client
        if anon_key:
            _anon_client = create_client(url, anon_key)
        if service_key:
            _service_client = create_client(url, service_key)
        if _anon_client is None and _service_client is None:
            _init_error = "No Supabase keys configured"
        elif _service_client is None:
            _init_error = "SUPABASE_SERVICE_KEY not set"
    except Exception as e:
        _init_error = str(e)


def init_status() -> tuple[bool, str | None]:
    return (_service_client is not None, _init_error)


_init()


def log_query(
    prompt: str,
    pair: str | None,
    direction: str | None,
    magnitude_pct: float | None,
    horizon_days: int | None,
    target_z: float | None,
    carry_regime: int | None,
    top_structure: str | None,
    llm_response: str,
    user_email: str | None = None,
    session_id: str | None = None,
) -> None:
    if _service_client is None:
        return
    row = {
        "prompt":        prompt,
        "pair":          pair,
        "direction":     direction,
        "magnitude_pct": magnitude_pct,
        "horizon_days":  horizon_days,
        "target_z":      round(target_z, 4) if target_z is not None else None,
        "carry_regime":  carry_regime,
        "top_structure": top_structure,
        "llm_response":  llm_response[:8000],
    }
    if user_email:
        row["user_email"] = user_email
    if session_id:
        row["session_id"] = session_id
    _insert_with_optional("queries", row, optional=("user_email", "session_id"))


def log_feedback(
    prompt: str | None,
    pair: str | None,
    answers: list[bool | None],
    questions: list[str],
    note: str | None = None,
    user_email: str | None = None,
    session_id: str | None = None,
) -> None:
    if _service_client is None:
        return
    row: dict = {
        "prompt": prompt,
        "pair":   pair,
        "note":   note,
    }
    if user_email:
        row["user_email"] = user_email
    if session_id:
        row["session_id"] = session_id
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        row[f"q{i}_text"] = q
        row[f"q{i}_answer"] = a
    _insert_with_optional("feedback", row, optional=("user_email", "session_id"))


# ---------------------------------------------------------------------------
# Tester-rollout logging: chat transcripts, app errors, one-click reactions.
# All fail-open (never raise) and no-op when Supabase isn't configured, so a
# logging outage can never break a tester's session. New tables — see the
# CREATE TABLE migration in db/migrations_tester_logging.sql.
# ---------------------------------------------------------------------------

def _insert_with_optional(table: str, row: dict, optional: tuple[str, ...] = ()) -> None:
    """Insert a row; on failure retry once without the ``optional`` columns (so a
    not-yet-migrated column like ``session_id`` degrades instead of erroring), then
    give up silently. Never raises."""
    try:
        _service_client.table(table).insert(row, returning="minimal").execute()
        return
    except Exception:
        pass
    slim = {k: v for k, v in row.items() if k not in optional}
    try:
        _service_client.table(table).insert(slim, returning="minimal").execute()
    except Exception:
        pass


def log_chat_turn(
    *,
    session_id: str | None,
    chat_id: str,
    seq: int,
    surface: str,             # "agent_tab" | "trade_view"
    role: str,                # "user" | "assistant"
    text: str | None,
    tool_trace: list | None = None,
    pair: str | None = None,
    view_json: dict | None = None,
    user_email: str | None = None,
) -> None:
    """One row per chat turn (both the Agent tab and the in-context trade chat)."""
    if _service_client is None:
        return
    row = {
        "session_id": session_id,
        "chat_id":    chat_id,
        "seq":        seq,
        "surface":    surface,
        "role":       role,
        "text":       (text or "")[:20000],
        "tool_trace": tool_trace,
        "pair":       pair,
        "view_json":  view_json,
    }
    if user_email:
        row["user_email"] = user_email
    _insert_with_optional("chat_turns", row, optional=("user_email", "session_id"))


def log_app_error(
    *,
    context: str,
    error_type: str,
    message: str | None,
    traceback: str | None = None,
    user_email: str | None = None,
    session_id: str | None = None,
) -> None:
    """Persist an application error so failures testers hit are visible remotely
    (the local logs/session.log file is ephemeral on Streamlit Cloud)."""
    if _service_client is None:
        return
    row = {
        "context":    context,
        "error_type": error_type,
        "message":    (message or "")[:4000],
        "traceback":  traceback[:8000] if traceback else None,
    }
    if user_email:
        row["user_email"] = user_email
    if session_id:
        row["session_id"] = session_id
    _insert_with_optional("app_errors", row, optional=("user_email", "session_id"))


def log_reaction(
    *,
    session_id: str | None,
    surface: str,             # "trade_view" | "agent_tab"
    target_kind: str,         # "recommendation" | "chat"
    target_ref: str,
    rating: str,              # "up" | "down"
    reason: str | None = None,
    pair: str | None = None,
    view_summary: str | None = None,
    chat_id: str | None = None,
    seq: int | None = None,
    user_email: str | None = None,
) -> None:
    """One-click 👍/👎 (+ optional one-tap reason on 👎) on a recommendation or a
    chat reply."""
    if _service_client is None:
        return
    row = {
        "session_id":   session_id,
        "surface":      surface,
        "target_kind":  target_kind,
        "target_ref":   target_ref,
        "rating":       rating,
        "reason":       reason,
        "pair":         pair,
        "view_summary": view_summary,
        "chat_id":      chat_id,
        "seq":          seq,
    }
    if user_email:
        row["user_email"] = user_email
    _insert_with_optional("reactions", row, optional=("user_email", "session_id"))


GLOBAL_SCENARIO_WEIGHTS_KEY = "scenario_definitions"


def personal_weights_key(email: str) -> str:
    """config_history key for a user's personal scenario-weights profile. The global
    profile keeps the bare key; personal profiles namespace by email. Same table, same
    versioning — lookup is by key, so this needs no schema change."""
    return f"{GLOBAL_SCENARIO_WEIGHTS_KEY}::{email}"


def fetch_config_for_engine(key: str) -> dict | None:
    data, _ = fetch_config_for_engine_with_meta(key)
    return data


def fetch_config_for_engine_with_meta(key: str) -> tuple[dict | None, str]:
    if _service_client is None:
        return None, "unavailable"
    try:
        result = (
            _service_client.table("config_history")
            .select("value")
            .eq("key", key)
            .order("saved_at", desc=True)
            .limit(1)
            .execute()
        )
        data = result.data[0].get("value") if result.data else None
        return (data if data else None), ("supabase" if data else "missing")
    except Exception:
        return None, "error"


def save_config(key: str, value: dict, *, _admin: bool, user_email: str | None = None) -> bool:
    if not _admin:
        raise PermissionError("admin only")
    if _service_client is None:
        return False
    try:
        row = {
            "key": key,
            "value": value,
        }
        if user_email:
            row["user_email"] = user_email
        _service_client.table("config_history").insert(row).execute()
        return True
    except Exception:
        if not user_email:
            return False
        try:
            _service_client.table("config_history").insert({
                "key": key,
                "value": value,
            }).execute()
            return True
        except Exception:
            return False


def fetch_config_history(key: str, limit: int = 30, *, _admin: bool) -> list[dict]:
    if not _admin:
        raise PermissionError("admin only")
    if _service_client is None:
        return []
    try:
        result = (
            _service_client.table("config_history")
            .select("saved_at, user_email, value")
            .eq("key", key)
            .order("saved_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception:
        try:
            result = (
                _service_client.table("config_history")
                .select("saved_at, value")
                .eq("key", key)
                .order("saved_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception:
            return []


def fetch_queries(*, _admin: bool) -> list[dict]:
    if not _admin:
        raise PermissionError("admin only")
    if _service_client is None:
        return []
    try:
        result = _service_client.table("queries").select(
            "created_at, user_email, pair, direction, magnitude_pct, horizon_days, target_z, carry_regime, top_structure, prompt"
        ).order("created_at", desc=True).execute()
        return result.data or []
    except Exception:
        result = _service_client.table("queries").select(
            "created_at, pair, direction, magnitude_pct, horizon_days, target_z, carry_regime, top_structure, prompt"
        ).order("created_at", desc=True).execute()
        return result.data or []


def reinit() -> None:
    _init()
