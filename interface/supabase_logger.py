"""
Supabase client surfaces.

- anon client: user-facing inserts only
- service client: server-side engine config reads and admin operations
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
    return ((_anon_client is not None and _service_client is not None), _init_error)


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
) -> None:
    if _anon_client is None:
        return
    _anon_client.table("queries").insert({
        "prompt":        prompt,
        "pair":          pair,
        "direction":     direction,
        "magnitude_pct": magnitude_pct,
        "horizon_days":  horizon_days,
        "target_z":      round(target_z, 4) if target_z is not None else None,
        "carry_regime":  carry_regime,
        "top_structure": top_structure,
        "llm_response":  llm_response[:8000],
    }).execute()


def log_feedback(
    prompt: str | None,
    pair: str | None,
    answers: list[bool | None],
    questions: list[str],
    note: str | None = None,
) -> None:
    if _anon_client is None:
        return
    row: dict = {
        "prompt": prompt,
        "pair":   pair,
        "note":   note,
    }
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        row[f"q{i}_text"] = q
        row[f"q{i}_answer"] = a
    _anon_client.table("feedback").insert(row).execute()


def fetch_config_for_engine(key: str) -> dict | None:
    if _service_client is None:
        return None
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
        return data if data else None
    except Exception:
        return None


def save_config(key: str, value: dict, *, _admin: bool) -> bool:
    if not _admin:
        raise PermissionError("admin only")
    if _service_client is None:
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
    result = _service_client.table("queries").select(
        "created_at, pair, direction, magnitude_pct, horizon_days, target_z, carry_regime, top_structure, prompt"
    ).order("created_at", desc=True).execute()
    return result.data or []


def reinit() -> None:
    _init()
