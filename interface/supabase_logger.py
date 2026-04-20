"""
Supabase query logger — no-op safe if keys are not configured.

Logs one row per query to the `queries` table:
  prompt, pair, direction, magnitude_pct, horizon_days,
  target_z, carry_regime, top_structure, llm_response
"""

from __future__ import annotations

import os
from typing import Any

_client = None


def _init() -> None:
    global _client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return
    try:
        from supabase import create_client
        _client = create_client(url, key)
    except Exception:
        pass


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
    if _client is None:
        return
    try:
        _client.table("queries").insert({
            "prompt":        prompt,
            "pair":          pair,
            "direction":     direction,
            "magnitude_pct": magnitude_pct,
            "horizon_days":  horizon_days,
            "target_z":      round(target_z, 4) if target_z is not None else None,
            "carry_regime":  carry_regime,
            "top_structure": top_structure,
            "llm_response":  llm_response[:8000],  # guard against max row size
        }).execute()
    except Exception:
        pass


def reinit() -> None:
    """Call after os.environ is updated (e.g. after st.secrets injection)."""
    _init()
