"""Pure helpers for the Trade View sizing-method toggle (Kelly vs fixed loss).

The Streamlit widgets in `app.py` just wire `st.session_state` → these helpers →
`SizingSpec`. Keeping the logic here (no Streamlit imports) makes it unit-testable;
the widget layer is verified manually (see KELLY_SIZING_PLAN §8).
"""
from __future__ import annotations

from analytics.sizing import SizingSpec, view_implied_distribution

_FIXED, _KELLY = "fixed_loss", "kelly"
_RESEED_KEYS = ("pair", "horizon", "target", "conviction")


def build_sizing_spec(state: dict, ms=None, target: float | None = None) -> SizingSpec:
    """Assemble a SizingSpec from session-state-like values.

    Fixed-loss carries the R:R. Kelly carries the elicited distribution; if none is
    present yet, it falls back to the view-implied seed (so the table renders before
    the PM elicits). Missing seed inputs (no target) → fixed-loss fallback."""
    method = state.get("sizing_method", _FIXED)
    if method != _KELLY:
        return SizingSpec(method=_FIXED, target_rr=float(state.get("target_rr", 3.0)))

    probs = state.get("kelly_probs")
    bins = state.get("kelly_bins")
    if (probs is None or bins is None) and ms is not None and target:
        probs, bins = view_implied_distribution(
            ms.spot, ms.fwd, ms.vol, ms.T, target, state.get("conviction", "medium"),
            n_bins=int(state.get("kelly_n_bins", 41)),
        )
    if probs is None or bins is None:
        # Kelly requested but no distribution and nothing to seed from → fixed-loss.
        return SizingSpec(method=_FIXED, target_rr=float(state.get("target_rr", 3.0)))

    return SizingSpec(
        method=_KELLY,
        kelly_lambda=float(state.get("kelly_lambda", 0.5)),
        bankroll=float(state.get("bankroll", 100.0)),
        kelly_probs=tuple(probs),
        kelly_bins=tuple(bins),
    )


def should_reseed(prev_ctx: dict | None, new_ctx: dict) -> bool:
    """Re-seed the view-implied distribution only on real context changes — pair /
    horizon / target / conviction change, or switching *into* Kelly mode. Not on λ
    moves or plain reruns (which would wipe an elicited curve)."""
    if prev_ctx is None:
        return True
    if new_ctx.get("sizing_method") == _KELLY and prev_ctx.get("sizing_method") != _KELLY:
        return True
    return any(prev_ctx.get(k) != new_ctx.get(k) for k in _RESEED_KEYS)


def notional_column_label(method: str) -> str:
    return "Notional (Kelly)" if method == _KELLY else "Notional (max-loss)"


def meaning_banner(method: str) -> str:
    if method == _KELLY:
        return ("Sized to each variant's growth-optimal bet under your distribution — "
                "a bigger notional means better edge/odds, not just bigger risk.")
    return "Sized to equal max loss (R:R-derived) — same risk per variant, compare the reward."


def kelly_row_flag(structure_notional: float | None, cap: float) -> str:
    """Per-variant UI flag for the Kelly notional column."""
    if structure_notional is None:
        return ""
    if structure_notional <= 0.0:
        return "no Kelly edge"
    if structure_notional >= cap - 1e-6:
        return "capped"
    return ""
