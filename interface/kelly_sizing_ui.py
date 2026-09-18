"""Pure helpers for the Trade View sizing-method toggle (Kelly vs fixed loss).

The Streamlit widgets in `app.py` just wire `st.session_state` → these helpers →
`SizingSpec`. Keeping the logic here (no Streamlit imports) makes it unit-testable;
the widget layer is verified manually (see KELLY_SIZING_PLAN §8).
"""
from __future__ import annotations

from analytics.sizing import SizingSpec, curve_for_trade

_FIXED, _KELLY = "fixed_loss", "kelly"


def build_sizing_spec(state: dict, ms=None, trade_key: tuple | None = None) -> SizingSpec:
    """Assemble a SizingSpec from session-state-like values.

    Fixed-loss carries the R:R. Kelly sizes against the PM's stated distribution **for
    this trade** — ``state["kelly_curve_key"]`` must equal ``curve_key(*trade_key)``,
    trade_key = (pair, expiry). With no stated distribution the trade is sized
    FIXED-LOSS and flagged (``kelly_fallback``) so the UI says so: the tool never
    sizes on an edge the PM didn't give."""
    method = state.get("sizing_method", _FIXED)
    rr = float(state.get("target_rr", 3.0))
    if method != _KELLY:
        return SizingSpec(method=_FIXED, target_rr=rr)

    curve = None
    stated_key = state.get("kelly_curve_key")
    if trade_key is not None and stated_key and state.get("kelly_probs") and state.get("kelly_bins"):
        curve = curve_for_trade({stated_key: (state["kelly_probs"], state["kelly_bins"])},
                                *trade_key)
    if curve is None:
        return SizingSpec(method=_FIXED, target_rr=rr, kelly_fallback=True)

    probs, bins = curve
    return SizingSpec(
        method=_KELLY,
        kelly_lambda=float(state.get("kelly_lambda", 0.5)),
        bankroll=float(state.get("bankroll", 100.0)),
        kelly_probs=probs,
        kelly_bins=bins,
    )


def notional_column_label(method: str) -> str:
    return "Notional (Kelly)" if method == _KELLY else "Notional (max-loss)"


KELLY_FALLBACK_MSG = ("Kelly is selected, but you haven't set up a distribution for this "
                      "trade (pair + expiry), so it is sized **FIXED-LOSS**. Set up your "
                      "distribution to size under Kelly.")


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
