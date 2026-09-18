"""Pure helpers for the Trade View sizing-method toggle (Kelly vs fixed loss).

The Streamlit widgets in `app.py` just wire `st.session_state` → these helpers →
`SizingSpec`. Keeping the logic here (no Streamlit imports) makes it unit-testable;
the widget layer is verified manually (see KELLY_SIZING_PLAN §8).
"""
from __future__ import annotations

from analytics.sizing import SizingSpec, curve_for_trade, market_distribution

_FIXED, _KELLY = "fixed_loss", "kelly"


def build_sizing_spec(state: dict, ms=None, trade_key: tuple | None = None) -> SizingSpec:
    """Assemble a SizingSpec from session-state-like values.

    Fixed-loss carries the R:R. Kelly sizes against the PM's stated distribution
    **for this trade** (``state["kelly_curve_key"]`` must equal ``trade_key`` =
    (pair, horizon_days)); otherwise against the market distribution, which states
    no edge. The tool never synthesises an edge the PM didn't give."""
    method = state.get("sizing_method", _FIXED)
    if method != _KELLY:
        return SizingSpec(method=_FIXED, target_rr=float(state.get("target_rr", 3.0)))

    curve = None
    if trade_key is not None:
        curve = curve_for_trade(state.get("kelly_curve_key"), state.get("kelly_probs"),
                                state.get("kelly_bins"), *trade_key)
    if curve is not None:
        probs, bins = curve
        source = "explicit"
    elif ms is not None:
        probs, bins = market_distribution(ms.spot, ms.fwd, ms.vol, ms.T)
        source = "market"
    else:
        # No market state to anchor even the market curve — cannot size Kelly at all.
        return SizingSpec(method=_FIXED, target_rr=float(state.get("target_rr", 3.0)))

    return SizingSpec(
        method=_KELLY,
        kelly_lambda=float(state.get("kelly_lambda", 0.5)),
        bankroll=float(state.get("bankroll", 100.0)),
        kelly_probs=tuple(probs),
        kelly_bins=tuple(bins),
        distribution_source=source,
    )


def notional_column_label(method: str) -> str:
    return "Notional (Kelly)" if method == _KELLY else "Notional (max-loss)"


def meaning_banner(method: str, source: str | None = None) -> str:
    if method == _KELLY and source == "market":
        return ("Kelly is sizing against the MARKET distribution — you haven't stated a "
                "distribution for this trade, so there is no edge and sizes are small or "
                "zero. Shape your distribution to size under Kelly.")
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
