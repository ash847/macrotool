"""
Kelly v2 prototype — subjective distribution → edge vs market-implied.

Option 1 (CDF mode): PM enters the price level at each of N fixed quantiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from baseline import synthetic_lognormal_baseline
from edge import anchors_from_baseline, compute_edge
from elicitation import elicit_from_cdf_anchors


ANCHOR_PRESETS: dict[int, tuple[float, ...]] = {
    5:  (0.05, 0.25, 0.50, 0.75, 0.95),
    7:  (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    9:  (0.02, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.98),
    11: (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98),
}


# Discount factor — currently fixed at 1.0 for the prototype. It cancels out of
# the edge calculation when applied consistently on both sides, so this only
# affects the absolute scale of pm_price / mkt_price, not the edge itself.
DISCOUNT_FACTOR: float = 1.0


def init_state() -> None:
    if "n_anchors" not in st.session_state:
        st.session_state.n_anchors = 7
    if "forward" not in st.session_state:
        st.session_state.forward = 5.00
    if "sigma" not in st.session_state:
        st.session_state.sigma = 0.10
    if "tenor_years" not in st.session_state:
        st.session_state.tenor_years = 0.25
    if "strike" not in st.session_state:
        st.session_state.strike = 5.00
    if "is_call" not in st.session_state:
        st.session_state.is_call = True


def reset_anchors_to_baseline() -> None:
    """Seed anchor prices from the current market baseline (default starting view)."""
    base = synthetic_lognormal_baseline(
        forward=st.session_state.forward,
        sigma=st.session_state.sigma,
        tenor_years=st.session_state.tenor_years,
        n_bins=400,
        n_stdev=6.0,
    )
    quantiles = ANCHOR_PRESETS[st.session_state.n_anchors]
    seed = anchors_from_baseline(base, list(quantiles))
    for i, p in enumerate(seed):
        st.session_state[f"anchor_{i}"] = float(p)


def sync_anchors_on_n_change() -> None:
    """When N changes, re-seed anchor inputs (they'd otherwise carry stale values)."""
    reset_anchors_to_baseline()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Market baseline")
        st.number_input("Forward", min_value=0.01, step=0.01, format="%.4f", key="forward")
        st.number_input(
            "Vol (annualised)",
            min_value=0.001,
            max_value=2.0,
            step=0.005,
            format="%.4f",
            key="sigma",
        )
        st.number_input(
            "Tenor (years)",
            min_value=1.0 / 365,
            max_value=10.0,
            step=0.05,
            format="%.4f",
            key="tenor_years",
        )

        st.divider()
        st.header("Anchors")
        st.selectbox(
            "Number of anchors",
            options=sorted(ANCHOR_PRESETS.keys()),
            key="n_anchors",
            on_change=sync_anchors_on_n_change,
        )
        st.button("Reset anchors to market baseline", on_click=reset_anchors_to_baseline)

        st.caption(
            "Discount factor for pricing is fixed at 1.0 in this prototype "
            "(it cancels out of edge under consistent application)."
        )


def render_anchor_inputs(quantiles: tuple[float, ...]) -> np.ndarray:
    st.subheader("Your view — price at each quantile")
    st.caption(
        "Enter the price level at which you expect the cumulative probability "
        "to reach each quantile. Values must strictly increase from top to bottom."
    )

    # Lazy seed on first render.
    if f"anchor_0" not in st.session_state:
        reset_anchors_to_baseline()

    prices = []
    cols = st.columns(len(quantiles))
    for i, (q, col) in enumerate(zip(quantiles, cols)):
        with col:
            # If a session value doesn't yet exist for this anchor, seed it
            # before instantiating the widget (otherwise Streamlit complains).
            key = f"anchor_{i}"
            if key not in st.session_state:
                st.session_state[key] = float(st.session_state.forward)
            v = st.number_input(
                f"P ≤ {int(round(q * 100))}%",
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=key,
            )
            prices.append(v)

    return np.array(prices, dtype=float)


def render_structure_inputs() -> tuple[float, bool]:
    st.subheader("Option to price")
    col1, col2 = st.columns([2, 1])
    with col1:
        strike = st.number_input("Strike", min_value=0.0001, step=0.01, format="%.4f", key="strike")
    with col2:
        if "option_type" not in st.session_state:
            st.session_state.option_type = "Call" if st.session_state.is_call else "Put"
        kind = st.radio("Type", options=["Call", "Put"], horizontal=False, key="option_type")
        is_call = kind == "Call"
        st.session_state.is_call = is_call
    return strike, is_call


def render_edge_panel(rep, strike: float, is_call: bool) -> None:
    st.subheader("Edge vs market-implied")

    if rep.out_of_range:
        st.warning(
            f"Strike {strike:.4f} is outside your elicited support — your "
            f"distribution puts no mass beyond the outer anchors, so the edge "
            f"below reflects that explicitly. Widen the outer anchors if you "
            f"actually have a view at this strike."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("PM price", f"{rep.pm_price:.4f}")
    col2.metric("Market price", f"{rep.mkt_price:.4f}")
    if rep.edge_pct_of_mid is None:
        col3.metric("Edge (abs / % mid)", f"{rep.edge_absolute:+.4f}", "—")
    else:
        col3.metric(
            "Edge (abs / % mid)",
            f"{rep.edge_absolute:+.4f}",
            f"{rep.edge_pct_of_mid:+.1f}% of mid",
        )

    st.caption(
        "Edge is **vs market-implied** pricing — the market baseline is the "
        "risk-neutral distribution, so this measures deviation from the "
        "market's pricing of risk, not pure forecasting edge."
    )


def main() -> None:
    st.set_page_config(page_title="Kelly v2 — Subjective Edge", layout="wide")
    st.title("Kelly v2 — Subjective Distribution → Edge")
    st.caption("Option 1 (CDF mode) prototype. Edge vs market-implied pricing only; Kelly sizing deferred.")

    init_state()
    render_sidebar()

    quantiles = ANCHOR_PRESETS[st.session_state.n_anchors]
    prices = render_anchor_inputs(quantiles)

    # Validate monotonicity before constructing the distribution.
    if not np.all(np.diff(prices) > 0):
        st.error("Anchor prices must be strictly increasing. Adjust the values above.")
        return

    base = synthetic_lognormal_baseline(
        forward=st.session_state.forward,
        sigma=st.session_state.sigma,
        tenor_years=st.session_state.tenor_years,
        n_bins=400,
        n_stdev=6.0,
    )
    pm = elicit_from_cdf_anchors(prices, list(quantiles))

    strike, is_call = render_structure_inputs()

    rep = compute_edge(pm, base, strike=strike, is_call=is_call, discount_factor=DISCOUNT_FACTOR)
    render_edge_panel(rep, strike, is_call)


if __name__ == "__main__":
    main()
