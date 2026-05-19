"""
Kelly v2 prototype — subjective distribution → edge vs market-implied.

Option 1 (CDF mode): PM enters the price level at each of N fixed quantiles.
Option 2 (PDF mode): PM enters the probability mass in each of N sigma-anchored
buckets, with bucket boundaries on the lognormal market smile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from baseline import load_snapshot, synthetic_lognormal_baseline
from edge import anchors_from_baseline, compute_edge
from elicitation import (
    Distribution,
    default_sigma_boundaries,
    elicit_from_cdf_anchors,
    elicit_from_pdf_buckets,
    sigma_boundaries_to_prices,
)
from pricing import forward_of
from viz import (
    render_option1_chart,
    render_option2_chart,
    render_option2_stacked_chart,
)


FIXTURE_DIR = PKG_ROOT / "fixtures"
BASELINE_SOURCE_SYNTHETIC = "Synthetic lognormal (set F, σ, T)"
BASELINE_SOURCE_FIXTURE = "Load saved baseline (.json)"


MODE_OPTION1 = "Use fixed probability bins"
MODE_OPTION2 = "Use fixed spot ranges"

ANCHOR_PRESETS: dict[int, tuple[float, ...]] = {
    5:  (0.05, 0.25, 0.50, 0.75, 0.95),
    7:  (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    9:  (0.02, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.98),
    11: (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98),
}

DISCOUNT_FACTOR: float = 1.0


# --- session state ---


def init_state() -> None:
    defaults = {
        "mode": MODE_OPTION1,
        "n_anchors": 7,
        "baseline_source": BASELINE_SOURCE_SYNTHETIC,
        "forward": 5.00,
        "sigma": 0.10,
        "tenor_years": 0.25,
        "strike": 5.00,
        "is_call": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _list_fixtures() -> list[Path]:
    if not FIXTURE_DIR.exists():
        return []
    return sorted(p for p in FIXTURE_DIR.iterdir() if p.suffix == ".json")


def _current_baseline() -> Distribution:
    if st.session_state.baseline_source == BASELINE_SOURCE_FIXTURE and st.session_state.get("fixture_path"):
        dist, meta = load_snapshot(st.session_state.fixture_path)
        # Keep F/σ/T inputs in sync with the loaded fixture so derived UI elements
        # (sigma-bucket boundaries in Option 2) line up with the loaded distribution.
        st.session_state.forward = float(meta.get("forward", forward_of(dist)))
        st.session_state.tenor_years = float(meta.get("tenor_years", st.session_state.tenor_years))
        return dist
    return synthetic_lognormal_baseline(
        forward=st.session_state.forward,
        sigma=st.session_state.sigma,
        tenor_years=st.session_state.tenor_years,
        n_bins=400,
        n_stdev=6.0,
    )


def reset_anchors_to_baseline() -> None:
    """Option 1: seed anchor prices from the baseline CDF at the active quantiles."""
    quantiles = ANCHOR_PRESETS[st.session_state.n_anchors]
    seed = anchors_from_baseline(_current_baseline(), list(quantiles))
    for i, p in enumerate(seed):
        st.session_state[f"anchor_{i}"] = float(p)


def reset_buckets_to_uniform() -> None:
    """Option 2: seed bucket probabilities to uniform."""
    n = st.session_state.n_anchors
    each = 1.0 / n
    for i in range(n):
        st.session_state[f"bucket_{i}"] = each


def reset_buckets_to_baseline() -> None:
    """Option 2: seed bucket probabilities to match the baseline mass in each bucket."""
    base = _current_baseline()
    offsets = default_sigma_boundaries(st.session_state.n_anchors)
    boundaries = sigma_boundaries_to_prices(
        offsets,
        forward=st.session_state.forward,
        sigma=st.session_state.sigma,
        tenor_years=st.session_state.tenor_years,
    )
    masses = []
    for i in range(len(boundaries) - 1):
        mask = (base.bins >= boundaries[i]) & (base.bins < boundaries[i + 1])
        masses.append(float(base.probs[mask].sum()))
    masses = np.array(masses)
    if masses.sum() > 0:
        masses = masses / masses.sum()
    for i, m in enumerate(masses):
        st.session_state[f"bucket_{i}"] = float(m)


def sync_on_n_change() -> None:
    """When N changes, re-seed whichever inputs the active mode depends on."""
    if st.session_state.mode == MODE_OPTION1:
        reset_anchors_to_baseline()
    else:
        reset_buckets_to_baseline()


# --- rendering ---


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Elicitation mode")
        st.radio(
            "Mode",
            options=[MODE_OPTION1, MODE_OPTION2],
            key="mode",
            label_visibility="collapsed",
        )

        st.divider()
        st.header("Market baseline")
        st.radio(
            "Source",
            options=[BASELINE_SOURCE_SYNTHETIC, BASELINE_SOURCE_FIXTURE],
            key="baseline_source",
        )

        if st.session_state.baseline_source == BASELINE_SOURCE_FIXTURE:
            fixtures = _list_fixtures()
            if not fixtures:
                st.warning("No fixtures found in fixtures/.")
            else:
                choice = st.selectbox(
                    "Fixture",
                    options=fixtures,
                    format_func=lambda p: p.name,
                    key="fixture_choice",
                )
                st.session_state.fixture_path = str(choice)
                _, meta = load_snapshot(choice)
                st.caption(f"pair: {meta.get('pair', '?')}, source: {meta.get('source', '?')}")
        else:
            st.number_input("Forward", min_value=0.01, step=0.01, format="%.4f", key="forward")
            st.number_input(
                "Vol (annualised)", min_value=0.001, max_value=2.0,
                step=0.005, format="%.4f", key="sigma",
            )
            st.number_input(
                "Tenor (years)", min_value=1.0 / 365, max_value=10.0,
                step=0.05, format="%.4f", key="tenor_years",
            )

        st.divider()
        st.header("Anchors / buckets")
        st.selectbox(
            "Number",
            options=sorted(ANCHOR_PRESETS.keys()),
            key="n_anchors",
            on_change=sync_on_n_change,
        )
        if st.session_state.mode == MODE_OPTION1:
            st.button("Reset anchors to market baseline", on_click=reset_anchors_to_baseline)
        else:
            col1, col2 = st.columns(2)
            col1.button("Reset to baseline", on_click=reset_buckets_to_baseline)
            col2.button("Reset to uniform", on_click=reset_buckets_to_uniform)

        st.caption(
            "Discount factor for pricing is fixed at 1.0 in this prototype "
            "(it cancels out of edge under consistent application)."
        )


def render_option1_inputs(quantiles: tuple[float, ...]) -> np.ndarray:
    st.subheader("Your view — price at each quantile")
    st.caption(
        "Enter the price level at which you expect the cumulative probability "
        "to reach each quantile. Values must strictly increase from left to right."
    )

    if "anchor_0" not in st.session_state:
        reset_anchors_to_baseline()

    prices = []
    cols = st.columns(len(quantiles))
    for i, (q, col) in enumerate(zip(quantiles, cols)):
        with col:
            key = f"anchor_{i}"
            if key not in st.session_state:
                st.session_state[key] = float(st.session_state.forward)
            v = st.number_input(
                f"P ≤ {int(round(q * 100))}%",
                min_value=0.0001, step=0.01, format="%.4f", key=key,
            )
            prices.append(v)

    return np.array(prices, dtype=float)


def render_option2_inputs(n_buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Render Option 2 bucket inputs. Returns (boundaries_in_prices, bucket_probs)."""
    offsets = default_sigma_boundaries(n_buckets)
    boundaries = sigma_boundaries_to_prices(
        offsets,
        forward=st.session_state.forward,
        sigma=st.session_state.sigma,
        tenor_years=st.session_state.tenor_years,
    )

    st.subheader("Your view — probability in each bucket")
    st.caption(
        "Buckets are sigma-anchored on the market smile. Enter the probability "
        "(in %) you assign to each bucket. The buckets together must sum to 100%."
    )

    if "bucket_0" not in st.session_state:
        reset_buckets_to_baseline()

    probs = []
    cols = st.columns(n_buckets)
    for i, col in enumerate(cols):
        with col:
            lo = boundaries[i]
            hi = boundaries[i + 1]
            sig_lo = offsets[i]
            sig_hi = offsets[i + 1]
            st.caption(
                f"{lo:.4f}–{hi:.4f}\n\n{sig_lo:+.2g}σ → {sig_hi:+.2g}σ"
            )
            key = f"bucket_{i}"
            if key not in st.session_state:
                st.session_state[key] = 1.0 / n_buckets
            v = st.number_input(
                "%",
                min_value=0.0, max_value=1.0, step=0.005,
                format="%.4f", key=key, label_visibility="collapsed",
            )
            probs.append(v)

    probs = np.array(probs, dtype=float)

    total = float(probs.sum())
    diff = total - 1.0
    if abs(diff) < 1e-6:
        st.success(f"Bucket probabilities sum to 1.0000 ✓")
    else:
        msg_col, btn_col = st.columns([3, 1])
        msg_col.warning(
            f"Bucket probabilities sum to {total:.4f}, not 1.0 (off by {diff:+.4f})."
        )
        if btn_col.button("Renormalise to 1"):
            if total > 0:
                for i in range(n_buckets):
                    st.session_state[f"bucket_{i}"] = float(probs[i] / total)
                st.rerun()

    return boundaries, probs


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


def render_edge_panel(rep, strike: float) -> None:
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
    st.caption("Edge vs market-implied pricing for a vanilla option. Kelly sizing deferred.")

    init_state()
    render_sidebar()

    n = st.session_state.n_anchors
    base = _current_baseline()

    if st.session_state.mode == MODE_OPTION1:
        quantiles = ANCHOR_PRESETS[n]
        prices = render_option1_inputs(quantiles)
        if not np.all(np.diff(prices) > 0):
            st.error("Anchor prices must be strictly increasing. Adjust the values above.")
            return
        pm = elicit_from_cdf_anchors(prices, list(quantiles))
        st.altair_chart(
            render_option1_chart(prices, np.asarray(quantiles), base),
            use_container_width=True,
        )
    else:
        n_buckets = n
        sigma_offsets = default_sigma_boundaries(n_buckets)
        boundaries_preview = sigma_boundaries_to_prices(
            sigma_offsets,
            forward=st.session_state.forward,
            sigma=st.session_state.sigma,
            tenor_years=st.session_state.tenor_years,
        )
        # Show the market-allocation snapshot above the inputs so PMs see the
        # baseline shape before they decide how to deviate from it.
        from viz import _market_mass_per_range  # local import to avoid widening public API
        market_probs_buckets = _market_mass_per_range(base, boundaries_preview)
        if market_probs_buckets.sum() > 0:
            market_probs_buckets = market_probs_buckets / market_probs_buckets.sum()

        boundaries, probs = render_option2_inputs(n)
        total = float(probs.sum())
        if abs(total - 1.0) > 1e-6:
            st.info(
                "Adjust the bucket values above (or click *Renormalise to 1*) before pricing."
            )
            st.altair_chart(
                render_option2_stacked_chart(probs / max(total, 1e-12), market_probs_buckets),
                use_container_width=True,
            )
            return
        if np.any(probs < 0):
            st.error("Bucket probabilities cannot be negative.")
            return
        pm = elicit_from_pdf_buckets(boundaries, probs)
        st.altair_chart(
            render_option2_stacked_chart(probs, market_probs_buckets),
            use_container_width=True,
        )
        st.altair_chart(
            render_option2_chart(boundaries, probs, base, sigma_offsets=sigma_offsets),
            use_container_width=True,
        )

    strike, is_call = render_structure_inputs()
    rep = compute_edge(pm, base, strike=strike, is_call=is_call, discount_factor=DISCOUNT_FACTOR)
    render_edge_panel(rep, strike)


if __name__ == "__main__":
    main()
