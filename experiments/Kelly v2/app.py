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
from edge import (
    anchors_from_baseline,
    compute_edge,
    shadow_market_from_cdf_anchors,
    shadow_market_from_pdf_buckets,
)
from elicitation import (
    Distribution,
    default_sigma_boundaries,
    elicit_from_cdf_anchors,
    elicit_from_pdf_buckets,
    sigma_boundaries_to_prices,
)
from kelly import compute_kelly
from pricing import call_payoff, forward_of, put_payoff
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
        "kelly_multiplier": 0.50,
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


def _largest_remainder_round(values: np.ndarray, target_sum: int) -> np.ndarray:
    """Round float allocations to ints that sum to exactly target_sum.

    Largest-remainder (Hamilton) method — floor everything, then award the
    remaining units to the entries with the largest fractional remainders.
    """
    floored = np.floor(values).astype(int)
    remainders = values - floored
    deficit = int(target_sum - floored.sum())
    if deficit > 0:
        order = np.argsort(-remainders)
        for i in range(deficit):
            floored[order[i]] += 1
    elif deficit < 0:
        order = np.argsort(remainders)  # smallest remainders first
        for i in range(-deficit):
            floored[order[i]] -= 1
    return floored


def reset_buckets_to_uniform() -> None:
    """Option 2: seed bucket probabilities to uniform integer-percent."""
    n = st.session_state.n_anchors
    base_value = 100 // n
    remainder = 100 - base_value * n
    for i in range(n):
        st.session_state[f"bucket_{i}"] = base_value + (1 if i < remainder else 0)


def reset_buckets_to_baseline() -> None:
    """Option 2: seed bucket probabilities (as integer %) to the baseline mass per bucket."""
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
        masses = masses / masses.sum() * 100.0  # → percent
    rounded = _largest_remainder_round(masses, 100)
    for i, m in enumerate(rounded):
        st.session_state[f"bucket_{i}"] = int(m)


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

        st.divider()
        st.header("Kelly sizing")
        st.slider(
            "Kelly multiplier",
            min_value=0.10, max_value=1.00, step=0.05,
            key="kelly_multiplier",
            help="Fractional Kelly. Default 0.5 (half-Kelly). Full Kelly is famously aggressive.",
        )


def render_option1_inputs(quantiles: tuple[float, ...]) -> np.ndarray:
    st.markdown("##### Your view — price at each quantile (must strictly increase)")

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

    st.markdown("##### Your view — probability (%) per σ-anchored bucket (sum to 100)")

    if "bucket_0" not in st.session_state:
        reset_buckets_to_baseline()

    probs_pct = []
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
                st.session_state[key] = int(round(100 / n_buckets))
            v = st.number_input(
                "%",
                min_value=0, max_value=100, step=1,
                format="%d", key=key, label_visibility="collapsed",
            )
            probs_pct.append(v)

    probs_pct = np.array(probs_pct, dtype=int)
    # Engine and pricing layer expect fractional probabilities in [0, 1].
    probs = probs_pct.astype(float) / 100.0

    total_pct = int(probs_pct.sum())
    if total_pct == 100:
        st.success("Bucket probabilities sum to 100% ✓")
    else:
        msg_col, btn_col = st.columns([3, 1])
        diff = total_pct - 100
        msg_col.warning(
            f"Bucket probabilities sum to {total_pct}%, not 100% (off by {diff:+d}%)."
        )
        btn_col.button(
            "Renormalise to 100%",
            on_click=_renormalise_buckets,
            args=(n_buckets,),
        )

    return boundaries, probs


def _renormalise_buckets(n_buckets: int) -> None:
    """Proportionally rescale bucket_i percent values to sum to exactly 100.

    Runs as a button `on_click` callback so Streamlit applies the writes
    BEFORE the next rerun instantiates the bucket widgets. Uses largest-
    remainder rounding so the integer percents sum to exactly 100.
    """
    raw = np.array([float(st.session_state[f"bucket_{i}"]) for i in range(n_buckets)])
    total = raw.sum()
    if total <= 0:
        return
    scaled = raw / total * 100.0
    rounded = _largest_remainder_round(scaled, 100)
    for i in range(n_buckets):
        st.session_state[f"bucket_{i}"] = int(rounded[i])


def render_structure_inputs() -> tuple[float, bool]:
    st.markdown("##### Option to price")
    col_k, col_t = st.columns([3, 2])
    with col_k:
        strike = st.number_input(
            "Strike", min_value=0.0001, step=0.01, format="%.4f", key="strike",
            label_visibility="collapsed",
        )
    with col_t:
        if "option_type" not in st.session_state:
            st.session_state.option_type = "Call" if st.session_state.is_call else "Put"
        kind = st.radio(
            "Type", options=["Call", "Put"], horizontal=True, key="option_type",
            label_visibility="collapsed",
        )
        is_call = kind == "Call"
        st.session_state.is_call = is_call
    return strike, is_call


def render_edge_panel(rep, strike: float) -> None:
    if rep.out_of_range:
        st.warning(
            f"Strike {strike:.4f} is outside your elicited support — your "
            f"distribution puts no mass beyond the outer anchors. Widen the "
            f"outer anchors if you actually have a view at this strike."
        )

    def _fmt_pct(pct):
        return f"{pct:+.1f}% of mid" if pct is not None else "—"

    col_pm, col_mkt, col_edge = st.columns([1, 1, 2])
    col_pm.metric("PM price", f"{rep.pm_price:.4f}")
    col_mkt.metric("Market price", f"{rep.mkt_price:.4f}")
    col_edge.metric(
        "View edge — your view-divergence vs market",
        f"{rep.view_edge:+.4f}",
        _fmt_pct(rep.view_edge_pct_of_mid),
    )

    # Anchoring cost shown as an accuracy / measurement-quality indicator,
    # not as a P&L signal. It tells the PM how much the ~4% dropped tails
    # distort the pricing, so they can judge how reliable the view edge number is.
    anc_pct_str = ""
    if rep.mkt_price > 1e-10:
        anc_pct = rep.anchoring_cost / rep.mkt_price * 100.0
        anc_pct_str = f" ({anc_pct:+.1f}% of mid)"
    st.caption(
        f"**Distribution accuracy — truncation error: {rep.anchoring_cost:+.4f}{anc_pct_str}.** "
        f"Pricing impact of the ~4% tail mass outside your outer anchors. "
        f"This is a measurement artifact — it does not affect your trade P&L. "
        f"Large values mean the tails matter for this strike; small values mean "
        f"the view edge number is a clean read."
    )


def render_kelly_panel(rep_kelly) -> None:
    if rep_kelly.unbounded_loss:
        st.warning(
            "This structure has potential losses that exceed the premium by a large multiple. "
            "Kelly sizing requires bounded downside — add a defined stop (e.g. an outer wing) "
            "before sizing this trade."
        )
        return

    if rep_kelly.expected_return <= 0:
        st.info(
            "Expected return under your distribution is ≤ 0 — Kelly says don't take the trade. "
            f"(E[r] = {rep_kelly.expected_return:+.2%})"
        )
        return

    col_disp, col_raw, col_growth = st.columns([2, 1, 1])
    col_disp.metric(
        f"Kelly fraction (× {rep_kelly.multiplier:.2f} of full Kelly)",
        f"{rep_kelly.f_displayed:.1%}",
    )
    col_raw.metric("Full Kelly (before multiplier)", f"{rep_kelly.f_discrete:.1%}")
    col_growth.metric("Expected log-growth", f"{rep_kelly.expected_log_growth:+.4f}")

    with st.expander("Kelly breakdown — solvers and risk metrics", expanded=False):
        st.caption(
            "**r = (payoff(S_T) − cost) / cost** — return on premium over the option's life "
            "(not annualised). r = −1 means total loss of premium; r = +1 means you double your money."
        )
        col_c, col_d = st.columns(2)
        col_c.metric("Continuous (Thorp)", f"{rep_kelly.f_continuous:.1%}")
        col_c.caption(
            "Closed-form `E[r] / Var[r]`. Taylor expansion — breaks down when |r| is large."
        )
        col_d.metric("Discrete (numerical)", f"{rep_kelly.f_discrete:.1%}")
        col_d.caption(
            "Numerical max of `E[log(1 + f·r)]`. Canonical Kelly — trust this."
        )

        st.markdown("**Return distribution under your view:**")
        col_e, col_v, col_l, col_t = st.columns(4)
        col_e.metric("E[r] (over tenor, not annualised)", f"{rep_kelly.expected_return:+.2%}")
        col_v.metric("Variance of r", f"{rep_kelly.variance:.3f}")
        col_l.metric("P(loss)", f"{rep_kelly.prob_loss:.1%}")
        col_t.metric("P(total loss)", f"{rep_kelly.prob_total_loss:.1%}")

        st.caption(
            "Sized on **view edge** — your view-divergence within the elicited range, "
            "with the truncation artifact stripped out. Adjust the sidebar sliders "
            "to see how multiplier and cap change the displayed fraction."
        )


_COMPACT_CSS = """
<style>
section.main > div.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100% !important; }
h1 { font-size: 1.45rem !important; margin-bottom: 0.15rem !important; }
h2 { font-size: 1.05rem !important; margin: 0.25rem 0 0.15rem 0 !important; }
h3 { font-size: 0.95rem !important; margin: 0.20rem 0 0.10rem 0 !important; }
h5 { font-size: 0.90rem !important; margin: 0.20rem 0 0.10rem 0 !important; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.80rem !important; }
[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.80rem !important; }
[data-testid="stCaptionContainer"] { font-size: 0.78rem !important; line-height: 1.1 !important; }
.stNumberInput label, .stRadio label, .stSelectbox label { font-size: 0.78rem !important; }
.stNumberInput input { padding: 0.20rem 0.35rem !important; }
.element-container { margin-bottom: 0.15rem !important; }
hr { margin: 0.35rem 0 !important; }
.stAlert { padding: 0.35rem 0.6rem !important; }
.stAlert p { margin: 0 !important; font-size: 0.80rem !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 0.8rem !important; }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Kelly v2 — Subjective Edge", layout="wide")
    st.markdown(_COMPACT_CSS, unsafe_allow_html=True)
    st.markdown("##### Kelly v2 — Subjective Distribution → Edge")

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
        shadow = shadow_market_from_cdf_anchors(base, list(quantiles))
        st.altair_chart(
            render_option1_chart(prices, np.asarray(quantiles), base),
            use_container_width=True,
        )
    else:
        n_buckets = n
        sigma_offsets = default_sigma_boundaries(n_buckets)

        boundaries, probs = render_option2_inputs(n)

        # Charts always render — even when sum != 1 — so PMs can see what
        # they're entering as they type. Pricing is gated on sum-to-1.
        from viz import _market_mass_per_range  # local import to avoid widening public API
        market_probs_buckets = _market_mass_per_range(base, boundaries)
        if market_probs_buckets.sum() > 0:
            market_probs_buckets = market_probs_buckets / market_probs_buckets.sum()

        total = float(probs.sum())
        # For the stacked allocation chart we display normalised user probs
        # so the row width stays at 100% even mid-edit; this is purely visual.
        display_probs = probs / total if total > 1e-12 else probs

        st.altair_chart(
            render_option2_stacked_chart(display_probs, market_probs_buckets),
            use_container_width=True,
        )
        st.altair_chart(
            render_option2_chart(boundaries, probs, base, sigma_offsets=sigma_offsets),
            use_container_width=True,
        )

        if np.any(probs < 0):
            st.error("Bucket probabilities cannot be negative.")
            return
        if abs(total - 1.0) > 1e-6:
            st.info(
                "Adjust the bucket values above (or click *Renormalise to 100%*) before pricing."
            )
            return
        pm = elicit_from_pdf_buckets(boundaries, probs)
        shadow = shadow_market_from_pdf_buckets(base, boundaries)

    strike, is_call = render_structure_inputs()
    rep = compute_edge(pm, base, shadow, strike=strike, is_call=is_call, discount_factor=DISCOUNT_FACTOR)
    render_edge_panel(rep, strike)

    payoff = call_payoff(strike) if is_call else put_payoff(strike)
    # Kelly sizes on view edge: use shadow_price as the cost basis so that
    # r = (DF·payoff − shadow_price) / shadow_price and E[r] = view_edge / shadow_price.
    # This treats the truncation artifact as measurement noise, not real edge.
    if rep.shadow_price > 1e-10:
        rep_kelly = compute_kelly(
            pm, payoff, cost=rep.shadow_price, discount_factor=DISCOUNT_FACTOR,
            multiplier=float(st.session_state.kelly_multiplier),
        )
        render_kelly_panel(rep_kelly)


if __name__ == "__main__":
    main()
