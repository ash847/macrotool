"""
Kelly v2 prototype — subjective distribution → edge vs market-implied.

Option 1 (CDF mode): PM enters the price level at each of N fixed quantiles.
Option 2 (PDF mode): PM enters the probability mass in each of N sigma-anchored
buckets, with bucket boundaries on the lognormal market smile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st

from analytics.distributions import interpolate_atm_vol
from data.snapshot_loader import load_snapshot as load_live_snapshot
from data.snapshot_overrides import apply_overrides
from pricing.forwards import interpolate_forward
from .baseline import synthetic_lognormal_baseline
from .edge import (
    anchors_from_baseline,
    compute_edge,
    compute_edge_for_payoff,
    shadow_market_from_cdf_anchors,
    shadow_market_from_pdf_buckets,
)
from .elicitation import (
    Distribution,
    default_sigma_boundaries,
    elicit_from_cdf_anchors,
    elicit_from_pdf_buckets,
    sigma_boundaries_to_prices,
)
from .kelly import compute_kelly, kelly_growth_curve
from .pricing import (
    base_ccy_payoff_for_trade_rec,
    call_payoff,
    put_payoff,
)
from .viz import (
    render_kelly_growth_curve,
    render_option1_chart,
    render_option2_chart,
)


PKG_ROOT = Path(__file__).resolve().parent
KELLY_SOURCE_STANDALONE = "Standalone"
KELLY_SOURCE_TRADE_REC = "From Trade Rec"

# Max recommended variants surfaced in the "From Trade Rec" dropdown. Variants are
# built in the same order Trade View renders them (selector_result.shortlist), so
# this is the first N of that list.
TRADE_REC_DROPDOWN_LIMIT = 20

MODE_OPTION1 = "Use fixed probability bins"
MODE_OPTION2 = "Use fixed spot ranges"

ANCHOR_PRESETS: dict[int, tuple[float, ...]] = {
    5:  (0.05, 0.25, 0.50, 0.75, 0.95),
    7:  (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    9:  (0.02, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.98),
    11: (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98),
}

DISCOUNT_FACTOR: float = 1.0
_HORIZON_OPTIONS: list[tuple[str, int]] = [
    (f"{month}M", round(month * 365 / 12)) for month in range(1, 13)
]


@dataclass(frozen=True)
class TradeRecCandidate:
    id: str
    pair: str
    horizon_days: int
    structure_id: str
    display_name: str
    structure_label: str
    variant_label: str
    strikes: list[float]
    barrier: float | None
    wing_ratio: float | None
    is_call: bool
    entry_spot: float
    forward: float
    sigma: float
    tenor_years: float
    net_premium_pct: float
    max_loss_pct: float


# --- session state ---


def init_state() -> None:
    defaults = {
        "kelly_source_mode": KELLY_SOURCE_STANDALONE,
        "mode": MODE_OPTION1,
        "n_anchors": 7,
        "kelly_pair": "USDBRL",
        "kelly_horizon": "3M",
        "forward": 5.00,
        "sigma": 0.10,
        "tenor_years": 0.25,
        "strike": 5.00,
        "option_type": "Call",
        "is_call": True,
        "kelly_multiplier": 50,
        "kelly_trade_rec_choice": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "_kelly_state_initialized" not in st.session_state:
        st.session_state._kelly_state_initialized = True


def _effective_snapshot():
    base = load_live_snapshot()
    overrides = st.session_state.get("market_overrides", {})
    return apply_overrides(base, overrides) if overrides else base


def _apply_baseline_signature(
    *,
    signature: tuple,
    forward: float,
    sigma: float,
    tenor_years: float,
    strike: float | None = None,
    is_call: bool | None = None,
) -> None:
    changed = st.session_state.get("_kelly_baseline_signature") != signature
    st.session_state.forward = float(forward)
    st.session_state.sigma = float(sigma)
    st.session_state.tenor_years = float(tenor_years)
    if strike is not None and (changed or "strike" not in st.session_state):
        st.session_state.strike = float(strike)
    if is_call is not None and changed:
        st.session_state.is_call = bool(is_call)
        st.session_state.option_type = "Call" if is_call else "Put"
    if changed:
        reset_anchors_to_baseline()
        reset_buckets_to_baseline()
        st.session_state._kelly_baseline_signature = signature


def _current_baseline() -> Distribution:
    return synthetic_lognormal_baseline(
        forward=st.session_state.forward,
        sigma=st.session_state.sigma,
        tenor_years=st.session_state.tenor_years,
        n_bins=400,
        n_stdev=6.0,
    )


def _standalone_is_call() -> bool:
    return st.session_state.get("option_type", "Call") == "Call"


def _pair_direction_caption(pair: str) -> str:
    base = pair[:3]
    quote = pair[3:]
    return (
        f"Call = {base} up / {quote} down (pair higher). "
        f"Put = {base} down / {quote} up (pair lower)."
    )


def _option_type_label(pair: str, kind: str) -> str:
    base = pair[:3]
    quote = pair[3:]
    if kind == "Call":
        return f"{base} up / {quote} down"
    return f"{base} down / {quote} up"


def _standalone_market_context() -> dict:
    snapshot = _effective_snapshot()
    pair_options = list(snapshot.currencies.keys())
    default_pair = "USDBRL" if "USDBRL" in pair_options else pair_options[0]
    if st.session_state.kelly_pair not in pair_options:
        st.session_state.kelly_pair = default_pair
    horizon_labels = [label for label, _ in _HORIZON_OPTIONS]
    if st.session_state.kelly_horizon not in horizon_labels:
        st.session_state.kelly_horizon = "3M"
    pair = st.session_state.kelly_pair
    horizon_days = dict(_HORIZON_OPTIONS)[st.session_state.kelly_horizon]
    ccy = snapshot.get(pair)
    tenor_years = horizon_days / 365.0
    forward = interpolate_forward(ccy, tenor_years)
    sigma = interpolate_atm_vol(ccy, horizon_days)
    _apply_baseline_signature(
        signature=(KELLY_SOURCE_STANDALONE, pair, horizon_days),
        forward=forward,
        sigma=sigma,
        tenor_years=tenor_years,
        strike=forward,
    )
    return {
        "pair_options": pair_options,
        "pair": pair,
        "ccy": ccy,
        "horizon_days": horizon_days,
        "horizon_label": st.session_state.kelly_horizon,
    }


def _live_trade_rec_candidates() -> list[TradeRecCandidate]:
    flow = st.session_state.get("flow")
    if not (
        flow
        and getattr(flow, "view", None)
        and getattr(flow, "market_state", None)
        and getattr(flow, "selector_result", None)
        and flow.selector_result.shortlist
    ):
        return []

    from interface.structure_eval import fmt_ccy, target_price, variant_label_with_strikes

    target = target_price(flow)
    if target is None:
        return []
    ms = flow.market_state
    is_call = flow.view.direction == "base_higher"
    supported_structure_ids = {
        "vanilla",
        "1x1_spread",
        "1x1.5_spread",
        "1x2_spread",
        "seagull",
        "european_digital",
        "european_digital_rko",
        "european_rko",
    }

    candidates: list[TradeRecCandidate] = []
    pair = flow.view.pair
    ranked_variants = st.session_state.get("kelly_ranked_trade_rec_variants")
    if ranked_variants:
        for rank, ranked_entry in enumerate(ranked_variants, start=1):
            item = ranked_entry.get("item")
            ev_v = ranked_entry.get("ev_v", {})
            pv = ev_v.get("pv")
            if item is None or pv is None or item.structure_id not in supported_structure_ids:
                continue
            variant_label = variant_label_with_strikes(item.structure_id, pv)
            notional = (
                f" · Notional {fmt_ccy(pv.structure_notional, pair[:3])}"
                if pv.structure_notional is not None
                else ""
            )
            display_name = f"{rank}. {variant_label}{notional}"
            candidates.append(
                TradeRecCandidate(
                    id=f"ranked:{item.structure_id}:{rank}",
                    pair=pair,
                    horizon_days=flow.view.horizon_days,
                    structure_id=item.structure_id,
                    display_name=display_name,
                    structure_label=item.display_name,
                    variant_label=variant_label,
                    strikes=list(pv.strikes),
                    barrier=pv.barrier,
                    wing_ratio=pv.wing_ratio,
                    is_call=is_call,
                    entry_spot=ms.spot,
                    forward=ms.fwd,
                    sigma=ms.vol,
                    tenor_years=ms.T,
                    net_premium_pct=pv.net_premium_pct,
                    max_loss_pct=pv.max_loss_pct,
                )
            )
            if len(candidates) >= 10:
                return candidates

    from analytics.structure_pricer import price_variants

    move_pct = abs(target - ms.fwd) / ms.fwd
    stop_pct = move_pct / flow.target_rr
    stop_price = ms.fwd * (1 - stop_pct) if is_call else ms.fwd * (1 + stop_pct)
    loss_budget = 100.0 * stop_pct

    for struct_rank, item in enumerate(flow.selector_result.shortlist, start=1):
        if item.structure_id not in supported_structure_ids:
            continue
        priced = price_variants(
            ms,
            item.structure_id,
            target=target,
            is_call=is_call,
            stop_price=stop_price,
            loss_budget=loss_budget,
        )
        for variant_rank, pv in enumerate(priced, start=1):
            variant_label = variant_label_with_strikes(item.structure_id, pv)
            notional = (
                f" · Notional {fmt_ccy(pv.structure_notional, pair[:3])}"
                if pv.structure_notional is not None
                else ""
            )
            display_name = f"{struct_rank}.{variant_rank} {variant_label}{notional}"
            candidates.append(
                TradeRecCandidate(
                    id=f"{item.structure_id}:{struct_rank}:{variant_rank}",
                    pair=pair,
                    horizon_days=flow.view.horizon_days,
                    structure_id=item.structure_id,
                    display_name=display_name,
                    structure_label=item.display_name,
                    variant_label=variant_label,
                    strikes=list(pv.strikes),
                    barrier=pv.barrier,
                    wing_ratio=pv.wing_ratio,
                    is_call=is_call,
                    entry_spot=ms.spot,
                    forward=ms.fwd,
                    sigma=ms.vol,
                    tenor_years=ms.T,
                    net_premium_pct=pv.net_premium_pct,
                    max_loss_pct=pv.max_loss_pct,
                )
            )
            if len(candidates) >= TRADE_REC_DROPDOWN_LIMIT:
                return candidates
    return candidates


def _selected_trade_rec_candidate() -> TradeRecCandidate | None:
    candidates = _live_trade_rec_candidates()
    if not candidates:
        st.session_state.kelly_trade_rec_choice = None
        return None
    ids = [candidate.id for candidate in candidates]
    if st.session_state.kelly_trade_rec_choice not in ids:
        st.session_state.kelly_trade_rec_choice = ids[0]
    selected = next(candidate for candidate in candidates if candidate.id == st.session_state.kelly_trade_rec_choice)
    _apply_baseline_signature(
        signature=(KELLY_SOURCE_TRADE_REC, selected.id),
        forward=selected.forward,
        sigma=selected.sigma,
        tenor_years=selected.tenor_years,
        strike=selected.strikes[0] if selected.strikes else selected.forward,
        is_call=selected.is_call,
    )
    return selected


def reset_anchors_to_baseline() -> None:
    """Option 1: seed anchor prices from the baseline CDF at the active quantiles."""
    quantiles = ANCHOR_PRESETS[st.session_state.n_anchors]
    seed = anchors_from_baseline(_current_baseline(), list(quantiles))
    for i, p in enumerate(seed):
        st.session_state[f"anchor_{i}"] = float(p)
    st.session_state["_kelly_anchor_widget_signature"] = (
        st.session_state.get("_kelly_baseline_signature"),
        st.session_state.n_anchors,
    )


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
    st.session_state["_kelly_bucket_widget_signature"] = (
        st.session_state.get("_kelly_baseline_signature"),
        st.session_state.n_anchors,
    )


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
    st.session_state["_kelly_bucket_widget_signature"] = (
        st.session_state.get("_kelly_baseline_signature"),
        st.session_state.n_anchors,
    )


def _ensure_anchor_widget_state(quantiles: tuple[float, ...]) -> None:
    expected = (
        st.session_state.get("_kelly_baseline_signature"),
        len(quantiles),
    )
    current = st.session_state.get("_kelly_anchor_widget_signature")
    missing = any(f"anchor_{i}" not in st.session_state for i in range(len(quantiles)))
    if current != expected or missing:
        reset_anchors_to_baseline()


def _ensure_bucket_widget_state(n_buckets: int) -> None:
    expected = (
        st.session_state.get("_kelly_baseline_signature"),
        n_buckets,
    )
    current = st.session_state.get("_kelly_bucket_widget_signature")
    missing = any(f"bucket_{i}" not in st.session_state for i in range(n_buckets))
    total = sum(int(st.session_state.get(f"bucket_{i}", 0)) for i in range(n_buckets))
    if current != expected or missing or total <= 0:
        reset_buckets_to_baseline()


def sync_on_n_change() -> None:
    """When N changes, re-seed whichever inputs the active mode depends on."""
    if st.session_state.mode == MODE_OPTION1:
        reset_anchors_to_baseline()
    else:
        reset_buckets_to_baseline()


# --- rendering ---


def render_sidebar() -> None:
    st.header("Elicitation mode")
    st.radio(
        "Mode",
        options=[MODE_OPTION1, MODE_OPTION2],
        key="mode",
        label_visibility="collapsed",
    )

    st.divider()
    st.header("Market baseline")
    if st.session_state.kelly_source_mode == KELLY_SOURCE_STANDALONE:
        snapshot = _effective_snapshot()
        st.selectbox("Pair", list(snapshot.currencies.keys()), key="kelly_pair")
        st.selectbox(
            "Horizon",
            [label for label, _ in _HORIZON_OPTIONS],
            key="kelly_horizon",
        )
        market = _standalone_market_context()
        col_spot, col_fwd = st.columns(2)
        col_spot.metric("Spot", f"{market['ccy'].spot:.4f}")
        col_fwd.metric("Forward", f"{st.session_state.forward:.4f}")
        col_vol, col_tenor = st.columns(2)
        col_vol.metric("ATM vol", f"{st.session_state.sigma:.1%}")
        col_tenor.metric("Tenor", market["horizon_label"])
    else:
        candidate = _selected_trade_rec_candidate()
        if candidate is None:
            st.info("No live Trade Rec found. Run Trade View first or switch to Standalone.")
        else:
            st.caption(f"{candidate.pair} · {candidate.horizon_days}d")
            st.caption(candidate.structure_label)
            st.caption(candidate.variant_label)
            col_spot, col_fwd = st.columns(2)
            col_spot.metric("Spot", f"{candidate.entry_spot:.4f}")
            col_fwd.metric("Forward", f"{candidate.forward:.4f}")
            col_vol, col_loss = st.columns(2)
            col_vol.metric("ATM vol", f"{candidate.sigma:.1%}")
            col_loss.metric("Max loss", f"{candidate.max_loss_pct:.1%}")

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
    st.markdown("##### Your view — price at each quantile (must strictly increase)")

    _ensure_anchor_widget_state(quantiles)

    cols = st.columns(len(quantiles))
    for i, (q, col) in enumerate(zip(quantiles, cols)):
        with col:
            key = f"anchor_{i}"
            if key not in st.session_state:
                st.session_state[key] = float(st.session_state.forward)
            st.number_input(
                f"P ≤ {int(round(q * 100))}%",
                min_value=0.0001, step=0.01, format="%.4f", key=key,
            )
    return np.array(
        [float(st.session_state[f"anchor_{i}"]) for i in range(len(quantiles))],
        dtype=float,
    )


def render_option2_inputs(n_buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Render Option 2 bucket inputs. Returns (boundaries_in_prices, bucket_probs)."""
    offsets = default_sigma_boundaries(n_buckets)
    boundaries = sigma_boundaries_to_prices(
        offsets,
        forward=st.session_state.forward,
        sigma=st.session_state.sigma,
        tenor_years=st.session_state.tenor_years,
    )

    st.markdown("##### Your view — probability (%) per bucket (sum to 100)")

    _ensure_bucket_widget_state(n_buckets)

    # Offset the input columns by a narrow spacer to visually align them with
    # the bars in the chart below (which has a y-axis of roughly 35 px).
    # Using 0.25 units gives ~35 px on a typical wide-layout content area.
    all_cols = st.columns([0.25] + [1] * n_buckets)
    cols = all_cols[1:]  # skip the spacer
    for i, col in enumerate(cols):
        with col:
            key = f"bucket_{i}"
            if key not in st.session_state:
                st.session_state[key] = int(round(100 / n_buckets))
            st.number_input(
                "%",
                min_value=0, max_value=100, step=1,
                format="%d", key=key, label_visibility="collapsed",
            )
    probs_pct = np.array(
        [int(st.session_state[f"bucket_{i}"]) for i in range(n_buckets)],
        dtype=int,
    )
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


def render_vanilla_inputs() -> tuple[float, bool]:
    pair = st.session_state.get("kelly_pair", "FX pair")
    st.markdown("##### Vanilla option to price")
    st.caption(_pair_direction_caption(pair))
    col_k, col_t = st.columns([3, 2])
    with col_k:
        strike = st.number_input(
            "Strike", min_value=0.0001, step=0.01, format="%.4f", key="strike",
            label_visibility="collapsed",
        )
    with col_t:
        kind = st.radio(
            "Type",
            options=["Call", "Put"],
            format_func=lambda choice: _option_type_label(pair, choice),
            horizontal=True,
            key="option_type",
            label_visibility="collapsed",
        )
        is_call = kind == "Call"
        st.session_state.is_call = is_call
    return strike, is_call


def render_trade_rec_selector() -> TradeRecCandidate | None:
    candidates = _live_trade_rec_candidates()
    if not candidates:
        st.info("No live Trade Rec found in this session yet. Run Trade View first or switch back to Standalone.")
        return None
    ids = [candidate.id for candidate in candidates]
    if st.session_state.kelly_trade_rec_choice not in ids:
        st.session_state.kelly_trade_rec_choice = ids[0]
    st.selectbox(
        "Recommended trade",
        options=ids,
        key="kelly_trade_rec_choice",
        format_func=lambda candidate_id: next(
            candidate.display_name for candidate in candidates if candidate.id == candidate_id
        ),
    )
    candidate = next(
        candidate for candidate in candidates if candidate.id == st.session_state.kelly_trade_rec_choice
    )
    _selected_trade_rec_candidate()
    return candidate


def render_trade_rec_summary(candidate: TradeRecCandidate) -> None:
    st.markdown("##### Linked Trade Rec")
    st.caption(f"{candidate.pair} · {candidate.horizon_days}d · {candidate.structure_label}")
    st.markdown(candidate.variant_label)
    col_spot, col_fwd, col_vol, col_loss = st.columns(4)
    col_spot.metric("Spot", f"{candidate.entry_spot:.4f}")
    col_fwd.metric("Forward", f"{candidate.forward:.4f}")
    col_vol.metric("ATM vol", f"{candidate.sigma:.1%}")
    col_loss.metric("Max loss", f"{candidate.max_loss_pct:.1%}")


def render_edge_panel(rep, *, out_of_range_label: str | None = None) -> None:
    if rep.out_of_range and out_of_range_label:
        st.warning(
            f"{out_of_range_label} is outside your elicited support — your "
            f"distribution puts no mass beyond the outer anchors. Widen the "
            f"outer anchors if you actually have a view at that level."
        )

    def _fmt_pct(pct):
        return f"{pct:+.1f}% of mid" if pct is not None else "—"

    # Four columns: PM price | Truncated market | Full market | View edge
    # Showing truncated market makes the view edge arithmetic visible:
    # PM price − Truncated market = View edge (should be positive when PM > truncated mkt)
    col_pm, col_shadow, col_mkt, col_edge = st.columns([1, 1, 1, 2])
    col_pm.metric("PM price", f"{rep.pm_price:.4f}")
    col_shadow.metric(
        "Truncated market",
        f"{rep.shadow_price:.4f}",
        help="Market price computed over your anchor range only — same truncation as your view.",
    )
    col_mkt.metric("Full market price", f"{rep.mkt_price:.4f}")
    col_edge.metric(
        "View edge",
        f"{rep.view_edge:+.4f}",
        _fmt_pct(rep.view_edge_pct_of_mid),
        help="PM price − Truncated market. Positive when your distribution is richer than market over your anchor range.",
    )

    # Anchoring cost: the gap between truncated market and full market price.
    # Shown as an accuracy indicator — tells the PM how much the dropped tails matter.
    anc_pct_str = ""
    if rep.mkt_price > 1e-10:
        anc_pct = rep.anchoring_cost / rep.mkt_price * 100.0
        anc_pct_str = f" ({anc_pct:+.1f}% of mid)"
    st.caption(
        f"**Distribution accuracy — truncation error: {rep.anchoring_cost:+.4f}{anc_pct_str}.** "
        f"Gap between truncated and full market price — pricing impact of the ~4% tail mass "
        f"outside your outer anchors. Measurement artifact; does not affect your trade P&L. "
        "Large values mean the tails matter for the selected payoff."
    )


def render_kelly_panel(rep_kelly, pm, payoff, cost_basis: float, *, cost_label: str) -> None:
    st.markdown("##### Kelly sizing")

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

    # Full Kelly + log-growth on one row
    col_fstar, col_growth = st.columns([1, 1])
    col_fstar.metric("Full Kelly (f*)", f"{rep_kelly.f_discrete:.1%}")
    col_growth.metric("Expected log-growth at your fraction", f"{rep_kelly.expected_log_growth:+.4f}")

    # Multiplier input — left-aligned, integer %, 1% steps, no buttons
    col_num, col_result, _ = st.columns([1, 1, 2])
    col_num.number_input(
        "Multiplier (%)", min_value=10, max_value=100, step=1, format="%d",
        key="kelly_multiplier",
    )
    col_result.markdown(f"→ &nbsp; **{rep_kelly.f_displayed:.1%}** of bankroll")

    f_vals, geo_vals, er_vals = kelly_growth_curve(
        pm, payoff, cost=cost_basis, discount_factor=DISCOUNT_FACTOR,
        f_star=rep_kelly.f_raw,
    )
    st.altair_chart(
        render_kelly_growth_curve(f_vals, geo_vals, er_vals, rep_kelly.f_raw, rep_kelly.f_displayed),
        use_container_width=True,
    )
    st.caption(f"Kelly cost basis: **{cost_label}**.")

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
            "with the truncation artifact stripped out."
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


def render_page() -> None:
    st.markdown(_COMPACT_CSS, unsafe_allow_html=True)
    st.markdown("##### Kelly v2 — Subjective Distribution → Edge")

    init_state()
    st.segmented_control(
        "Source mode",
        options=[KELLY_SOURCE_STANDALONE, KELLY_SOURCE_TRADE_REC],
        key="kelly_source_mode",
        selection_mode="single",
    )

    trade_rec_candidate: TradeRecCandidate | None = None
    if st.session_state.kelly_source_mode == KELLY_SOURCE_STANDALONE:
        market = _standalone_market_context()
        st.caption(
            f"Standalone market baseline: {market['pair']} · {market['horizon_label']} · "
            f"spot {market['ccy'].spot:.4f} · forward {st.session_state.forward:.4f} · "
            f"ATM vol {st.session_state.sigma:.1%}"
        )
    else:
        trade_rec_candidate = render_trade_rec_selector()
        if trade_rec_candidate is None:
            return
        render_trade_rec_summary(trade_rec_candidate)

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
        boundaries, probs = render_option2_inputs(n)

        # Chart renders even when sum != 1 so PMs can see what they're entering.
        # Pricing is gated on sum-to-1 below.
        st.altair_chart(
            render_option2_chart(boundaries, probs, base),
            use_container_width=True,
        )

        total = float(probs.sum())
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

    if st.session_state.kelly_source_mode == KELLY_SOURCE_STANDALONE:
        strike, is_call = render_vanilla_inputs()
        payoff = call_payoff(strike) if is_call else put_payoff(strike)
        rep = compute_edge(pm, base, shadow, strike=strike, is_call=is_call, discount_factor=DISCOUNT_FACTOR)
        render_edge_panel(rep, out_of_range_label=f"Strike {strike:.4f}")
        cost_basis = rep.shadow_price
        cost_label = "truncated market price"
    else:
        assert trade_rec_candidate is not None
        payoff = base_ccy_payoff_for_trade_rec(
            trade_rec_candidate.structure_id,
            strikes=trade_rec_candidate.strikes,
            barrier=trade_rec_candidate.barrier,
            is_call=trade_rec_candidate.is_call,
            entry_spot=trade_rec_candidate.entry_spot,
            wing_ratio=trade_rec_candidate.wing_ratio,
        )
        ref_levels = trade_rec_candidate.strikes + (
            [trade_rec_candidate.barrier] if trade_rec_candidate.barrier is not None else []
        )
        rep = compute_edge_for_payoff(
            pm,
            base,
            payoff,
            shadow_dist=shadow,
            discount_factor=DISCOUNT_FACTOR,
            reference_levels=ref_levels,
        )
        render_edge_panel(
            rep,
            out_of_range_label=f"Selected trade ({trade_rec_candidate.structure_label})",
        )
        if rep.shadow_price > 1e-10:
            cost_basis = rep.shadow_price
            cost_label = "truncated market price"
        else:
            cost_basis = trade_rec_candidate.max_loss_pct
            cost_label = "max-loss capital proxy"

    st.markdown("---")
    if cost_basis > 1e-10:
        rep_kelly = compute_kelly(
            pm, payoff, cost=cost_basis, discount_factor=DISCOUNT_FACTOR,
            multiplier=st.session_state.kelly_multiplier / 100.0,
        )
        render_kelly_panel(rep_kelly, pm, payoff, cost_basis, cost_label=cost_label)
    else:
        st.info("Kelly sizing is unavailable because the selected trade has no positive cost basis yet.")


def main() -> None:
    st.set_page_config(page_title="Kelly v2 — Subjective Edge", layout="wide")
    init_state()
    with st.sidebar:
        render_sidebar()
    render_page()


if __name__ == "__main__":
    main()
