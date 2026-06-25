"""Inline Kelly edge elicitation for the Trade View page.

Mirrors the Kelly Sizing screen's elicitation (CDF quantiles / fixed-range PDF
buckets, bucket-count dropdown, the same Altair charts, the 100% checksum), but
inline on Trade View — writing the result to ``st.session_state.kelly_probs /
kelly_bins`` (consumed by ``build_sizing_spec``). Reuses the pure elicitation
functions + chart renderers; the standalone Kelly screen is untouched.

The **market-implied** distribution (lognormal at the forward, ATM vol) is the
baseline AND the seed — so at inception the elicited distribution equals the
market one; the PM moves mass to express their edge. Re-seeds on context change
or via the "Reset to market baseline" button.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from analytics.sizing import view_implied_distribution
from interface.kelly_v2.elicitation import (
    Distribution,
    default_sigma_boundaries,
    elicit_from_cdf_anchors,
    elicit_from_pdf_buckets,
    sigma_boundaries_to_prices,
)
from interface.kelly_v2.viz import render_option1_chart, render_option2_chart

_CDF = "Quantiles (CDF)"
_PDF = "Fixed-range buckets (PDF)"
# Bucket-count presets — same dropdown options as the Kelly tab (ANCHOR_PRESETS).
_QUANTILE_PRESETS: dict[int, tuple[float, ...]] = {
    5:  (0.05, 0.25, 0.50, 0.75, 0.95),
    7:  (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    9:  (0.02, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.98),
    11: (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98),
}
_N_OPTIONS = sorted(_QUANTILE_PRESETS.keys())
_N_BINS = 400          # output resolution of the discretised distribution (matches Kelly tab; not user-facing)
_KP = "tvk_"           # session-key prefix, isolated from the Kelly screen


def _largest_remainder_round(values: np.ndarray, total: int = 100) -> np.ndarray:
    """Round to integers summing exactly to `total` (largest-remainder)."""
    v = np.asarray(values, dtype=float)
    out = np.zeros(v.size, dtype=int)
    if v.sum() <= 0:
        if v.size:
            out[v.size // 2] = total
        return out
    scaled = v / v.sum() * total
    floor = np.floor(scaled).astype(int)
    for i in np.argsort(-(scaled - floor))[: int(total - floor.sum())]:
        floor[i] += 1
    return floor


def _seed_bucket_pct(bp: np.ndarray, bb: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Integer % per bucket from the baseline mass, summing to 100."""
    masses = np.array([
        float(bp[(bb >= boundaries[i]) & (bb < boundaries[i + 1])].sum())
        for i in range(len(boundaries) - 1)
    ])
    return _largest_remainder_round(masses, 100)


def _renormalise(n: int) -> None:
    raw = np.array([float(st.session_state[_KP + f"bucket_{i}"]) for i in range(n)])
    rounded = _largest_remainder_round(raw, 100)
    for i in range(n):
        st.session_state[_KP + f"bucket_{i}"] = int(rounded[i])


def render_kelly_elicitation(ms, target: float | None = None, direction: str | None = None):
    """Render the elicitation block (inputs + chart + means); return (probs, bins) or (None, None)."""
    # Market-implied baseline = lognormal centred at the forward (ATM vol). Both the
    # baseline series and the elicitation inputs seed from this, so at inception the
    # elicited distribution matches the market one.
    mk_probs, mk_bins = view_implied_distribution(ms.spot, ms.fwd, ms.vol, ms.T, ms.fwd)
    bp, bb = np.array(mk_probs), np.array(mk_bins)
    baseline = Distribution(bins=bb, probs=bp)

    st.markdown("**Your edge — terminal-spot distribution**")
    c_mode, c_n = st.columns([2, 1])
    mode = c_mode.radio("Input style", [_CDF, _PDF], horizontal=True, key=_KP + "mode")
    n = int(c_n.selectbox("Buckets", _N_OPTIONS, index=0, key=_KP + "n"))

    sig = (round(ms.fwd, 6), round(ms.vol, 6), round(ms.T, 6), mode, n)
    reseed = st.session_state.get(_KP + "sig") != sig
    if st.button("Reset to market baseline", key=_KP + "reset"):
        reseed = True
    st.session_state[_KP + "sig"] = sig

    if mode == _CDF:
        quantiles = _QUANTILE_PRESETS[n]
        cdf = np.cumsum(bp)
        seed = [float(np.interp(q, cdf, bb)) for q in quantiles]
        # Market reference = the baseline through the SAME elicitation, so at inception
        # (inputs == seed) the reference mean equals the elicited mean.
        seed_dist = elicit_from_cdf_anchors(seed, list(quantiles), n_bins=_N_BINS)
        cols = st.columns(n)
        prices: list[float] = []
        for i, (q, c) in enumerate(zip(quantiles, cols)):
            k = _KP + f"anchor_{i}"
            if reseed or k not in st.session_state:
                st.session_state[k] = round(seed[i], 4)
            prices.append(c.number_input(f"{int(q*100)}%", format="%.4f", key=k))
        try:
            dist = elicit_from_cdf_anchors(prices, list(quantiles), n_bins=_N_BINS)
        except ValueError as e:
            st.warning(f"Quantile prices must strictly increase — {e}")
            return None, None
        st.altair_chart(render_option1_chart(np.array(prices), np.array(quantiles), baseline),
                        use_container_width=True)
    else:
        boundaries = sigma_boundaries_to_prices(default_sigma_boundaries(n), forward=ms.fwd,
                                                sigma=ms.vol, tenor_years=ms.T)
        seed_pct = _seed_bucket_pct(bp, bb, boundaries)
        # Market reference through the SAME elicitation (see CDF branch).
        seed_dist = elicit_from_pdf_buckets(
            list(boundaries), list(seed_pct / max(seed_pct.sum(), 1)), n_bins=_N_BINS,
        )
        cols = st.columns(n)
        probs_pct: list[int] = []
        for i in range(n):
            k = _KP + f"bucket_{i}"
            if reseed or k not in st.session_state:
                st.session_state[k] = int(seed_pct[i]) if i < len(seed_pct) else 0
            probs_pct.append(int(cols[i].number_input(
                f"{boundaries[i]:.2f}", min_value=0, max_value=100, step=1, format="%d", key=k,
            )))
        total = int(sum(probs_pct))
        if total == 100:
            st.success("Bucket probabilities sum to 100% ✓")
        else:
            m_col, b_col = st.columns([3, 1])
            m_col.warning(f"Bucket probabilities sum to {total}%, not 100% (off by {total - 100:+d}%).")
            b_col.button("Renormalise to 100%", key=_KP + "renorm", on_click=_renormalise, args=(n,))
        if total <= 0:
            return None, None
        probs_norm = np.array(probs_pct, dtype=float) / total
        try:
            dist = elicit_from_pdf_buckets(list(boundaries), list(probs_norm), n_bins=_N_BINS)
        except ValueError as e:
            st.warning(str(e))
            return None, None
        st.altair_chart(render_option2_chart(boundaries, probs_norm, baseline),
                        use_container_width=True)

    probs = tuple(float(p) for p in dist.probs)
    bins = tuple(float(b) for b in dist.bins)
    st.session_state.kelly_probs = probs
    st.session_state.kelly_bins = bins

    # Market reference uses the same elicitation lens as the elicited curve, so the
    # delta is purely the PM's move (zero at inception, not a truncation artefact).
    mk_mean = float(np.dot(seed_dist.probs, seed_dist.bins))
    el_mean = float(np.dot(np.array(probs), np.array(bins)))
    c_mk, c_el = st.columns(2)
    c_mk.metric("Market-implied mean", f"{mk_mean:.4f}",
                help="Market baseline through the same elicitation — the starting point you move from.")
    c_el.metric("Your elicited mean", f"{el_mean:.4f}", delta=f"{el_mean - mk_mean:+.4f}")

    # Directional-consistency hint: warn only when the elicited mean sits clearly on
    # the opposite side of the forward from the trade direction (deadband = 10% of a
    # 1σ move, so a roughly-flat view doesn't nag). Advisory — Kelly still does the
    # actual sizing (an adverse mean → small/zero size, shown per-variant).
    if direction in ("base_higher", "base_lower"):
        deadband = 0.10 * ms.vol * (ms.T ** 0.5) * ms.fwd
        is_long = direction == "base_higher"
        against = (is_long and el_mean < ms.fwd - deadband) or \
                  ((not is_long) and el_mean > ms.fwd + deadband)
        if against:
            side = "long (base higher)" if is_long else "short (base lower)"
            lean = "below" if is_long else "above"
            st.warning(
                f"⚠ Your distribution's mean ({el_mean:.4f}) is {lean} the forward "
                f"({ms.fwd:.4f}) — against this {side} view. Kelly will size this small or to zero."
            )
    return probs, bins
