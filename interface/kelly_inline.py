"""Inline Kelly edge elicitation for the Trade View page.

Mirrors the Kelly Sizing screen's elicitation (CDF quantiles / fixed-range PDF
buckets, with a bucket-count dropdown and the same Altair charts), but inline on
Trade View — writing the result to ``st.session_state.kelly_probs / kelly_bins``
(consumed by ``build_sizing_spec``). Reuses the pure elicitation functions + the
chart renderers; the standalone Kelly screen is untouched. Inputs seed from the
view-implied distribution and re-seed on context change (or via reset).
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


def _seed_bucket_masses(bp: np.ndarray, bb: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    masses = np.array([
        float(bp[(bb >= boundaries[i]) & (bb < boundaries[i + 1])].sum())
        for i in range(len(boundaries) - 1)
    ])
    return (masses / masses.sum() * 100.0) if masses.sum() > 0 else masses


def render_kelly_elicitation(ms, target: float, conviction: str):
    """Render the elicitation block (inputs + chart); return (probs, bins) or (None, None)."""
    base_probs, base_bins = view_implied_distribution(ms.spot, ms.fwd, ms.vol, ms.T, target, conviction)
    bp, bb = np.array(base_probs), np.array(base_bins)
    # Market-implied baseline (centred at the forward) — the chart's reference series.
    mk_probs, mk_bins = view_implied_distribution(ms.spot, ms.fwd, ms.vol, ms.T, ms.fwd, conviction)
    baseline = Distribution(bins=np.array(mk_bins), probs=np.array(mk_probs))

    st.markdown("**Your edge — terminal-spot distribution**")
    c_mode, c_n = st.columns([2, 1])
    mode = c_mode.radio("Input style", [_CDF, _PDF], horizontal=True, key=_KP + "mode")
    n = int(c_n.selectbox("Buckets", _N_OPTIONS, index=0, key=_KP + "n"))

    # Re-seed inputs when the market/view context or shape changes, or on explicit reset.
    sig = (round(ms.fwd, 6), round(ms.vol, 6), round(target, 6), conviction, mode, n)
    reseed = st.session_state.get(_KP + "sig") != sig
    if st.button("Reset to view-implied", key=_KP + "reset"):
        reseed = True
    st.session_state[_KP + "sig"] = sig

    if mode == _CDF:
        quantiles = _QUANTILE_PRESETS[n]
        cdf = np.cumsum(bp)
        seed = [float(np.interp(q, cdf, bb)) for q in quantiles]
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
        masses = _seed_bucket_masses(bp, bb, boundaries)
        cols = st.columns(n)
        probs_in: list[float] = []
        for i in range(n):
            k = _KP + f"bucket_{i}"
            if reseed or k not in st.session_state:
                st.session_state[k] = round(float(masses[i]), 1) if i < len(masses) else 0.0
            probs_in.append(cols[i].number_input(f"{boundaries[i]:.2f}", min_value=0.0, step=1.0, key=k))
        tot = sum(probs_in)
        if tot <= 0:
            st.warning("Bucket probabilities must sum to more than 0.")
            return None, None
        probs_norm = np.array([p / tot for p in probs_in])
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
    mean_spot = float(np.dot(np.array(probs), np.array(bins)))
    lean = "bullish" if mean_spot > ms.fwd else "bearish"
    st.caption(f"Implied mean spot {mean_spot:.4f} vs forward {ms.fwd:.4f} ({lean} edge).")
    return probs, bins
