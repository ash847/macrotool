"""Inline Kelly edge elicitation for the Trade View page.

When the PM toggles Kelly sizing, this renders the same CDF / fixed-range-PDF
elicitation the Kelly Sizing screen offers — but inline on the Trade View — and
writes the resulting distribution to ``st.session_state.kelly_probs / kelly_bins``
(consumed by ``build_sizing_spec``). It reuses the pure elicitation functions; the
standalone Kelly screen is untouched. Inputs are seeded from the view-implied
distribution and re-seed when the market/view context changes (or via a reset).
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from analytics.sizing import view_implied_distribution
from interface.kelly_v2.elicitation import (
    default_sigma_boundaries,
    elicit_from_cdf_anchors,
    elicit_from_pdf_buckets,
    sigma_boundaries_to_prices,
)

_CDF = "Quantiles (CDF)"
_PDF = "Fixed-range buckets (PDF)"
_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_KP = "tvk_"   # session-key prefix, isolated from the Kelly screen's keys


def _seed_cdf_prices(bp: np.ndarray, bb: np.ndarray) -> list[float]:
    cdf = np.cumsum(bp)
    return [float(np.interp(q, cdf, bb)) for q in _QUANTILES]


def _seed_bucket_masses(bp: np.ndarray, bb: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    masses = np.array([
        float(bp[(bb >= boundaries[i]) & (bb < boundaries[i + 1])].sum())
        for i in range(len(boundaries) - 1)
    ])
    return (masses / masses.sum() * 100.0) if masses.sum() > 0 else masses


def render_kelly_elicitation(ms, target: float, conviction: str):
    """Render the elicitation block; return (probs, bins) tuples or (None, None)."""
    base_probs, base_bins = view_implied_distribution(ms.spot, ms.fwd, ms.vol, ms.T, target, conviction)
    bp, bb = np.array(base_probs), np.array(base_bins)
    n_bins = int(st.session_state.get("kelly_n_bins", 41))

    st.markdown("**Your edge — terminal-spot distribution**")
    mode = st.radio("Input style", [_CDF, _PDF], horizontal=True, key=_KP + "mode")

    # Re-seed the inputs when the market/view context changes, or on explicit reset.
    sig = (round(ms.fwd, 6), round(ms.vol, 6), round(target, 6), conviction, mode,
           int(st.session_state.get(_KP + "nbuckets", 6)))
    reseed = st.session_state.get(_KP + "sig") != sig
    if st.button("Reset to view-implied", key=_KP + "reset"):
        reseed = True
    st.session_state[_KP + "sig"] = sig

    if mode == _CDF:
        seed = _seed_cdf_prices(bp, bb)
        cols = st.columns(len(_QUANTILES))
        prices: list[float] = []
        for i, (q, c) in enumerate(zip(_QUANTILES, cols)):
            k = _KP + f"anchor_{i}"
            if reseed or k not in st.session_state:
                st.session_state[k] = round(seed[i], 4)
            prices.append(c.number_input(f"{int(q*100)}%", format="%.4f", key=k))
        try:
            dist = elicit_from_cdf_anchors(prices, list(_QUANTILES), n_bins=n_bins)
        except ValueError as e:
            st.warning(f"Quantile prices must strictly increase — {e}")
            return None, None
    else:
        n_buckets = int(st.number_input(
            "Buckets", min_value=3, max_value=12,
            value=int(st.session_state.get(_KP + "nbuckets", 6)), step=1, key=_KP + "nbuckets",
        ))
        offsets = default_sigma_boundaries(n_buckets)
        boundaries = sigma_boundaries_to_prices(offsets, forward=ms.fwd, sigma=ms.vol, tenor_years=ms.T)
        masses = _seed_bucket_masses(bp, bb, boundaries)
        cols = st.columns(min(n_buckets, 6))
        probs_in: list[float] = []
        for i in range(n_buckets):
            k = _KP + f"bucket_{i}"
            if reseed or k not in st.session_state:
                st.session_state[k] = round(float(masses[i]), 1) if i < len(masses) else 0.0
            c = cols[i % len(cols)]
            probs_in.append(c.number_input(
                f"{boundaries[i]:.3f}–{boundaries[i+1]:.3f}", min_value=0.0, step=1.0, key=k,
            ))
        tot = sum(probs_in)
        if tot <= 0:
            st.warning("Bucket probabilities must sum to more than 0.")
            return None, None
        try:
            dist = elicit_from_pdf_buckets(list(boundaries), [p / tot for p in probs_in], n_bins=n_bins)
        except ValueError as e:
            st.warning(str(e))
            return None, None

    probs = tuple(float(p) for p in dist.probs)
    bins = tuple(float(b) for b in dist.bins)
    st.session_state.kelly_probs = probs
    st.session_state.kelly_bins = bins
    mean_spot = float(np.dot(np.array(probs), np.array(bins)))
    lean = "bullish" if mean_spot > ms.fwd else "bearish"
    st.caption(f"Implied mean spot {mean_spot:.4f} vs forward {ms.fwd:.4f} ({lean} edge).")
    return probs, bins
