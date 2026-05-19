"""
Visualisations comparing PM's elicited distribution to the market baseline,
rendered as Altair charts inside the Streamlit app.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from elicitation import Distribution


_COLOUR_MARKET = "#888888"
_COLOUR_USER = "#1f77b4"
_COLOUR_SCALE = alt.Scale(
    domain=["Market-implied", "Your view"],
    range=[_COLOUR_MARKET, _COLOUR_USER],
)


def _market_mass_per_range(baseline: Distribution, edges: np.ndarray) -> np.ndarray:
    """Probability mass of `baseline` between consecutive `edges`."""
    masses = np.array([
        float(baseline.probs[(baseline.bins >= edges[i]) & (baseline.bins < edges[i + 1])].sum())
        for i in range(len(edges) - 1)
    ])
    return masses


def render_option1_chart(prices: np.ndarray, quantiles: np.ndarray, baseline: Distribution) -> alt.Chart:
    """
    Histogram of mass per bin between consecutive user-input prices.

    Bin probabilities for the user are fixed quantile differences (e.g. for
    default 7 anchors: 8/15/25/25/15/8 %). Bin probabilities for the market
    are the baseline mass in each price range. As PM moves anchor prices, the
    bin widths shrink/grow and the market mass per bin changes; user mass per
    bin stays constant by construction.
    """
    n_bins = len(prices) - 1
    user_mass = np.diff(quantiles)
    market_mass = _market_mass_per_range(baseline, prices)
    labels = [f"{prices[i]:.3f} – {prices[i+1]:.3f}" for i in range(n_bins)]

    df = pd.DataFrame({
        "bin": labels * 2,
        "source": ["Market-implied"] * n_bins + ["Your view"] * n_bins,
        "mass": np.concatenate([market_mass, user_mass]),
        "bin_order": list(range(n_bins)) * 2,
    })

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("bin:N", title="Price range", sort=labels, axis=alt.Axis(labelAngle=-25)),
            xOffset=alt.XOffset("source:N", scale=alt.Scale(paddingOuter=0.1)),
            y=alt.Y("mass:Q", title="Probability mass", axis=alt.Axis(format=".0%")),
            color=alt.Color("source:N", scale=_COLOUR_SCALE, title=None),
            tooltip=["bin", "source", alt.Tooltip("mass:Q", format=".2%", title="Mass")],
        )
        .properties(height=240)
    )


def render_option2_chart(
    boundaries: np.ndarray,
    user_probs: np.ndarray,
    baseline: Distribution,
    sigma_offsets: np.ndarray | None = None,
) -> alt.Chart:
    """Grouped bar chart of probability per σ-bucket: market vs user."""
    n_buckets = len(user_probs)
    market_mass = _market_mass_per_range(baseline, boundaries)
    if market_mass.sum() > 0:
        market_mass = market_mass / market_mass.sum()

    if sigma_offsets is not None:
        labels = [
            f"{boundaries[i]:.3f}–{boundaries[i+1]:.3f}\n"
            f"({sigma_offsets[i]:+.2g}σ → {sigma_offsets[i+1]:+.2g}σ)"
            for i in range(n_buckets)
        ]
    else:
        labels = [f"{boundaries[i]:.3f}–{boundaries[i+1]:.3f}" for i in range(n_buckets)]

    df = pd.DataFrame({
        "bucket": labels * 2,
        "source": ["Market-implied"] * n_buckets + ["Your view"] * n_buckets,
        "prob": np.concatenate([market_mass, user_probs]),
    })

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("bucket:N", title="Spot range (σ)", sort=labels, axis=alt.Axis(labelAngle=-25)),
            xOffset=alt.XOffset("source:N", scale=alt.Scale(paddingOuter=0.1)),
            y=alt.Y("prob:Q", title="Probability mass", axis=alt.Axis(format=".0%")),
            color=alt.Color("source:N", scale=_COLOUR_SCALE, title=None),
            tooltip=["bucket", "source", alt.Tooltip("prob:Q", format=".2%", title="Mass")],
        )
        .properties(height=240)
    )


def render_option2_stacked_chart(
    user_probs: np.ndarray,
    market_probs: np.ndarray,
) -> alt.Chart:
    """
    Horizontal stacked bar: total length 100%, one row for market, one for user.
    Segment widths show probability mass per bucket; consistent palette across
    the two rows so PMs can see allocation differences at a glance.
    """
    n = len(user_probs)
    df = pd.DataFrame({
        "source": ["Market-implied"] * n + ["Your view"] * n,
        "bucket_idx": list(range(n)) * 2,
        "bucket_label": [f"B{i+1}" for i in range(n)] * 2,
        "prob": np.concatenate([market_probs, user_probs]),
    })

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("source:N", title=None, sort=["Market-implied", "Your view"]),
            x=alt.X("prob:Q", stack="zero", title="Probability allocation", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "bucket_idx:O",
                scale=alt.Scale(scheme="blueorange"),
                legend=None,
            ),
            order=alt.Order("bucket_idx:O"),
            tooltip=["source", "bucket_label", alt.Tooltip("prob:Q", format=".2%", title="Mass")],
        )
        .properties(height=110)
    )
