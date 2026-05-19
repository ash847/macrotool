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


def _quantile_of_distribution(dist: Distribution, q: float) -> float:
    """Inverse CDF on a discrete distribution by linear interpolation."""
    cum = np.cumsum(dist.probs)
    return float(np.interp(q, cum, dist.bins))


def render_option1_chart(prices: np.ndarray, quantiles: np.ndarray, baseline: Distribution) -> alt.Chart:
    """
    Strip plot: each fixed quantile is a marker on a single price axis.

    Two rows — market-implied price for that quantile (top) and your view's
    input (bottom). Moving an anchor moves only the x-coordinate; the y-row
    and quantile label stay fixed. This matches what PM is actually doing in
    Option 1: positioning fixed quantile levels along the price axis.
    """
    market_prices = np.array([_quantile_of_distribution(baseline, q) for q in quantiles])
    quantile_labels = [f"{int(round(q * 100))}%" for q in quantiles]

    df = pd.DataFrame({
        "price": np.concatenate([market_prices, prices]),
        "quantile": quantile_labels * 2,
        "source": ["Market-implied"] * len(quantiles) + ["Your view"] * len(prices),
    })

    # Shared x-axis range covers both market and user marker positions.
    x_lo = float(min(market_prices.min(), prices.min()))
    x_hi = float(max(market_prices.max(), prices.max()))
    pad = 0.04 * (x_hi - x_lo) if x_hi > x_lo else 0.04
    x_domain = [x_lo - pad, x_hi + pad]

    base = alt.Chart(df).encode(
        x=alt.X("price:Q", title="Price", scale=alt.Scale(domain=x_domain, nice=False)),
        y=alt.Y("source:N", title=None, sort=["Market-implied", "Your view"]),
        color=alt.Color("source:N", scale=_COLOUR_SCALE, legend=None),
    )

    points = base.mark_point(size=180, filled=True, opacity=0.9).encode(
        tooltip=[
            "source",
            "quantile",
            alt.Tooltip("price:Q", format=".4f"),
        ],
    )

    labels = base.mark_text(dy=-12, fontSize=10).encode(text="quantile:N")

    return (points + labels).properties(height=140)


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
