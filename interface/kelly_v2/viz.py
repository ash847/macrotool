"""
Visualisations comparing PM's elicited distribution to the market baseline,
rendered as Altair charts inside the Streamlit app.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from .elicitation import Distribution


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
    and quantile label stay fixed.

    Coloured horizontal segments between adjacent markers demarcate the
    inter-quantile probability bands (e.g. 2%→10%, 10%→25%, …) using the same
    blueorange palette as Option 2's stacked allocation bar.
    """
    market_prices = np.array([_quantile_of_distribution(baseline, q) for q in quantiles])
    quantile_labels = [f"{int(round(q * 100))}%" for q in quantiles]

    points_df = pd.DataFrame({
        "price": np.concatenate([market_prices, prices]),
        "quantile": quantile_labels * 2,
        "source": ["Market-implied"] * len(quantiles) + ["Your view"] * len(prices),
    })

    # Inter-quantile bands: one segment per (i, i+1) per source row.
    segments_records = []
    for src_name, p_array in (("Market-implied", market_prices), ("Your view", prices)):
        for i in range(len(p_array) - 1):
            segments_records.append({
                "source": src_name,
                "x_start": float(p_array[i]),
                "x_end": float(p_array[i + 1]),
                "band_idx": i,
                "band_label": f"{quantile_labels[i]}–{quantile_labels[i + 1]}",
            })
    segments_df = pd.DataFrame(segments_records)

    x_lo = float(min(market_prices.min(), prices.min()))
    x_hi = float(max(market_prices.max(), prices.max()))
    pad = 0.04 * (x_hi - x_lo) if x_hi > x_lo else 0.04
    x_domain = [x_lo - pad, x_hi + pad]

    segments = (
        alt.Chart(segments_df)
        .mark_rule(strokeWidth=10, opacity=0.55, strokeCap="butt")
        .encode(
            x=alt.X("x_start:Q", title="Price", scale=alt.Scale(domain=x_domain, nice=False)),
            x2="x_end:Q",
            y=alt.Y("source:N", title=None, sort=["Market-implied", "Your view"]),
            color=alt.Color(
                "band_idx:O",
                scale=alt.Scale(scheme="blueorange"),
                legend=None,
            ),
            tooltip=["source", "band_label", alt.Tooltip("x_start:Q", format=".4f", title="from"), alt.Tooltip("x_end:Q", format=".4f", title="to")],
        )
    )

    base_points = alt.Chart(points_df).encode(
        x=alt.X("price:Q", scale=alt.Scale(domain=x_domain, nice=False)),
        y=alt.Y("source:N", sort=["Market-implied", "Your view"]),
    )
    points = base_points.mark_point(
        size=140, filled=True, opacity=1.0, color="#222222", stroke="#222222",
    ).encode(
        tooltip=["source", "quantile", alt.Tooltip("price:Q", format=".4f")],
    )
    labels = base_points.mark_text(dy=-12, fontSize=10, color="#222222").encode(text="quantile:N")

    return (segments + points + labels).properties(height=120)


def render_option2_chart(
    boundaries: np.ndarray,
    user_probs: np.ndarray,
    baseline: Distribution,
) -> alt.Chart:
    """Grouped bar chart of probability per spot-range bucket: market vs user."""
    n_buckets = len(user_probs)
    market_mass = _market_mass_per_range(baseline, boundaries)
    if market_mass.sum() > 0:
        market_mass = market_mass / market_mass.sum()

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
            x=alt.X("bucket:N", title="Spot range", sort=labels, axis=alt.Axis(labelAngle=-25, orient="bottom")),
            xOffset=alt.XOffset("source:N", scale=alt.Scale(paddingOuter=0.1)),
            y=alt.Y("prob:Q", title="Probability mass", axis=alt.Axis(format=".0%")),
            color=alt.Color("source:N", scale=_COLOUR_SCALE, title=None),
            tooltip=["bucket", "source", alt.Tooltip("prob:Q", format=".2%", title="Mass")],
        )
        .properties(height=510)
    )


def render_kelly_growth_curve(
    f_values: np.ndarray,
    geo_growth: np.ndarray,
    er_curve: np.ndarray,
    f_star: float,
    f_displayed: float,
) -> alt.Chart:
    """Kelly growth curve with expected-return overlay.

    Blue/red line — geometric return: exp(E[log(1+f·r)]) − 1. Blue above zero,
      red below. Peaks at f_star.
    Grey dashed  — expected return: f · E[r]. Linear, ignores variance drag.

    Note: no alt.datum() or transform_filter — uses explicit DataFrames with
    consistent field names so the chart renders reliably across Altair versions.
    """
    _COLOUR_ER = _COLOUR_MARKET
    _COLOUR_NEG = "red"

    f_max = float(f_values[-1])

    # Clip geo_growth at -0.1% so the red tail floors at the bottom of the
    # chart and the y-axis auto-scales to show the positive E[r] region.
    geo_growth = np.maximum(geo_growth, -0.001)

    # X-axis: whole-number % ticks, ~4–6 across the range.
    if f_max <= 0.04:
        tick_step = 0.01
    elif f_max <= 0.12:
        tick_step = 0.02
    elif f_max <= 0.30:
        tick_step = 0.05
    elif f_max <= 0.60:
        tick_step = 0.10
    else:
        tick_step = 0.20
    tick_values = [round(v, 4) for v in np.arange(0.0, f_max + tick_step * 0.5, tick_step)]
    x_ax = alt.Axis(format=".0%", values=tick_values)
    y_ax = alt.Axis(format=".2%")

    # ── Main curves ───────────────────────────────────────────────────────
    # All layers share "f" (x) and "value" (y) field names so Vega-Lite
    # resolves them onto the same scale without needing alt.datum().

    # Geometric return split at zero using pandas pre-filter (no transform_filter)
    geo_full = pd.DataFrame({"f": f_values, "value": geo_growth})
    geo_pos_df = geo_full[geo_full["value"] >= 0].copy()
    geo_neg_df = geo_full[geo_full["value"] < 0].copy()

    geo_pos = (
        alt.Chart(geo_pos_df).mark_line(color=_COLOUR_USER, strokeWidth=2)
        .encode(
            x=alt.X("f:Q", title="Fraction of bankroll (f)", axis=x_ax),
            y=alt.Y("value:Q", title="Return per trade", axis=y_ax),
            tooltip=[alt.Tooltip("f:Q", format=".1%", title="f"),
                     alt.Tooltip("value:Q", format=".2%", title="Geometric return")],
        )
    )
    geo_neg = (
        alt.Chart(geo_neg_df).mark_line(color=_COLOUR_NEG, strokeWidth=2)
        .encode(
            x=alt.X("f:Q", axis=x_ax),
            y=alt.Y("value:Q", axis=y_ax),
            tooltip=[alt.Tooltip("f:Q", format=".1%", title="f"),
                     alt.Tooltip("value:Q", format=".2%", title="Geometric return")],
        )
    )

    # Expected return overlay — same field names, different data
    er_df = pd.DataFrame({"f": f_values, "value": er_curve})
    er_line = (
        alt.Chart(er_df).mark_line(color=_COLOUR_ER, strokeWidth=1.5, strokeDash=[4, 3])
        .encode(
            x=alt.X("f:Q", axis=x_ax),
            y=alt.Y("value:Q", axis=y_ax),
            tooltip=[alt.Tooltip("f:Q", format=".1%", title="f"),
                     alt.Tooltip("value:Q", format=".2%", title="E[r] × f")],
        )
    )

    # Zero reference: explicit 2-point line at value=0 (no alt.datum)
    zero_df = pd.DataFrame({"f": [0.0, f_max], "value": [0.0, 0.0]})
    zero_line = (
        alt.Chart(zero_df).mark_line(color="#cccccc", strokeDash=[3, 3], strokeWidth=1)
        .encode(x="f:Q", y="value:Q")
    )

    # ── Inline labels at right edge ───────────────────────────────────────
    er_end_df = pd.DataFrame({"f": [f_max], "value": [float(er_curve[-1])],
                               "label": ["E[r] × f"]})
    er_end_label = (
        alt.Chart(er_end_df).mark_text(align="right", dx=-4, dy=-8, fontSize=9, color=_COLOUR_ER)
        .encode(x="f:Q", y="value:Q", text="label:N")
    )
    y_at_star = float(np.interp(f_star, f_values, geo_growth))
    geo_peak_df = pd.DataFrame({"f": [f_star], "value": [y_at_star],
                                 "label": ["Geometric return"]})
    geo_peak_label = (
        alt.Chart(geo_peak_df).mark_text(align="left", dx=4, dy=-8, fontSize=9, color=_COLOUR_USER)
        .encode(x="f:Q", y="value:Q", text="label:N")
    )

    # ── Point layers: marker + crosshairs to axes ─────────────────────────
    # Crosshairs use explicit 2-point line segments (no alt.datum).
    # Axis labels use DataFrames with value=0 (x-axis) or f=0 (y-axis).
    def _point_layers(x_p: float, y_p: float, colour: str, shape: str) -> list:
        marker = (
            alt.Chart(pd.DataFrame({"f": [x_p], "value": [y_p]}))
            .mark_point(color=colour, size=90, filled=True, shape=shape)
            .encode(x="f:Q", y="value:Q")
        )
        # Vertical dotted: (x_p, 0) → (x_p, y_p)
        v_line = (
            alt.Chart(pd.DataFrame({"f": [x_p, x_p], "value": [0.0, y_p]}))
            .mark_line(color=colour, strokeDash=[4, 4], strokeWidth=1, opacity=0.7)
            .encode(x="f:Q", y="value:Q")
        )
        # Horizontal dotted: (0, y_p) → (x_p, y_p)
        h_line = (
            alt.Chart(pd.DataFrame({"f": [0.0, x_p], "value": [y_p, y_p]}))
            .mark_line(color=colour, strokeDash=[4, 4], strokeWidth=1, opacity=0.7)
            .encode(x="f:Q", y="value:Q")
        )
        # X-axis label: centred at x_p, placed at value=0
        x_lbl = (
            alt.Chart(pd.DataFrame({"f": [x_p], "value": [0.0]}))
            .mark_text(align="center", dy=13, fontSize=9, color=colour, fontWeight="bold")
            .encode(x="f:Q", y="value:Q", text=alt.value(f"{x_p:.1%}"))
        )
        # Y-axis label: right-aligned at f=0, placed at value=y_p
        y_lbl = (
            alt.Chart(pd.DataFrame({"f": [0.0], "value": [y_p]}))
            .mark_text(align="right", dx=-6, fontSize=9, color=colour, fontWeight="bold")
            .encode(x="f:Q", y="value:Q", text=alt.value(f"{y_p:.1%}"))
        )
        return [v_line, h_line, marker, x_lbl, y_lbl]

    star_layers = _point_layers(f_star, y_at_star, "#F58518", "triangle-up")
    y_at_displayed = float(np.interp(f_displayed, f_values, geo_growth))
    disp_layers = (
        _point_layers(f_displayed, y_at_displayed, _COLOUR_USER, "circle")
        if abs(f_displayed - f_star) > 1e-4 else []
    )

    return (
        alt.layer(
            zero_line, geo_pos, geo_neg, er_line,
            *star_layers, *disp_layers,
            er_end_label, geo_peak_label,
        )
        .resolve_scale(y="shared", x="shared")
        .properties(height=405)
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
            x=alt.X("prob:Q", stack="zero", title=None, axis=None),
            color=alt.Color(
                "bucket_idx:O",
                scale=alt.Scale(scheme="blueorange"),
                legend=None,
            ),
            order=alt.Order("bucket_idx:O"),
            tooltip=["source", "bucket_label", alt.Tooltip("prob:Q", format=".2%", title="Mass")],
        )
        .properties(height=50)
    )
