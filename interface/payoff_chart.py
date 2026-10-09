"""Plotly payoff-at-expiry chart for one priced structure.

Pure presentation over ``analytics.payoff_curve.PayoffCurve`` — it never computes a
number the curve didn't already reconcile. Returns ``None`` (draw nothing) when the
curve failed any hard check: a missing chart is acceptable, a wrong one is not.

Interaction (all client-side, no Python round-trip): drag-box zoom on both axes,
scroll-wheel zoom, pan, double-click to reset. The *default* window is ±2σ√T around
spot widened to include every strike / barrier / target, with the y-range fitted to that
window — so an open tail running off to the ±3σ data edge doesn't flatten the part of
the chart the PM actually cares about. The full ±3σ data is still there to zoom out to.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import plotly.graph_objects as go

from analytics.payoff_curve import PayoffCurve

Units = Literal["pct", "notional"]

_GREEN = "rgba(38, 166, 91, 0.30)"
_RED = "rgba(214, 69, 65, 0.30)"
_LINE = "#4C78A8"
_TARGET = "#E4A11B"
_SPOT = "#7F7F7F"
_FWD = "#9B7FB8"
_STRIKE = "rgba(127,127,127,0.55)"
_BARRIER = "#D64541"

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "doubleClick": "reset",
}


def _units_for(curve: PayoffCurve, units: Units) -> Units:
    return "notional" if (units == "notional" and curve.notional is not None) else "pct"


def default_window(curve: PayoffCurve, sigma_sqrt_t: float | None = None) -> tuple[float, float]:
    """Default x-window: ±2σ√T around spot ∪ every key level, with a little padding."""
    levels = [curve.spot, curve.fwd, *curve.strikes]
    if curve.barrier is not None:
        levels.append(curve.barrier)
    if curve.target is not None:
        levels.append(curve.target)
    if sigma_sqrt_t is None:
        # Recover σ√T from the grid: it was built log-symmetric ±3σ about spot, before any
        # widening for key levels — good enough for a display window.
        sigma_sqrt_t = max(math.log(curve.spots[-1] / curve.spots[0]) / 6.0, 1e-4)
    lo = curve.spot * math.exp(-2.0 * sigma_sqrt_t)
    hi = curve.spot * math.exp(2.0 * sigma_sqrt_t)
    lo, hi = min(lo, min(levels)), max(hi, max(levels))
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def payoff_figure(
    curve: PayoffCurve | None,
    units: Units = "pct",
    sigma_sqrt_t: float | None = None,
) -> go.Figure | None:
    """Expiry P&L chart, or None if the curve is missing or failed a hard check."""
    if curve is None or not curve.ok:
        return None

    units = _units_for(curve, units)
    x = curve.spots
    y = curve.pnl(units)
    ccy = curve.base_ccy
    y_label = f"P&L, % of {ccy} notional" if units == "pct" else f"P&L, {ccy} (sized notional {curve.notional:,.0f})"
    y_fmt = "%{y:.2f}%" if units == "pct" else "%{y:,.2f}"

    fig = go.Figure()

    # Shaded profit / loss regions (clipped at zero; the exact zero-crossings are grid points).
    fig.add_trace(go.Scatter(
        x=x, y=np.maximum(y, 0.0), mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor=_GREEN, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=np.minimum(y, 0.0), mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor=_RED, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name="P&L at expiry", line=dict(color=_LINE, width=2.5),
        hovertemplate=f"spot %{{x:.4f}}<br>P&L {y_fmt}<extra></extra>",
    ))

    # Breakeven markers on the zero line.
    if curve.breakevens:
        fig.add_trace(go.Scatter(
            x=list(curve.breakevens), y=[0.0] * len(curve.breakevens), mode="markers",
            name="Breakeven", marker=dict(symbol="diamond", size=10, color=_LINE,
                                          line=dict(color="white", width=1.5)),
            hovertemplate="breakeven %{x:.4f}<extra></extra>",
        ))

    # Reference levels as dashed verticals with staggered labels.
    levels: list[tuple[float, str, str, str]] = [
        (curve.spot, "Spot", _SPOT, "dot"),
        (curve.fwd, "Fwd", _FWD, "dot"),
    ]
    if curve.target is not None:
        levels.append((curve.target, "Target", _TARGET, "dash"))
    for i, k in enumerate(curve.strikes, start=1):
        levels.append((k, f"K{i}" if len(curve.strikes) > 1 else "K", _STRIKE, "dash"))
    if curve.barrier is not None:
        levels.append((curve.barrier, "Barrier", _BARRIER, "dash"))
    levels.sort(key=lambda t: t[0])
    for j, (lvl, name, colour, dash) in enumerate(levels):
        fig.add_vline(x=lvl, line=dict(color=colour, width=1.3, dash=dash), layer="below")
        fig.add_annotation(
            x=lvl, y=1.0 - 0.055 * (j % 4), yref="paper", xref="x",
            text=f"{name} {lvl:.4g}" if lvl >= 100 else f"{name} {lvl:.4f}",
            showarrow=False, font=dict(size=11, color=colour), xanchor="left", xshift=3,
            bgcolor="rgba(255,255,255,0.0)",
        )
    fig.add_hline(y=0.0, line=dict(color="rgba(127,127,127,0.7)", width=1))

    # Default window + a y-range fitted to it.
    x0, x1 = default_window(curve, sigma_sqrt_t)
    mask = (x >= x0) & (x <= x1)
    ys = y[mask] if mask.any() else y
    ylo, yhi = float(min(ys.min(), 0.0)), float(max(ys.max(), 0.0))
    ypad = 0.10 * ((yhi - ylo) or 1.0)

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="zoom",
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(title=f"{curve.pair} spot at expiry", range=[x0, x1], showgrid=True,
                   gridcolor="rgba(127,127,127,0.18)", zeroline=False),
        yaxis=dict(title=y_label, range=[ylo - ypad, yhi + ypad], showgrid=True,
                   gridcolor="rgba(127,127,127,0.18)", zeroline=False),
        # Zoom survives reruns that don't change the structure/units; a new key resets it.
        uirevision=f"{curve.pair}|{curve.structure_id}|{curve.label}|{units}",
    )
    return fig
