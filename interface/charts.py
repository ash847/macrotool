"""
Plotly chart builders for the MacroTool Streamlit UI.

Two main visuals:
  1. Scenario P&L heatmap — spot × vol grid for the top structure
  2. Sizing metrics — displayed via st.metric in app.py

charts_for_step() and metrics_for_step() are called by app.py
after each flow step to decide what to render.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from conversation.flow import ConversationFlow, Step
from data.schema import CurrencySnapshot
from knowledge_engine.models import SizingOutput, StructureSelectionResult, TradeView
from pricing.black_scholes import call_value, put_value
from pricing.forwards import rate_context_for_snapshot, tenor_to_years
from pricing.scenario import ScenarioConfig, build_scenario_matrix
from analytics.models import PriceDistribution


# ---------------------------------------------------------------------------
# Scenario heatmap
# ---------------------------------------------------------------------------

def build_scenario_heatmap(flow: ConversationFlow) -> go.Figure | None:
    """
    Returns a Plotly figure with one tab per time horizon.
    X = spot % change, Y = implied vol shift, Z = P&L in $.
    """
    view = flow.view
    ccy = flow.ccy
    selector = flow.selector_result

    if not (view and ccy and selector):
        return None

    top = selector.shortlist[0] if selector.shortlist else None
    if not top or top.structure_id in ("spot", "forward"):
        return None

    try:
        rate_ctx = rate_context_for_snapshot(ccy, tenor_to_years("3M"))
        atm_vol = ccy.get_atm_vol("3M")
        strike = rate_ctx.forward

        if top.structure_id in ("vanilla_call", "call_spread"):
            def pricer(spot, T_rem, sigma):
                return call_value(spot, strike, T_rem, sigma, rate_ctx.r_d, rate_ctx.r_f)
        else:
            def pricer(spot, T_rem, sigma):
                return put_value(spot, strike, T_rem, sigma, rate_ctx.r_d, rate_ctx.r_f)

        initial_value = pricer(rate_ctx.spot, rate_ctx.T, atm_vol)

        cfg = ScenarioConfig(
            spot_range_pct=8, spot_steps=9,
            vol_range_pct=30, vol_steps=7,
            time_horizons_years=[1 / 12, 2 / 12, 3 / 12],
            tenor_labels=["1M", "2M", "3M"],
        )
        matrix = build_scenario_matrix(
            pricer, rate_ctx.spot, atm_vol, rate_ctx.T, initial_value, cfg
        )

        notional = view.notional_usd or 1_000_000
        # matrix.pnl_matrix shape: [n_horizons, n_spot, n_vol]

        # Build spot/vol axis labels
        spot_changes = np.linspace(
            -cfg.spot_range_pct / 100,
            cfg.spot_range_pct / 100,
            cfg.spot_steps,
        )
        vol_changes = np.linspace(
            -cfg.vol_range_pct / 100,
            cfg.vol_range_pct / 100,
            cfg.vol_steps,
        )

        spot_labels = [f"{c:+.1%}" for c in spot_changes]
        vol_labels = [f"{c:+.1%}" for c in vol_changes]

        # Build figure with 3 subplots (one per horizon)
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f"At {lbl}" for lbl in cfg.tenor_labels],
            horizontal_spacing=0.06,
        )

        for col_idx, (h_idx, lbl) in enumerate(zip(range(3), cfg.tenor_labels), start=1):
            # pnl_matrix[h_idx] shape: [n_spot, n_vol]
            z = matrix.pnl_matrix[h_idx] * notional  # in $

            # Clip for color scale (extreme values distort colour)
            abs_max = max(abs(z.min()), abs(z.max()), 1)

            heat = go.Heatmap(
                z=z.tolist(),
                x=vol_labels,
                y=spot_labels,
                colorscale=[
                    [0.0, "#d73027"],
                    [0.45, "#fee08b"],
                    [0.5, "#ffffff"],
                    [0.55, "#d9ef8b"],
                    [1.0, "#1a9850"],
                ],
                zmid=0,
                zmin=-abs_max,
                zmax=abs_max,
                showscale=(col_idx == 3),
                colorbar=dict(title="P&L ($)", thickness=12) if col_idx == 3 else None,
                hovertemplate=(
                    "Spot: %{y}<br>"
                    "Vol: %{x}<br>"
                    f"Horizon: {lbl}<br>"
                    "P&L: $%{z:,.0f}<extra></extra>"
                ),
            )
            fig.add_trace(heat, row=1, col=col_idx)

        spot_str = f"{rate_ctx.spot:.4f}"
        fig.update_layout(
            title=dict(
                text=(
                    f"Scenario P&L — {top.display_name} | {view.pair} "
                    f"spot {spot_str} | ${notional:,.0f} notional"
                ),
                font=dict(size=14),
            ),
            height=380,
            margin=dict(t=80, b=40, l=60, r=40),
            font=dict(size=11),
        )

        for col_idx in range(1, 4):
            fig.update_xaxes(title_text="Vol shift", row=1, col=col_idx)
            fig.update_yaxes(title_text="Spot change" if col_idx == 1 else "", row=1, col=col_idx)

        return fig

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Maturity histogram
# ---------------------------------------------------------------------------

def build_maturity_histogram(
    flat: PriceDistribution | None,
    smile: PriceDistribution | None,
    target_price: float | None = None,
) -> go.Figure | None:
    """
    Grouped bar chart: probability of spot landing in each 5%-of-spot bin at maturity.
    Flat-vol bars (blue) and smile-adjusted bars (orange) shown side by side.
    """
    import math
    from scipy.stats import norm

    if not flat:
        return None

    S0 = flat.spot
    # σ√T recovered from stored terminal bands
    sigma_sqrtT = math.log(flat.terminal_plus1s / flat.terminal_median)
    # (μ − σ²/2)T = log(median / S0)
    drift_term = math.log(flat.terminal_median / S0)

    # ── Bin edges at 5% of spot increments ───────────────────────────────
    bin_width = S0 * 0.05
    lo = math.floor(flat.terminal_median * math.exp(-3.5 * sigma_sqrtT) / bin_width) * bin_width
    hi = math.ceil(flat.terminal_median * math.exp(3.5 * sigma_sqrtT) / bin_width) * bin_width
    edges: list[float] = []
    k = lo
    while k <= hi + bin_width * 0.01:
        edges.append(round(k, 6))
        k += bin_width

    n_bins = len(edges) - 1
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

    # ── Flat-vol probabilities (exact lognormal) ─────────────────────────
    def flat_cdf(K: float) -> float:
        z = (math.log(K / S0) - drift_term) / sigma_sqrtT
        return norm.cdf(z)

    flat_probs = [
        (flat_cdf(edges[i + 1]) - flat_cdf(edges[i])) * 100
        for i in range(n_bins)
    ]

    # ── Smile probabilities (piecewise linear CDF in log-price space) ────
    smile_probs: list[float] = []
    if smile:
        cdf_pts = sorted([
            (smile.terminal_minus3s, 0.0013),
            (smile.terminal_minus2s, 0.0228),
            (smile.terminal_minus1s, 0.1587),
            (smile.terminal_median,  0.5000),
            (smile.terminal_plus1s,  0.8413),
            (smile.terminal_plus2s,  0.9772),
            (smile.terminal_plus3s,  0.9987),
        ], key=lambda x: x[0])

        def smile_cdf(K: float) -> float:
            if K <= cdf_pts[0][0]:
                return 0.0
            if K >= cdf_pts[-1][0]:
                return 1.0
            for i in range(len(cdf_pts) - 1):
                k0, p0 = cdf_pts[i]
                k1, p1 = cdf_pts[i + 1]
                if k0 <= K <= k1:
                    w = math.log(K / k0) / math.log(k1 / k0)
                    return p0 + w * (p1 - p0)
            return 1.0

        smile_probs = [
            (smile_cdf(edges[i + 1]) - smile_cdf(edges[i])) * 100
            for i in range(n_bins)
        ]

    # ── Bar colours — highlight bin containing target ─────────────────────
    target_bin = None
    if target_price is not None:
        for i in range(n_bins):
            if edges[i] <= target_price < edges[i + 1]:
                target_bin = i
                break

    flat_colors = [
        "rgba(251,191,36,0.85)" if i == target_bin else "rgba(96,165,250,0.75)"
        for i in range(n_bins)
    ]
    smile_colors = [
        "rgba(251,191,36,0.85)" if i == target_bin else "rgba(249,115,22,0.70)"
        for i in range(n_bins)
    ]

    x_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(n_bins)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_labels,
        y=flat_probs,
        name="Flat vol",
        marker_color=flat_colors,
        hovertemplate="%{x}<br>Flat: %{y:.1f}%<extra></extra>",
    ))

    if smile and smile_probs:
        fig.add_trace(go.Bar(
            x=x_labels,
            y=smile_probs,
            name="Smile",
            marker_color=smile_colors,
            hovertemplate="%{x}<br>Smile: %{y:.1f}%<extra></extra>",
        ))

    # Target vertical line (add_shape works on categorical axes; add_vline does not)
    if target_price is not None and target_bin is not None:
        target_label = f"{target_price:.2f}"
        line_color = "#16a34a" if target_price > S0 else "#dc2626"
        fig.add_shape(
            type="line",
            x0=x_labels[target_bin], x1=x_labels[target_bin],
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=line_color, width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=x_labels[target_bin], y=1,
            xref="x", yref="paper",
            text=f"Target {target_label}",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color=line_color),
        )

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"Probability at Maturity ({flat.horizon_days}d)  |  5% bins",
            font=dict(size=13),
        ),
        xaxis=dict(
            title=f"{flat.pair} spot",
            tickangle=-90,
        ),
        yaxis=dict(title="Probability (%)"),
        height=360,
        margin=dict(t=70, b=70, l=60, r=20),
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ---------------------------------------------------------------------------
# Sizing metrics
# ---------------------------------------------------------------------------

def build_sizing_metrics(flow: ConversationFlow) -> dict | None:
    """
    Returns a dict of key sizing values for display as st.metric cards.
    """
    sizing = flow.sizing
    ccy = flow.ccy
    if not (sizing and ccy):
        return None

    return {
        "Kelly fraction": f"{sizing.kelly_fraction:.2f} ({sizing.kelly_conviction_used} conviction)",
        "Adjusted Kelly": f"{sizing.adjusted_kelly:.3f}",
        "Notional": (
            f"${sizing.kelly_notional_usd:,.0f}"
            if sizing.kelly_notional_usd
            else "—"
        ),
        "Stop level": (
            f"{sizing.stop_level:.4f}  ({sizing.stop_distance_pct:.1f}% from {ccy.spot:.4f})"
            if sizing.stop_level
            else "—"
        ),
        "Tranches": (
            f"{sizing.tranche_count} × [{', '.join(f'{w*100:.0f}%' for w in sizing.tranche_schedule)}]"
            if sizing.tranche_schedule
            else "—"
        ),
    }


# ---------------------------------------------------------------------------
# Distribution fan chart
# ---------------------------------------------------------------------------

def build_distribution_fan(
    flat: PriceDistribution | None,
    smile: PriceDistribution | None,
    target_price: float | None = None,
) -> go.Figure | None:
    """
    Overlaid price distribution cone.

    Blue filled bands  = flat ATM vol (±1σ / ±2σ / ±3σ).
    Orange dashed lines = smile-adjusted bands at the same levels.
    X-axis = days, Y-axis = price (range ±5σ at maturity).
    """
    if not (flat and smile):
        return None

    x = flat.time_steps_days
    flat_prices = {b.label: b.prices for b in flat.bands}
    smile_prices = {b.label: b.prices for b in smile.bands}

    fig = go.Figure()

    # ── Flat vol: concentric filled bands (blue, outermost first) ────────
    # Each successive fill sits on top of the previous, so inner = darker.
    fill_specs = [
        ("-3σ", "+3σ", "rgba(191,219,254,0.40)", "rgba(191,219,254,0.15)", "±3σ flat"),
        ("-2σ", "+2σ", "rgba(147,197,253,0.50)", "rgba(147,197,253,0.20)", "±2σ flat"),
        ("-1σ", "+1σ", "rgba( 96,165,250,0.55)", "rgba( 96,165,250,0.28)", "±1σ flat"),
    ]
    for lower_lbl, upper_lbl, fill_color, line_color, legend_name in fill_specs:
        fig.add_trace(go.Scatter(
            x=x, y=flat_prices[lower_lbl],
            mode="lines",
            line=dict(width=0.5, color=line_color),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=flat_prices[upper_lbl],
            mode="lines",
            fill="tonexty",
            fillcolor=fill_color,
            line=dict(width=0.5, color=line_color),
            name=legend_name,
            hoverinfo="skip",
        ))

    # Forward / flat median
    fig.add_trace(go.Scatter(
        x=x, y=flat_prices["Median"],
        mode="lines",
        line=dict(color="#1e3a5f", width=1.5),
        name="Forward",
        hovertemplate="Day %{x}: %{y:.4f}<extra>Forward</extra>",
    ))

    # ── Smile: dashed lines (orange) ─────────────────────────────────────
    smile_specs = [
        ("-3σ", "+3σ", "rgba(249,115,22,0.50)", "dash",   "±3σ smile"),
        ("-2σ", "+2σ", "rgba(249,115,22,0.65)", "dash",   "±2σ smile"),
        ("-1σ", "+1σ", "rgba(249,115,22,0.85)", "dot",    "±1σ smile"),
    ]
    for lower_lbl, upper_lbl, color, dash, legend_name in smile_specs:
        for band_lbl, show_legend in [(lower_lbl, True), (upper_lbl, False)]:
            fig.add_trace(go.Scatter(
                x=x, y=smile_prices[band_lbl],
                mode="lines",
                line=dict(color=color, width=1.3, dash=dash),
                name=legend_name,
                showlegend=show_legend,
                legendgroup=legend_name,
                hovertemplate=f"{band_lbl} smile: %{{y:.4f}}<extra></extra>",
            ))

    # ── Layout ────────────────────────────────────────────────────────────
    y_lo = min(flat.axis_min, smile.axis_min) * 0.995
    y_hi = max(flat.axis_max, smile.axis_max) * 1.005

    tick_spacing = max(7, round(flat.horizon_days / 8 / 7) * 7)

    fig.update_layout(
        title=dict(
            text=(
                f"Price Distribution Cone — {flat.pair}  |  "
                f"{flat.horizon_days}d horizon  |  "
                f"ATM vol {flat.atm_vol * 100:.1f}%"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="Days", tickmode="linear", dtick=tick_spacing),
        yaxis=dict(title=f"{flat.pair} price", range=[y_lo, y_hi]),
        height=360,
        margin=dict(t=70, b=50, l=70, r=20),
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig.add_hline(
        y=flat.spot,
        line=dict(color="grey", width=1, dash="dot"),
        annotation_text="Spot",
        annotation_position="left",
        annotation_font_size=10,
    )

    # ── Target price marker ───────────────────────────────────────────────
    if target_price is not None:
        import math
        # Compute σ level: log(target/median) / log(+1σ/median) = z/1
        try:
            sigma_sqrtT = math.log(flat.terminal_plus1s / flat.terminal_median)
            z = math.log(target_price / flat.terminal_median) / sigma_sqrtT
            z_label = f"{z:+.1f}σ"
        except (ValueError, ZeroDivisionError):
            z_label = ""

        marker_color = "#16a34a" if target_price > flat.spot else "#dc2626"

        fig.add_trace(go.Scatter(
            x=[flat.horizon_days],
            y=[target_price],
            mode="markers+text",
            marker=dict(symbol="star", size=14, color=marker_color,
                        line=dict(width=1, color="white")),
            text=[f"  {target_price:.4f}  {z_label}"],
            textposition="middle right",
            textfont=dict(size=11, color=marker_color),
            name="Target",
            hovertemplate=f"Target: {target_price:.4f}  ({z_label})<extra></extra>",
        ))

    return fig


# ---------------------------------------------------------------------------
# Step-triggered chart/metric selection
# ---------------------------------------------------------------------------

def charts_for_step(step_completed: Step, flow: ConversationFlow) -> list[go.Figure]:
    """
    Returns list of Plotly figures to render after a step completes.
    Called by app.py with the step that just finished.
    """
    if step_completed in (Step.INTAKE, Step.STRUCTURE_REC):
        charts = []
        fan = build_distribution_fan(flow.flat_distribution, flow.smile_distribution)
        if fan:
            charts.append(fan)
        heatmap = build_scenario_heatmap(flow)
        if heatmap:
            charts.append(heatmap)
        return charts
    return []


def metrics_for_step(step_completed: Step, flow: ConversationFlow) -> dict | None:
    """
    Returns sizing metrics dict to render after a step completes.
    """
    if step_completed == Step.SIZING:
        return build_sizing_metrics(flow)
    return None
