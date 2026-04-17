"""
Scenario matrix builder.

Computes the mark-to-market P&L of a position across a grid of:
  - Spot perturbations  (e.g. ±10% in 7 steps)
  - Vol perturbations   (e.g. -25%, flat, +25%)
  - Time horizons       (e.g. 1M, 2M, 3M into a 3M trade)

The result is a 3D array: [horizon_idx, spot_idx, vol_idx].

P&L is expressed in premium currency units (USD for NDF pairs, EUR for EURPLN).
Multiply by the notional scale factor to get dollar P&L.

The pricer_fn signature is:
    pricer_fn(spot, T_remaining, sigma) -> float
    (Returns mark-to-market value per unit of notional at the given state)

The scenario matrix does NOT model path dependency for barrier options — each
node is an independent Black-Scholes valuation. This is an approximation (it
ignores the knock-out probability for RKO options at intermediate dates).
For a POC comparison tool this is clearly flagged and is sufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


PricerFn = Callable[[float, float, float], float]
# signature: pricer_fn(spot, T_remaining, sigma) -> value


@dataclass
class ScenarioConfig:
    """
    Configuration for the scenario matrix.
    These are the display defaults from ResolvedConfig.display.scenario_matrix.
    """
    spot_range_pct: float = 10.0     # ± this percentage around current spot
    spot_steps: int = 7              # must be odd (centre = 0 shift)
    vol_range_pct: float = 25.0      # ± this percentage of current vol
    vol_steps: int = 3               # typically 3: low, base, high
    time_horizons_years: list[float] = field(
        default_factory=lambda: [1/12, 2/12, 3/12]
    )
    tenor_labels: list[str] = field(
        default_factory=lambda: ["1M", "2M", "3M"]
    )


@dataclass
class ScenarioResult:
    """
    Output of build_scenario_matrix.

    pnl_matrix shape: [n_horizons, n_spot_steps, n_vol_steps]
    pnl_matrix[i, j, k] = P&L at horizon i, spot_shifts[j], vol_shifts[k]
                           relative to initial_value (i.e., after subtracting premium paid)
    """
    spot_shifts: list[float]           # e.g. [-10, -5, 0, 5, 10] (%)
    vol_shifts: list[float]            # e.g. [-25, 0, 25] (%)
    horizons: list[str]                # e.g. ["1M", "2M", "3M"]
    base_spot: float
    base_vol: float
    initial_value: float               # premium paid / received at inception
    pnl_matrix: np.ndarray             # shape: [n_horizons, n_spot_steps, n_vol_steps]


def build_scenario_matrix(
    pricer_fn: PricerFn,
    base_spot: float,
    base_vol: float,
    total_T: float,
    initial_value: float,
    config: ScenarioConfig | None = None,
) -> ScenarioResult:
    """
    Build the scenario P&L matrix.

    Args:
        pricer_fn:      Function(spot, T_remaining, sigma) -> current value.
        base_spot:      Current spot rate.
        base_vol:       Current ATM vol (decimal).
        total_T:        Total time to expiry of the trade (years).
        initial_value:  Premium paid at inception (positive = cost).
        config:         Scenario grid configuration.

    Returns:
        ScenarioResult with P&L relative to initial_value at each grid point.
    """
    if config is None:
        config = ScenarioConfig()

    # Build spot perturbation grid (centred on 0)
    spot_pcts = np.linspace(
        -config.spot_range_pct,
        config.spot_range_pct,
        config.spot_steps,
    )
    # Build vol perturbation grid
    vol_pcts = np.linspace(
        -config.vol_range_pct,
        config.vol_range_pct,
        config.vol_steps,
    )

    n_h = len(config.time_horizons_years)
    n_s = len(spot_pcts)
    n_v = len(vol_pcts)

    pnl = np.zeros((n_h, n_s, n_v))

    for h_idx, elapsed_T in enumerate(config.time_horizons_years):
        T_remaining = max(total_T - elapsed_T, 0.0)
        for s_idx, spot_pct in enumerate(spot_pcts):
            perturbed_spot = base_spot * (1 + spot_pct / 100)
            for v_idx, vol_pct in enumerate(vol_pcts):
                perturbed_vol = base_vol * (1 + vol_pct / 100)
                current_value = pricer_fn(perturbed_spot, T_remaining, perturbed_vol)
                pnl[h_idx, s_idx, v_idx] = current_value - initial_value

    return ScenarioResult(
        spot_shifts=spot_pcts.tolist(),
        vol_shifts=vol_pcts.tolist(),
        horizons=config.tenor_labels[:n_h],
        base_spot=base_spot,
        base_vol=base_vol,
        initial_value=initial_value,
        pnl_matrix=pnl,
    )


def format_scenario_table(
    result: ScenarioResult,
    notional_scale: float = 1.0,
    title: str = "Scenario P&L",
    decimal_places: int = 0,
) -> str:
    """
    Format the scenario matrix as a plain-text table for injection into the LLM context.

    Shows one sub-table per time horizon, with spot shifts as rows and vol shifts as columns.
    P&L values are scaled by notional_scale and formatted as integers (or to decimal_places).

    Returns a string suitable for embedding in the LLM system prompt.
    """
    lines = [f"=== {title} ===", ""]

    fmt = f"{{:+.{decimal_places}f}}" if decimal_places > 0 else "{:+,.0f}"

    for h_idx, horizon in enumerate(result.horizons):
        lines.append(f"[ {horizon} ]")

        # Header: vol shift labels
        vol_headers = ["Spot \\ Vol"] + [f"Vol {v:+.0f}%" for v in result.vol_shifts]
        col_widths = [max(12, len(h)) for h in vol_headers]
        header_row = "  ".join(h.rjust(w) for h, w in zip(vol_headers, col_widths))
        lines.append(header_row)
        lines.append("-" * len(header_row))

        for s_idx, spot_pct in enumerate(result.spot_shifts):
            spot_label = f"Spot {spot_pct:+.1f}%"
            row_vals = [spot_label.rjust(col_widths[0])]
            for v_idx in range(len(result.vol_shifts)):
                raw_pnl = result.pnl_matrix[h_idx, s_idx, v_idx] * notional_scale
                cell = fmt.format(raw_pnl)
                row_vals.append(cell.rjust(col_widths[v_idx + 1]))
            lines.append("  ".join(row_vals))

        lines.append("")

    return "\n".join(lines)
