"""
Deterministic scenario-grid generation from trade inputs.

Scenarios are emitted as explicit grid cells rather than broad families.
Each row corresponds to an elapsed-horizon checkpoint and each column to a
specific market state at that checkpoint.

The sigma used in sigma-anchored columns is always time-scaled:
    sigma_t = vol * sqrt(elapsed_time)
"""

from __future__ import annotations

import math

ONE_WEEK_YEARS = 7.0 / 365.0
SIX_WEEKS_YEARS = 42.0 / 365.0

GRID_ROWS = ["1w", "25%T", "50%T", "Expiry"]
GRID_COLS = ["F", "t%→K", "K", "K+½σ", "−½σ", "−1σ", "Δvol"]

ROW_TIME_FRACTIONS: dict[str, float | None] = {
    "1w": None,
    "25%T": 0.25,
    "50%T": 0.50,
    "Expiry": 1.0,
}

VALID_GRID_CELLS = {
    "1w": ["F", "Δvol"],
    "25%T": GRID_COLS[:],
    "50%T": GRID_COLS[:],
    "Expiry": ["F", "K", "K+½σ", "−1σ"],
}


def get_enumerations() -> dict:
    """Return valid grid rows/columns and legacy aliases used by tests/UI."""
    return {
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "valid_grid_cells": VALID_GRID_CELLS,
        # Legacy aliases retained for callers/tests that only need discovery.
        "families": GRID_ROWS,
        "time_fractions": [0.25, 0.50, 0.75, 1.00],
        "fwd_rules": ["F", "t%→K", "K", "K+½σ", "−½σ", "−1σ", "Δvol"],
        "vol_rules": ["VOL_FLAT", "VOL_AVG"],
        "skew_rules": ["SKEW_UNCHANGED"],
    }


def valid_grid_rows(T: float) -> list[str]:
    rows = ["25%T", "50%T", "Expiry"]
    if T > SIX_WEEKS_YEARS:
        return ["1w"] + rows
    return rows


def valid_grid_cells_for_tenor(T: float) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for row in valid_grid_rows(T):
        for col in VALID_GRID_CELLS[row]:
            cells.append((row, col))
    return cells


def cell_id(row: str, col: str) -> str:
    return f"{row}|{col}"


def generate_scenarios(trade_inputs: dict) -> list[dict]:
    """
    Generate the explicit scenario grid from trade inputs.

    Required keys: forward, target, tenor_years, implied_vol, r_d, r_f, spot.
    """
    F: float = trade_inputs["forward"]
    K: float = trade_inputs["target"]
    T: float = trade_inputs["tenor_years"]
    base_vol: float = trade_inputs["implied_vol"]
    r_d: float = trade_inputs["r_d"]
    r_f: float = trade_inputs["r_f"]

    sigma_T_full = base_vol * math.sqrt(T)
    direction = _compute_direction(F, K, sigma_T_full)

    scenarios: list[dict] = []
    for row in valid_grid_rows(T):
        elapsed = ONE_WEEK_YEARS if row == "1w" else T * float(ROW_TIME_FRACTIONS[row])
        tau = max(T - elapsed, 0.0)
        sigma_t = base_vol * math.sqrt(elapsed) if elapsed > 0 else 0.0

        for col in VALID_GRID_CELLS[row]:
            sc = _build_cell_scenario(
                row=row,
                col=col,
                F=F,
                K=K,
                T=T,
                elapsed=elapsed,
                tau=tau,
                sigma_t=sigma_t,
                base_vol=base_vol,
                r_d=r_d,
                r_f=r_f,
                direction=direction,
            )
            scenarios.append(sc)

    return scenarios


def _compute_direction(F: float, K: float, sigma_T: float) -> int:
    log_ratio = math.log(K / F)
    if sigma_T > 0 and abs(log_ratio) < 0.05 * sigma_T:
        return 0
    return 1 if log_ratio > 0 else -1


def _apply_fwd_rule(rule: str, F: float, K: float, sigma_t: float, direction: int) -> float:
    if rule == "F":
        return F
    if rule == "t%→K":
        raise ValueError("t%→K requires row-specific progress fraction")
    if rule == "K":
        return K
    if rule == "K+½σ":
        return K * math.exp(direction * 0.5 * sigma_t)
    if rule == "−½σ":
        return F * math.exp(-direction * 0.5 * sigma_t)
    if rule == "−1σ":
        return F * math.exp(-direction * 1.0 * sigma_t)
    raise ValueError(f"Unknown fwd_rule: {rule!r}")


def _proportional_progress_fwd(row: str, F: float, K: float) -> float:
    if row == "25%T":
        p = 0.25
    elif row == "50%T":
        p = 0.50
    else:
        raise ValueError(f"Row {row!r} does not support proportional-progress forward")
    return math.exp((1.0 - p) * math.log(F) + p * math.log(K))


def _build_cell_scenario(
    *,
    row: str,
    col: str,
    F: float,
    K: float,
    T: float,
    elapsed: float,
    tau: float,
    sigma_t: float,
    base_vol: float,
    r_d: float,
    r_f: float,
    direction: int,
) -> dict:
    if col == "t%→K":
        scenario_fwd = _proportional_progress_fwd(row, F, K)
    elif col == "Δvol":
        scenario_fwd = F if row == "1w" else _proportional_progress_fwd(row, F, K)
    else:
        scenario_fwd = _apply_fwd_rule(col, F, K, sigma_t, direction)

    scenario_spot = scenario_fwd * math.exp(-(r_d - r_f) * tau)
    sc = {
        "id": cell_id(row, col),
        "row": row,
        "col": col,
        "time_fraction": (elapsed / T) if T > 0 else 0.0,
        "fwd_rule": col,
        "vol_rule": "VOL_AVG" if col == "Δvol" else "VOL_FLAT",
        "skew_rule": "SKEW_UNCHANGED",
        "tags": [row, col],
        "derived": {
            "elapsed_time": round(elapsed, 8),
            "remaining_time": round(tau, 8),
            "scenario_fwd": round(scenario_fwd, 8),
            "scenario_spot": round(scenario_spot, 8),
            "vol_shift": None if col == "Δvol" else 0.0,
            "scenario_vol": round(base_vol, 8),
            "sigma_T": round(sigma_t, 8),
            "direction": direction,
            "skew_multiplier": 1.0,
        },
    }
    if col == "Δvol":
        sc["vol_shifts"] = [-0.01, 0.01]
    return sc
