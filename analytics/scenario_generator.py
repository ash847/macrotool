"""
Deterministic scenario-grid generation from trade inputs.

Scenarios are emitted as explicit grid cells rather than broad families.
Each row corresponds to an elapsed-horizon checkpoint and each column to a
specific market state at that checkpoint.

SPOT-anchored grid: the no-move centre (`S`) is spot unchanged, and the σ-offset
columns are measured from SPOT (not the forward). `scenario_spot` is built first,
then `scenario_fwd = scenario_spot · e^{(r_d−r_f)·remaining}` is derived for MtM
pricing (rolls down to spot at expiry). The favourable/adverse *orientation*
(`direction`) stays FORWARD-relative (sign of K vs the forward), so the up-weighted
"favourable" cells align with where the forward-constructed structure profits.

The sigma used in sigma-anchored columns is always time-scaled:
    sigma_t = vol * sqrt(elapsed_time)
"""

from __future__ import annotations

import math

ONE_WEEK_YEARS = 7.0 / 365.0
SIX_WEEKS_YEARS = 42.0 / 365.0

GRID_ROWS = ["1w", "25%T", "50%T", "Expiry"]
GRID_COLS = ["S", "t%→K", "K", "K+½σ", "−½σ", "−1σ", "Δvol"]

# Human-readable labels for display. The −½σ/−1σ columns offset from SPOT
# (not the target K), so spell that out; other columns are shown as-is.
GRID_COL_LABELS = {"S": "No move", "−½σ": "spot −½σ", "−1σ": "spot −1σ"}


def col_label(col: str) -> str:
    """Display label for a scenario column. Falls back to the raw id."""
    return GRID_COL_LABELS.get(col, col)

ROW_TIME_FRACTIONS: dict[str, float | None] = {
    "1w": None,
    "25%T": 0.25,
    "50%T": 0.50,
    "Expiry": 1.0,
}

VALID_GRID_CELLS = {
    "1w": ["S", "Δvol"],
    "25%T": GRID_COLS[:],
    "50%T": GRID_COLS[:],
    "Expiry": ["S", "K", "K+½σ", "−1σ"],
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
        "fwd_rules": ["S", "t%→K", "K", "K+½σ", "−½σ", "−1σ", "Δvol"],
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
    spot: float = trade_inputs["spot"]
    T: float = trade_inputs["tenor_years"]
    base_vol: float = trade_inputs["implied_vol"]
    r_d: float = trade_inputs["r_d"]
    r_f: float = trade_inputs["r_f"]

    sigma_T_full = base_vol * math.sqrt(T)
    # Orientation stays FORWARD-relative (matches the forward-constructed structure).
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
                spot=spot,
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


def _apply_spot_rule(rule: str, spot: float, K: float, sigma_t: float, direction: int) -> float:
    """Terminal/interim SPOT level for a σ-anchored column. Offsets measured from
    spot; orientation (`direction`) is forward-relative."""
    if rule == "S":
        return spot
    if rule == "t%→K":
        raise ValueError("t%→K requires row-specific progress fraction")
    if rule == "K":
        return K
    if rule == "K+½σ":
        return K * math.exp(direction * 0.5 * sigma_t)
    if rule == "−½σ":
        return spot * math.exp(-direction * 0.5 * sigma_t)
    if rule == "−1σ":
        return spot * math.exp(-direction * 1.0 * sigma_t)
    raise ValueError(f"Unknown spot_rule: {rule!r}")


def _proportional_progress_spot(row: str, spot: float, K: float) -> float:
    """Spot level at proportional progress from spot toward the target K."""
    if row == "25%T":
        p = 0.25
    elif row == "50%T":
        p = 0.50
    else:
        raise ValueError(f"Row {row!r} does not support proportional-progress spot")
    return math.exp((1.0 - p) * math.log(spot) + p * math.log(K))


def _build_cell_scenario(
    *,
    row: str,
    col: str,
    spot: float,
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
        scenario_spot = _proportional_progress_spot(row, spot, K)
    elif col == "Δvol":
        # Hold spot at no-move at every checkpoint so the Δvol column isolates the
        # vol shock (a clean vega read). Anchoring interim rows to proportional
        # progress toward K would bundle a directional spot drift into the cell,
        # flipping its sign vs the 1w no-move cell for directional structures.
        scenario_spot = spot
    else:
        scenario_spot = _apply_spot_rule(col, spot, K, sigma_t, direction)

    # Derive the scenario forward for MtM pricing (rolls down to spot at expiry).
    scenario_fwd = scenario_spot * math.exp((r_d - r_f) * tau)
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
        vol_bump = base_vol * 0.04
        sc["vol_shifts"] = [-vol_bump, vol_bump]
    return sc
