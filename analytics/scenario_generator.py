"""
Deterministic scenario generation from trade inputs.

Operates entirely in forward space: all fwd_rules compute a scenario_fwd,
and scenario_spot is back-derived via original rates.

Invariants across scenarios:
  - Vol surface: only absolute vol_shift applied on top of base_vol
  - Discount factors / rates: r_d, r_f unchanged
  - Strike, structure, expiry: fixed at trade entry (not this module's concern)
"""

from __future__ import annotations

import math
from copy import deepcopy

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

FAMILIES = [
    "CORRECT_PATH", "EARLY_TARGET", "NO_MOVE", "WRONG_WAY",
    "OVERSHOOT", "VOL_SENSITIVITY", "SKEW_SENSITIVITY", "EXPIRY",
]

TIME_FRACTIONS = [0.25, 0.50, 0.75, 1.00]

FWD_RULES = [
    "FORWARD", "TARGET",
    "PARTIAL_TO_TARGET_25", "PARTIAL_TO_TARGET_50", "PARTIAL_TO_TARGET_75",
    "ADVERSE_0_5SIGMA", "ADVERSE_1SIGMA",
    "OVERSHOOT_0_5SIGMA", "OVERSHOOT_1SIGMA",
    "NEUTRAL_FORWARD",
    "NEUTRAL_DOWN_0_5SIGMA", "NEUTRAL_DOWN_1SIGMA",
    "NEUTRAL_UP_0_5SIGMA", "NEUTRAL_UP_1SIGMA",
]

VOL_RULES = ["VOL_FLAT", "VOL_DOWN", "VOL_UP"]

SKEW_RULES = ["SKEW_UNCHANGED", "SKEW_FLATTER", "SKEW_STEEPER"]

_VOL_SHIFTS: dict[str, float] = {
    "VOL_FLAT": 0.0,
    "VOL_DOWN": -0.01,
    "VOL_UP": +0.01,
}

_SKEW_MULTIPLIERS: dict[str, float] = {
    "SKEW_UNCHANGED": 1.0,
    "SKEW_FLATTER": 0.75,
    "SKEW_STEEPER": 1.25,
}

# ---------------------------------------------------------------------------
# Standard scenario packs (template — no derived values)
# ---------------------------------------------------------------------------

_DIRECTIONAL_PACK: list[dict] = [
    # --- Correct Path ---
    {"id": "CORRECT_PATH_25",  "family": "CORRECT_PATH",    "time_fraction": 0.25, "fwd_rule": "PARTIAL_TO_TARGET_25",  "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["path_pnl", "correct_direction"]},
    {"id": "CORRECT_PATH_50",  "family": "CORRECT_PATH",    "time_fraction": 0.50, "fwd_rule": "PARTIAL_TO_TARGET_50",  "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["path_pnl", "correct_direction"]},
    {"id": "CORRECT_PATH_75",  "family": "CORRECT_PATH",    "time_fraction": 0.75, "fwd_rule": "PARTIAL_TO_TARGET_75",  "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["path_pnl", "correct_direction"]},
    {"id": "CORRECT_PATH_100", "family": "CORRECT_PATH",    "time_fraction": 1.00, "fwd_rule": "TARGET",                "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["path_pnl", "terminal"]},
    # --- Early Target ---
    {"id": "EARLY_TARGET_25",  "family": "EARLY_TARGET",    "time_fraction": 0.25, "fwd_rule": "TARGET",                "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["early_monetisation"]},
    {"id": "EARLY_TARGET_50",  "family": "EARLY_TARGET",    "time_fraction": 0.50, "fwd_rule": "TARGET",                "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["early_monetisation"]},
    {"id": "EARLY_TARGET_75",  "family": "EARLY_TARGET",    "time_fraction": 0.75, "fwd_rule": "TARGET",                "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["early_monetisation"]},
    # --- No Move ---
    {"id": "NO_MOVE_25",       "family": "NO_MOVE",         "time_fraction": 0.25, "fwd_rule": "FORWARD",               "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "no_move"]},
    {"id": "NO_MOVE_50",       "family": "NO_MOVE",         "time_fraction": 0.50, "fwd_rule": "FORWARD",               "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "no_move"]},
    {"id": "NO_MOVE_75",       "family": "NO_MOVE",         "time_fraction": 0.75, "fwd_rule": "FORWARD",               "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "no_move"]},
    # --- Wrong Way ---
    {"id": "WRONG_SMALL_50",   "family": "WRONG_WAY",       "time_fraction": 0.50, "fwd_rule": "ADVERSE_0_5SIGMA",      "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["robustness", "wrong_way"]},
    {"id": "WRONG_LARGE_50",   "family": "WRONG_WAY",       "time_fraction": 0.50, "fwd_rule": "ADVERSE_1SIGMA",        "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["robustness", "wrong_way"]},
    # --- Overshoot ---
    {"id": "OVERSHOOT_50",     "family": "OVERSHOOT",       "time_fraction": 0.50, "fwd_rule": "OVERSHOOT_0_5SIGMA",    "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["robustness", "overshoot"]},
    {"id": "OVERSHOOT_100",    "family": "EXPIRY",          "time_fraction": 1.00, "fwd_rule": "OVERSHOOT_0_5SIGMA",    "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["terminal", "overshoot"]},
    # --- Vol Sensitivity ---
    {"id": "VOL_DOWN_PARTIAL_50", "family": "VOL_SENSITIVITY", "time_fraction": 0.50, "fwd_rule": "PARTIAL_TO_TARGET_50", "vol_rule": "VOL_DOWN", "skew_rule": "SKEW_UNCHANGED", "tags": ["vol_sensitivity"]},
    {"id": "VOL_UP_PARTIAL_50",   "family": "VOL_SENSITIVITY", "time_fraction": 0.50, "fwd_rule": "PARTIAL_TO_TARGET_50", "vol_rule": "VOL_UP",   "skew_rule": "SKEW_UNCHANGED", "tags": ["vol_sensitivity"]},
    {"id": "VOL_DOWN_TARGET_50",  "family": "VOL_SENSITIVITY", "time_fraction": 0.50, "fwd_rule": "TARGET",               "vol_rule": "VOL_DOWN", "skew_rule": "SKEW_UNCHANGED", "tags": ["vol_sensitivity"]},
    {"id": "VOL_UP_TARGET_50",    "family": "VOL_SENSITIVITY", "time_fraction": 0.50, "fwd_rule": "TARGET",               "vol_rule": "VOL_UP",   "skew_rule": "SKEW_UNCHANGED", "tags": ["vol_sensitivity"]},
    # --- Skew Sensitivity ---
    {"id": "SKEW_FLATTER_TARGET_50",  "family": "SKEW_SENSITIVITY", "time_fraction": 0.50, "fwd_rule": "TARGET", "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_FLATTER",  "tags": ["skew_sensitivity"]},
    {"id": "SKEW_STEEPER_TARGET_50",  "family": "SKEW_SENSITIVITY", "time_fraction": 0.50, "fwd_rule": "TARGET", "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_STEEPER",  "tags": ["skew_sensitivity"]},
    # --- Expiry terminals ---
    {"id": "EXPIRY_FORWARD",       "family": "EXPIRY", "time_fraction": 1.00, "fwd_rule": "FORWARD",        "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["terminal", "no_move"]},
    {"id": "EXPIRY_ADVERSE_1SIGMA","family": "EXPIRY", "time_fraction": 1.00, "fwd_rule": "ADVERSE_1SIGMA", "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["terminal", "wrong_way"]},
]

_NEUTRAL_PACK: list[dict] = [
    {"id": "NEUTRAL_DOWN_1SIGMA_50",    "family": "WRONG_WAY",       "time_fraction": 0.50, "fwd_rule": "NEUTRAL_DOWN_1SIGMA",    "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "neutral"]},
    {"id": "NEUTRAL_DOWN_0_5SIGMA_50",  "family": "WRONG_WAY",       "time_fraction": 0.50, "fwd_rule": "NEUTRAL_DOWN_0_5SIGMA",  "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "neutral"]},
    {"id": "NEUTRAL_FORWARD_25",        "family": "NO_MOVE",         "time_fraction": 0.25, "fwd_rule": "NEUTRAL_FORWARD",        "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "neutral"]},
    {"id": "NEUTRAL_FORWARD_50",        "family": "NO_MOVE",         "time_fraction": 0.50, "fwd_rule": "NEUTRAL_FORWARD",        "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "neutral"]},
    {"id": "NEUTRAL_FORWARD_75",        "family": "NO_MOVE",         "time_fraction": 0.75, "fwd_rule": "NEUTRAL_FORWARD",        "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "neutral"]},
    {"id": "NEUTRAL_UP_0_5SIGMA_50",    "family": "CORRECT_PATH",    "time_fraction": 0.50, "fwd_rule": "NEUTRAL_UP_0_5SIGMA",    "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "neutral"]},
    {"id": "NEUTRAL_UP_1SIGMA_50",      "family": "CORRECT_PATH",    "time_fraction": 0.50, "fwd_rule": "NEUTRAL_UP_1SIGMA",      "vol_rule": "VOL_FLAT", "skew_rule": "SKEW_UNCHANGED", "tags": ["theta_bleed", "neutral"]},
    {"id": "VOL_DOWN_FORWARD_50",       "family": "VOL_SENSITIVITY", "time_fraction": 0.50, "fwd_rule": "NEUTRAL_FORWARD",        "vol_rule": "VOL_DOWN", "skew_rule": "SKEW_UNCHANGED", "tags": ["vol_sensitivity", "neutral"]},
    {"id": "VOL_UP_FORWARD_50",         "family": "VOL_SENSITIVITY", "time_fraction": 0.50, "fwd_rule": "NEUTRAL_FORWARD",        "vol_rule": "VOL_UP",   "skew_rule": "SKEW_UNCHANGED", "tags": ["vol_sensitivity", "neutral"]},
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_enumerations() -> dict:
    """Return all valid enum values."""
    return {
        "families": FAMILIES,
        "time_fractions": TIME_FRACTIONS,
        "fwd_rules": FWD_RULES,
        "vol_rules": VOL_RULES,
        "skew_rules": SKEW_RULES,
        "vol_shifts": _VOL_SHIFTS,
        "skew_multipliers": _SKEW_MULTIPLIERS,
    }


def generate_scenarios(trade_inputs: dict) -> list[dict]:
    """
    Generate the standard scenario pack from trade inputs.

    Required keys: forward, target, tenor_years, implied_vol, r_d, r_f, spot.
    Returns list of scenario dicts each with template fields + derived block.
    """
    F: float = trade_inputs["forward"]
    K: float = trade_inputs["target"]
    T: float = trade_inputs["tenor_years"]
    base_vol: float = trade_inputs["implied_vol"]
    r_d: float = trade_inputs["r_d"]
    r_f: float = trade_inputs["r_f"]

    sigma_T = base_vol * math.sqrt(T)
    direction = _compute_direction(F, K, sigma_T)

    templates = _NEUTRAL_PACK if direction == 0 else _DIRECTIONAL_PACK

    scenarios = []
    for tmpl in templates:
        new_fwd = _apply_fwd_rule(tmpl["fwd_rule"], F, K, sigma_T, direction)
        tau = T * (1.0 - tmpl["time_fraction"])
        elapsed = T * tmpl["time_fraction"]
        scenario_spot = new_fwd * math.exp(-(r_d - r_f) * tau)
        vol_shift = _VOL_SHIFTS[tmpl["vol_rule"]]
        scenario_vol = max(base_vol + vol_shift, 0.01)
        skew_multiplier = _SKEW_MULTIPLIERS[tmpl["skew_rule"]]

        sc = deepcopy(tmpl)
        sc["derived"] = {
            "elapsed_time": round(elapsed, 8),
            "remaining_time": round(tau, 8),
            "scenario_fwd": round(new_fwd, 8),
            "scenario_spot": round(scenario_spot, 8),
            "vol_shift": vol_shift,
            "scenario_vol": round(scenario_vol, 8),
            "skew_multiplier": skew_multiplier,
            "sigma_T": round(sigma_T, 8),
            "direction": direction,
        }
        scenarios.append(sc)

    return scenarios


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_direction(F: float, K: float, sigma_T: float) -> int:
    log_ratio = math.log(K / F)
    if sigma_T > 0 and abs(log_ratio) < 0.05 * sigma_T:
        return 0
    return 1 if log_ratio > 0 else -1


def _apply_fwd_rule(rule: str, F: float, K: float, sigma_T: float, direction: int) -> float:
    if rule == "FORWARD":
        return F
    if rule == "TARGET":
        return K
    if rule == "PARTIAL_TO_TARGET_25":
        return math.exp(0.75 * math.log(F) + 0.25 * math.log(K))
    if rule == "PARTIAL_TO_TARGET_50":
        return math.exp(0.50 * math.log(F) + 0.50 * math.log(K))
    if rule == "PARTIAL_TO_TARGET_75":
        return math.exp(0.25 * math.log(F) + 0.75 * math.log(K))
    if rule == "ADVERSE_0_5SIGMA":
        return F * math.exp(-direction * 0.5 * sigma_T)
    if rule == "ADVERSE_1SIGMA":
        return F * math.exp(-direction * 1.0 * sigma_T)
    if rule == "OVERSHOOT_0_5SIGMA":
        return K * math.exp(direction * 0.5 * sigma_T)
    if rule == "OVERSHOOT_1SIGMA":
        return K * math.exp(direction * 1.0 * sigma_T)
    if rule in ("NEUTRAL_FORWARD",):
        return F
    if rule == "NEUTRAL_DOWN_0_5SIGMA":
        return F * math.exp(-0.5 * sigma_T)
    if rule == "NEUTRAL_DOWN_1SIGMA":
        return F * math.exp(-1.0 * sigma_T)
    if rule == "NEUTRAL_UP_0_5SIGMA":
        return F * math.exp(0.5 * sigma_T)
    if rule == "NEUTRAL_UP_1SIGMA":
        return F * math.exp(1.0 * sigma_T)
    raise ValueError(f"Unknown fwd_rule: {rule!r}")
