"""
Scenario-grid context selector and multiplier loader.

Pure function: MarketState + PM preferences -> active context, plus normalized
weights over explicit scenario-grid cells.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from analytics.market_state import MarketState
from analytics.scenario_generator import (
    GRID_COLS,
    GRID_ROWS,
    cell_id,
    valid_grid_cells_for_tenor,
)

_REPO_ROOT = Path(__file__).parent.parent
_WEIGHTS_PATH = _REPO_ROOT / "knowledge" / "defaults" / "scenario_definitions.json"


@dataclass(frozen=True)
class FiredWeighting:
    id: str
    multipliers: dict[str, float]
    comment: str


FiredContext = FiredWeighting


@dataclass(frozen=True)
class WeighterResult:
    weights: dict[str, float]
    multipliers: dict[str, float]
    fired: list[FiredWeighting]
    base_fired: FiredWeighting | None
    overlay_fired: list[FiredWeighting]
    baseline: float
    min_multiplier: float


_weights_cache: dict | None = None
_weights_source = "local"


def _empty_grid_multipliers() -> dict[str, float]:
    return {cell_id(r, c): 1.0 for r in GRID_ROWS for c in GRID_COLS}


def _migrate_config_shape(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg.setdefault("baseline", 1.0)
    cfg.setdefault("min_multiplier", 0.1)
    raw_weightings = cfg.pop("weightings", None)
    if raw_weightings is not None:
        base_weightings = []
        preference_overlays = []
        for ctx in raw_weightings:
            migrated = dict(ctx)
            migrated.setdefault("multipliers", {})
            migrated.pop("adjustments", None)
            ctx_id = str(migrated.get("id", ""))
            if ctx_id == "cheap_carry":
                preference_overlays.append({
                    "id": "objective_keep_cost_low",
                    "comment": migrated.get("comment", ""),
                    "when": [{"field": "primary_objective", "op": "==", "value": "Keep cost low"}],
                    "multipliers": migrated.get("multipliers", {}),
                })
                continue
            if ctx_id == "conservative_carry":
                preference_overlays.append({
                    "id": "objective_keep_risk_clean",
                    "comment": migrated.get("comment", ""),
                    "when": [{"field": "primary_objective", "op": "==", "value": "Keep risk clean"}],
                    "multipliers": migrated.get("multipliers", {}),
                })
                continue
            if ctx_id == "delta_carry":
                preference_overlays.append({
                    "id": "management_monetise_early",
                    "comment": migrated.get("comment", ""),
                    "when": [{"field": "trade_management", "op": "==", "value": "May monetise early"}],
                    "multipliers": migrated.get("multipliers", {}),
                })
                continue
            migrated["when"] = [
                cond for cond in migrated.get("when", [])
                if cond.get("field") not in {"primary_objective", "trade_management"}
            ]
            base_weightings.append(migrated)
        cfg["base_weightings"] = base_weightings
        cfg["preference_overlays"] = preference_overlays
    cfg.setdefault("base_weightings", [])
    cfg.setdefault("preference_overlays", [])
    for bucket in ("base_weightings", "preference_overlays"):
        for ctx in cfg[bucket]:
            if "multipliers" not in ctx:
                ctx["multipliers"] = {}
            ctx.pop("adjustments", None)
    return cfg


def load_scenario_weights_config() -> dict:
    global _weights_cache, _weights_source
    if _weights_cache is not None:
        return _weights_cache
    try:
        from interface.supabase_logger import fetch_config_for_engine_with_meta
        data, source = fetch_config_for_engine_with_meta("scenario_definitions")
        if data:
            _weights_cache = _migrate_config_shape(data)
            _weights_source = source
            return _weights_cache
        if source == "error":
            with open(_WEIGHTS_PATH) as f:
                _weights_source = "local-fallback-error"
                return _migrate_config_shape(json.load(f))
    except Exception:
        pass
    with open(_WEIGHTS_PATH) as f:
        _weights_source = "local"
        return _migrate_config_shape(json.load(f))


def clear_scenario_weights_cache() -> None:
    global _weights_cache, _weights_source
    _weights_cache = None
    _weights_source = "local"


def get_scenario_weights_source() -> str:
    return _weights_source


_FIELD_GETTERS = {
    "target_z_abs":      lambda ms, prefs: abs(ms.target_z) if ms.target_z is not None else None,
    "carry_regime":      lambda ms, prefs: ms.carry_regime,
    "with_carry":        lambda ms, prefs: ms.with_carry,
    "T":                 lambda ms, prefs: ms.T,
    "vol":               lambda ms, prefs: ms.vol,
    "atmfsratio":        lambda ms, prefs: ms.atmfsratio,
    "primary_objective": lambda ms, prefs: prefs.get("primary_objective"),
    "trade_management":  lambda ms, prefs: prefs.get("trade_management"),
}

_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "in": lambda a, b: a in b,
}


def _evaluate_condition(cond: dict, ms: MarketState, prefs: dict) -> bool:
    field = cond["field"]
    op = cond["op"]
    value = cond["value"]
    if field not in _FIELD_GETTERS or op not in _OPS:
        raise ValueError(f"Unsupported scenario weighting condition: {cond}")
    actual = _FIELD_GETTERS[field](ms, prefs)
    if actual is None:
        return False
    return _OPS[op](actual, value)


def _all_conditions_met(when: list[dict], ms: MarketState, prefs: dict) -> bool:
    return all(_evaluate_condition(c, ms, prefs) for c in when)


def compute_family_weights(
    ms: MarketState,
    primary_objective: str = "Balanced",
    trade_management: str = "Standard hold",
) -> WeighterResult:
    """
    Backwards-compatible entry point. Returns normalized scenario-cell weights
    plus the raw multipliers used to derive them.
    """
    cfg = load_scenario_weights_config()
    baseline = float(cfg["baseline"])
    min_multiplier = float(cfg["min_multiplier"])

    valid_cells = [cell_id(r, c) for r, c in valid_grid_cells_for_tenor(ms.T)]
    multipliers = {cid: baseline for cid in valid_cells}

    prefs = {
        "primary_objective": primary_objective,
        "trade_management": trade_management,
    }

    base_fired: FiredContext | None = None
    for ctx in cfg["base_weightings"]:
        if not _all_conditions_met(ctx.get("when", []), ms, prefs):
            continue
        overrides = ctx.get("multipliers", {})
        for cid, value in overrides.items():
            if cid in multipliers:
                multipliers[cid] = max(float(value), min_multiplier)
        base_fired = FiredWeighting(
            id=ctx["id"],
            multipliers={cid: multipliers[cid] for cid in valid_cells},
            comment=ctx.get("comment", ""),
        )
        break

    overlay_fired: list[FiredWeighting] = []
    for ctx in cfg["preference_overlays"]:
        if not _all_conditions_met(ctx.get("when", []), ms, prefs):
            continue
        overrides = ctx.get("multipliers", {})
        for cid, value in overrides.items():
            if cid in multipliers:
                multipliers[cid] = max(multipliers[cid] * float(value), min_multiplier)
        overlay_fired.append(FiredWeighting(
            id=ctx["id"],
            multipliers={cid: multipliers[cid] for cid in valid_cells},
            comment=ctx.get("comment", ""),
        ))

    fired: list[FiredContext] = ([base_fired] if base_fired else []) + overlay_fired

    total = sum(multipliers.values())
    if total <= 0:
        weights = {cid: 1.0 / len(multipliers) for cid in multipliers}
    else:
        weights = {cid: m / total for cid, m in multipliers.items()}

    return WeighterResult(
        weights=weights,
        multipliers=multipliers,
        fired=fired,
        base_fired=base_fired,
        overlay_fired=overlay_fired,
        baseline=baseline,
        min_multiplier=min_multiplier,
    )
