"""
Tier 1 scenario family weighter.

Pure function: MarketState (already view-conditioned by the pipeline) →
{family: weight} summing to 1.0, plus a record of which rules fired so the
UI can explain the weights to the PM.

The weights are NOT structure-dependent: the same vector is applied to every
shortlisted structure when scoring, so weighted-P&L scores are directly
comparable across structures.

Rules live in `knowledge/defaults/scenario_weights.json`. Tunable without
code changes — see that file for the rule schema and prose explaining each
rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from analytics.market_state import MarketState
from analytics.scenario_generator import FAMILIES

_REPO_ROOT = Path(__file__).parent.parent
_WEIGHTS_PATH = _REPO_ROOT / "knowledge" / "defaults" / "scenario_weights.json"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FiredRule:
    """A rule that matched and was applied. Surfaced in the UI for transparency."""
    id: str
    family: str
    adjustment: float
    comment: str


@dataclass(frozen=True)
class WeighterResult:
    """Output of `compute_family_weights`. `weights` always sums to ~1.0 with
    every family present (floor enforced)."""
    weights: dict[str, float]
    fired: list[FiredRule]
    baseline: float
    floor: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_scenario_weights_config() -> dict:
    """Load and cache the rules JSON. Process restart needed to pick up edits."""
    with open(_WEIGHTS_PATH) as f:
        data = json.load(f)
    return data


def clear_scenario_weights_cache() -> None:
    """Invalidate the lru_cache (used in tests)."""
    load_scenario_weights_config.cache_clear()


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

# Map config field names → MarketState attribute getters. Keeps the JSON
# decoupled from any private MarketState shape and makes it explicit which
# fields the rule engine knows about.
_FIELD_GETTERS = {
    "target_z_abs": lambda ms: abs(ms.target_z) if ms.target_z is not None else None,
    "carry_regime": lambda ms: ms.carry_regime,
    "with_carry":   lambda ms: ms.with_carry,
    "T":            lambda ms: ms.T,
    "vol":          lambda ms: ms.vol,
    "atmfsratio":   lambda ms: ms.atmfsratio,
}

_OPS = {
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _evaluate_condition(cond: dict, ms: MarketState) -> bool:
    field = cond["field"]
    op = cond["op"]
    value = cond["value"]

    if field not in _FIELD_GETTERS:
        raise ValueError(f"Unknown field '{field}' in scenario_weights rule. "
                         f"Supported: {sorted(_FIELD_GETTERS)}")
    if op not in _OPS:
        raise ValueError(f"Unknown op '{op}' in scenario_weights rule. "
                         f"Supported: {sorted(_OPS)}")

    actual = _FIELD_GETTERS[field](ms)
    if actual is None:
        # Fields that are None on this MarketState (e.g. target_z when no
        # target supplied) cause the condition to fail rather than raise.
        return False
    return _OPS[op](actual, value)


def _all_conditions_met(when: list[dict], ms: MarketState) -> bool:
    return all(_evaluate_condition(c, ms) for c in when)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_family_weights(ms: MarketState) -> WeighterResult:
    """
    Compute scenario family weights from MarketState.

    Algorithm:
      1. Initialise every family at `baseline` (default 1/8 = 0.125).
      2. For each rule whose `when` conditions all evaluate true, add the
         rule's `adjustment` to that family's weight; record the rule.
      3. Floor every family at `floor` (default 0.02) to prevent any family
         collapsing to zero.
      4. Renormalise so weights sum to 1.0.

    Returns a WeighterResult with:
      - weights: {family: weight} for every family in FAMILIES
      - fired:   list of FiredRule (for UI transparency)
      - baseline, floor: parameters used
    """
    cfg = load_scenario_weights_config()
    baseline: float = cfg["baseline"]
    floor: float = cfg["floor"]

    weights: dict[str, float] = {f: baseline for f in FAMILIES}
    fired: list[FiredRule] = []

    for rule in cfg["rules"]:
        family = rule["family"]
        if family not in weights:
            raise ValueError(f"Rule '{rule['id']}' targets unknown family '{family}'. "
                             f"Known families: {FAMILIES}")
        if not _all_conditions_met(rule.get("when", []), ms):
            continue
        weights[family] += rule["adjustment"]
        fired.append(FiredRule(
            id=rule["id"],
            family=family,
            adjustment=rule["adjustment"],
            comment=rule.get("comment", ""),
        ))

    # Floor (no family below `floor`) then renormalise.
    weights = {f: max(w, floor) for f, w in weights.items()}
    total = sum(weights.values())
    if total <= 0:
        # Degenerate case shouldn't occur given non-zero floor, but be safe.
        weights = {f: 1.0 / len(FAMILIES) for f in FAMILIES}
    else:
        weights = {f: w / total for f, w in weights.items()}

    return WeighterResult(
        weights=weights,
        fired=fired,
        baseline=baseline,
        floor=floor,
    )
