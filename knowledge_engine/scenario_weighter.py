"""
Tier 1 scenario family weighter.

Pure function: MarketState (already view-conditioned by the pipeline) →
{family: weight} summing to 1.0, plus a record of which contexts fired so
the UI can explain the weights to the PM.

The weights are NOT structure-dependent: the same vector is applied to every
shortlisted structure when scoring, so weighted-P&L scores are directly
comparable across structures.

Contexts live in `knowledge/defaults/scenario_weights.json`. Tunable without
code changes — see that file for the context schema and prose explaining each
one. Each context may adjust multiple families simultaneously via the
`adjustments` dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from analytics.market_state import MarketState
from analytics.scenario_generator import FAMILIES

_REPO_ROOT = Path(__file__).parent.parent
_WEIGHTS_PATH = _REPO_ROOT / "knowledge" / "defaults" / "scenario_weights.json"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FiredContext:
    """A context that matched and was applied. Surfaced in the UI for transparency."""
    id: str
    adjustments: dict[str, float]    # family → delta that was applied
    comment: str


@dataclass(frozen=True)
class WeighterResult:
    """Output of `compute_family_weights`. `weights` always sums to ~1.0 with
    every family present (floor enforced)."""
    weights: dict[str, float]
    fired: list[FiredContext]
    baseline: float
    floor: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_weights_cache: dict | None = None


def load_scenario_weights_config() -> dict:
    """
    Load scenario weight contexts. Checks Supabase first (so in-app edits
    persist across sessions), falls back to the local JSON file.
    Uses a module-level cache; call `clear_scenario_weights_cache()` after
    a save to force the next call to re-fetch.
    """
    global _weights_cache
    if _weights_cache is not None:
        return _weights_cache
    try:
        from interface.supabase_logger import fetch_config
        data = fetch_config("scenario_weights")
        if data:
            _weights_cache = data
            return _weights_cache
    except Exception:
        pass
    with open(_WEIGHTS_PATH) as f:
        _weights_cache = json.load(f)
    return _weights_cache


def clear_scenario_weights_cache() -> None:
    """Invalidate the cache so the next call re-fetches from Supabase / file."""
    global _weights_cache
    _weights_cache = None


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

# Map config field names → getters. Two sources are exposed:
#   • MarketState attributes (carry, vol, target distance …)
#   • PM preferences passed through the pipeline (primary objective, trade mgmt)
# Keeps the JSON decoupled from any private MarketState shape and makes explicit
# which fields the context engine knows about.
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
    ">":   lambda a, b: a > b,
    ">=":  lambda a, b: a >= b,
    "<":   lambda a, b: a < b,
    "<=":  lambda a, b: a <= b,
    "==":  lambda a, b: a == b,
    "!=":  lambda a, b: a != b,
    # `in` accepts a list of allowed values — convenient for enums.
    "in":  lambda a, b: a in b,
}


def _evaluate_condition(cond: dict, ms: MarketState, prefs: dict) -> bool:
    field = cond["field"]
    op = cond["op"]
    value = cond["value"]

    if field not in _FIELD_GETTERS:
        raise ValueError(f"Unknown field '{field}' in scenario_weights context. "
                         f"Supported: {sorted(_FIELD_GETTERS)}")
    if op not in _OPS:
        raise ValueError(f"Unknown op '{op}' in scenario_weights context. "
                         f"Supported: {sorted(_OPS)}")

    actual = _FIELD_GETTERS[field](ms, prefs)
    if actual is None:
        # Fields that are None on this MarketState (e.g. target_z when no
        # target supplied) cause the condition to fail rather than raise.
        return False
    return _OPS[op](actual, value)


def _all_conditions_met(when: list[dict], ms: MarketState, prefs: dict) -> bool:
    return all(_evaluate_condition(c, ms, prefs) for c in when)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_family_weights(
    ms: MarketState,
    primary_objective: str = "Balanced",
    trade_management: str = "Standard hold",
) -> WeighterResult:
    """
    Compute scenario family weights from MarketState + PM preferences.

    `primary_objective` and `trade_management` are PM preference enums
    pulled through from the intake form.  Defaults preserve current
    behaviour (Balanced / Standard hold) so existing callers are unchanged.

    Algorithm:
      1. Initialise every family at `baseline` (default 1/8 = 0.125).
      2. For each context whose `when` conditions all evaluate true, apply
         every entry in its `adjustments` dict to the corresponding family
         weight; record the context.
      3. Floor every family at `floor` (default 0.02) to prevent any family
         collapsing to zero.
      4. Renormalise so weights sum to 1.0.

    Returns a WeighterResult with:
      - weights: {family: weight} for every family in FAMILIES
      - fired:   list of FiredContext (for UI transparency)
      - baseline, floor: parameters used
    """
    cfg = load_scenario_weights_config()
    baseline: float = cfg["baseline"]
    floor: float = cfg["floor"]

    prefs = {
        "primary_objective": primary_objective,
        "trade_management":  trade_management,
    }

    weights: dict[str, float] = {f: baseline for f in FAMILIES}
    fired: list[FiredContext] = []

    # Selection: first match wins (contexts are priority-ordered in the JSON).
    # `fired` holds at most one context — when both market-state and preference
    # conditions are evaluated together, exactly one wins per trade.
    for ctx in cfg["contexts"]:
        if not _all_conditions_met(ctx.get("when", []), ms, prefs):
            continue

        adjustments: dict[str, float] = ctx["adjustments"]
        for family, delta in adjustments.items():
            if family not in weights:
                raise ValueError(
                    f"Context '{ctx['id']}' targets unknown family '{family}'. "
                    f"Known families: {FAMILIES}"
                )
            weights[family] += delta

        fired.append(FiredContext(
            id=ctx["id"],
            adjustments=adjustments,
            comment=ctx.get("comment", ""),
        ))
        break  # first match — stop here

    # Floor (no family below `floor`) then renormalise.
    weights = {f: max(w, floor) for f, w in weights.items()}
    total = sum(weights.values())
    if total <= 0:
        weights = {f: 1.0 / len(FAMILIES) for f in FAMILIES}
    else:
        weights = {f: w / total for f, w in weights.items()}

    return WeighterResult(
        weights=weights,
        fired=fired,
        baseline=baseline,
        floor=floor,
    )
