"""
Structure selector — deterministic rules engine.

Reads decision_rules from knowledge/defaults/structure_defaults.json and evaluates
each rule's conditions against the TradeView. Returns an ordered shortlist of
structures with rationale.

No LLM is involved in this step. The same inputs always produce the same shortlist.

Condition evaluation:
  Direct fields (direction_conviction, timing_conviction):
      list membership — ["high", "medium"] matches if view.X in ["high", "medium"]

  Derived fields:
      budget_type        — "premium_constrained" if budget_usd set and no max_loss;
                           "any" matches unconditionally
      target_level_known — True if view.magnitude_pct is not None
      view_aligns_with_skew / skew_magnitude — skew-based conditions; always False/low
                           until vol regime classification is re-introduced.

Rule priority: rules are evaluated in order; for each matching rule, its shortlist
items are added to the output (deduplicated). Earlier matches set the rank of
a given structure. Capped at max_shortlist_size from config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.schema import ResolvedConfig, StructureConfig
from knowledge_engine.loader import get_decision_rules, get_structure_catalog
from knowledge_engine.models import (
    StructureShortlistItem,
    StructureSelectionResult,
    TradeView,
)


def select_structures(
    view: TradeView,
    cfg: ResolvedConfig,
) -> StructureSelectionResult:
    """
    Run the rules engine and return an ordered shortlist.

    Args:
        view: The PM's structured trade view.
        cfg:  Resolved config (controls max_shortlist_size, exclusions etc.)

    Returns:
        StructureSelectionResult with shortlist and context notes.
    """
    rules   = get_decision_rules()
    catalog = get_structure_catalog()
    struct_cfg = cfg.structures

    derived = _DerivedConditions.from_view(view)

    shortlist_map: dict[str, StructureShortlistItem] = {}   # structure_id → item
    rules_fired: list[str] = []
    rank_counter = 0

    for rule in rules:
        if not _rule_matches(rule["conditions"], view, derived):
            continue

        rules_fired.append(rule["id"])
        sizing_modifier = rule.get("sizing_modifier")

        for structure_id in rule["shortlist"]:
            # Skip excluded structures
            if structure_id in struct_cfg.excluded_structures:
                continue
            # Skip if already in shortlist (preserve earlier rank)
            if structure_id in shortlist_map:
                continue

            # Determine if this structure is directional and pick the right leg
            resolved_id = _resolve_direction(structure_id, view.direction)

            if resolved_id in shortlist_map:
                continue

            cat_info = catalog.get(resolved_id) or catalog.get(structure_id, {})
            is_exotic = cat_info.get("comparison_only", False)

            rank_counter += 1
            shortlist_map[resolved_id] = StructureShortlistItem(
                structure_id    = resolved_id,
                display_name    = cat_info.get("display_name", resolved_id),
                rank            = rank_counter,
                rationale       = rule["rationale"],
                rule_id         = rule["id"],
                sizing_modifier = sizing_modifier,
                caution         = cat_info.get("caution"),
                optimised_for   = cat_info.get("optimised_for", ""),
                is_exotic       = is_exotic,
            )

    # Apply preferred structures from config (bump to front)
    shortlist = _apply_preferences(list(shortlist_map.values()), struct_cfg)

    # Cap to max_shortlist_size (exotics don't count toward cap)
    vanilla_items = [s for s in shortlist if not s.is_exotic][:struct_cfg.max_shortlist_size]
    exotic_items  = [s for s in shortlist if s.is_exotic]
    shortlist = vanilla_items + exotic_items

    # Always include vanilla baseline if configured and not already present
    if struct_cfg.always_include_vanilla_baseline:
        shortlist = _ensure_vanilla_baseline(shortlist, view, catalog, rank_counter)

    return StructureSelectionResult(
        shortlist       = shortlist,
        rules_fired     = rules_fired,
    )


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

@dataclass
class _DerivedConditions:
    budget_type:            str    # "premium_constrained" | "any"
    target_level_known:     bool
    view_aligns_with_skew:  bool   # always False — skew classification not available
    skew_magnitude:         str    # always "low" — skew classification not available

    @classmethod
    def from_view(cls, view: TradeView) -> "_DerivedConditions":
        return cls(
            budget_type           = "premium_constrained" if view.has_budget_constraint else "any",
            target_level_known    = view.has_target_level,
            view_aligns_with_skew = False,
            skew_magnitude        = "low",
        )


def _rule_matches(conditions: dict, view: TradeView, derived: _DerivedConditions) -> bool:
    """Return True if all conditions in the rule are satisfied."""
    for field, allowed_values in conditions.items():
        if not _condition_satisfied(field, allowed_values, view, derived):
            return False
    return True


def _condition_satisfied(
    field: str,
    allowed_values: list,
    view: TradeView,
    derived: _DerivedConditions,
) -> bool:
    match field:
        case "direction_conviction":
            return view.direction_conviction in allowed_values
        case "timing_conviction":
            return view.timing_conviction in allowed_values
        case "budget_type":
            return "any" in allowed_values or derived.budget_type in allowed_values
        case "target_level_known":
            return derived.target_level_known in allowed_values
        case "view_aligns_with_skew":
            return derived.view_aligns_with_skew in allowed_values
        case "skew_magnitude":
            return derived.skew_magnitude in allowed_values
        case _:
            # Unknown condition — skip (don't block the rule)
            return True


# ---------------------------------------------------------------------------
# Direction resolution and utilities
# ---------------------------------------------------------------------------

def _resolve_direction(structure_id: str, direction: str) -> str:
    """
    Map a generic structure id (e.g. "vanilla_call") to the direction-specific one.

    "vanilla_call" + "base_lower" → "vanilla_put"
    "call_spread"  + "base_lower" → "put_spread"
    etc.
    """
    if direction == "base_lower":
        return {
            "vanilla_call": "vanilla_put",
            "call_spread":  "put_spread",
        }.get(structure_id, structure_id)
    return structure_id


def _apply_preferences(
    shortlist: list[StructureShortlistItem],
    struct_cfg: StructureConfig,
) -> list[StructureShortlistItem]:
    """Bump preferred structures to the front of the shortlist."""
    if not struct_cfg.preferred_structures:
        return shortlist
    preferred = [s for s in shortlist if s.structure_id in struct_cfg.preferred_structures]
    others    = [s for s in shortlist if s.structure_id not in struct_cfg.preferred_structures]
    return preferred + others


def _ensure_vanilla_baseline(
    shortlist: list[StructureShortlistItem],
    view: TradeView,
    catalog: dict,
    current_rank: int,
) -> list[StructureShortlistItem]:
    """
    If no vanilla call/put is already in the shortlist, append it as the baseline
    comparison instrument (not counted toward max_shortlist_size cap).
    """
    vanilla_id = "vanilla_call" if view.direction == "base_higher" else "vanilla_put"
    if any(s.structure_id == vanilla_id for s in shortlist):
        return shortlist

    cat_info = catalog.get(vanilla_id, {})
    baseline = StructureShortlistItem(
        structure_id    = vanilla_id,
        display_name    = cat_info.get("display_name", vanilla_id),
        rank            = current_rank + 1,
        rationale       = "Vanilla baseline included for comparison.",
        rule_id         = "BASELINE",
        sizing_modifier = None,
        caution         = None,
        optimised_for   = cat_info.get("optimised_for", ""),
    )
    return shortlist + [baseline]


