"""
Structure scorer — quantitative structure selection.

Replaces the flat rules engine (structure_selector.py) with a two-step approach:
  1. Gate: hard-filter structures whose market-state requirements are not met
  2. Score: sum affinity scores across dimensions (target_z_abs, carry_regime, atmfsratio)

Direction (call vs put) is implicit in MarketState.put_call, derived from sign(target - fwd).

Returns the same StructureSelectionResult interface as the old selector so downstream
code (context_builder, app) requires no changes.
"""

from __future__ import annotations

from analytics.market_state import MarketState
from knowledge_engine.loader import load_affinity_scores, load_structure_profiles
from knowledge_engine.models import StructureSelectionResult, StructureShortlistItem


def score_structures(
    market_state: MarketState,
    max_primary: int = 3,
) -> StructureSelectionResult:
    """
    Score all eligible structures against the market state and return a ranked shortlist.

    Primary structures (overlay_only=False) are capped at max_primary.
    Overlay structures are appended after, ranked by their own scores.
    """
    profiles = load_structure_profiles()
    scores_cfg = load_affinity_scores()

    thresholds = scores_cfg["thresholds"]
    struct_scores = scores_cfg["structures"]

    buckets = _compute_buckets(market_state, thresholds)

    primary: list[tuple[float, str]] = []
    overlays: list[tuple[float, str]] = []

    for struct_id, score_cfg in struct_scores.items():
        if struct_id not in profiles:
            continue
        profile = profiles[struct_id]

        if not _passes_gates(profile, score_cfg.get("gates", {}), market_state, buckets):
            continue

        total = sum(
            score_cfg.get(dim, {}).get(buckets[dim], 0)
            for dim in ("target_z_abs", "carry_regime", "atmfsratio", "carry_alignment")
        )

        if profile.get("overlay_only", False):
            overlays.append((total, struct_id))
        else:
            primary.append((total, struct_id))

    primary.sort(key=lambda x: -x[0])
    overlays.sort(key=lambda x: -x[0])

    shortlist: list[StructureShortlistItem] = []
    score_notes: list[str] = []

    for rank, (score, struct_id) in enumerate(primary[:max_primary] + overlays, start=1):
        profile = profiles[struct_id]
        score_cfg = struct_scores[struct_id]
        shortlist.append(_make_item(rank, score, struct_id, profile, score_cfg, buckets, market_state))
        score_notes.append(f"{struct_id}={score}")

    return StructureSelectionResult(
        shortlist=shortlist,
        rules_fired=score_notes,
    )


# ---------------------------------------------------------------------------
# Detail / debug
# ---------------------------------------------------------------------------

def get_scoring_detail(market_state: MarketState) -> list[dict]:
    """
    Return the full per-structure score breakdown for display.

    Each row: structure_id, display_name, overlay_only, eligible,
              total_score, dimensions {dim: {bucket, score}}.
    Sorted: primaries first (by score desc), then overlays (by score desc).
    """
    profiles = load_structure_profiles()
    scores_cfg = load_affinity_scores()
    thresholds = scores_cfg["thresholds"]
    struct_scores = scores_cfg["structures"]
    buckets = _compute_buckets(market_state, thresholds)

    rows = []
    for struct_id, score_cfg in struct_scores.items():
        if struct_id not in profiles:
            continue
        profile = profiles[struct_id]
        eligible = _passes_gates(profile, score_cfg.get("gates", {}), market_state, buckets)

        dims = {}
        total = 0
        for dim in ("target_z_abs", "carry_regime", "atmfsratio", "carry_alignment"):
            bucket = buckets[dim]
            score = score_cfg.get(dim, {}).get(bucket, 0) if eligible else 0
            dims[dim] = {"bucket": bucket, "score": score}
            if eligible:
                total += score

        rows.append({
            "structure_id": struct_id,
            "display_name": profile["display_name"],
            "overlay_only": profile.get("overlay_only", False),
            "eligible": eligible,
            "total_score": total if eligible else None,
            "dimensions": dims,
        })

    rows.sort(key=lambda r: (r["overlay_only"], -(r["total_score"] or -99)))
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passes_gates(
    profile: dict,
    gates: dict,
    ms: MarketState,
    buckets: dict,
) -> bool:
    # Structural gate from structure_profiles.json — structure needs a target level to be built
    if profile.get("requires_target") and ms.target_z is None:
        return False
    # target_z_abs_min — target must be at least this many σ from the forward.
    # Defined in affinity_scores.json _gates_schema. Prevents recommending spread/seagull
    # when the target is so near the forward that the sold strike contributes no premium saving.
    if "target_z_abs_min" in gates:
        if ms.target_z is None or abs(ms.target_z) < gates["target_z_abs_min"]:
            return False
    # target_z_abs_max — target must be no more than this many σ from the forward.
    # Defined in affinity_scores.json _gates_schema.
    if "target_z_abs_max" in gates:
        if ms.target_z is None or abs(ms.target_z) > gates["target_z_abs_max"]:
            return False
    return True


def _compute_buckets(ms: MarketState, thresholds: dict) -> dict[str, str]:
    tz_cuts = thresholds["target_z_abs"]
    if ms.target_z is None:
        tz_bucket = "no_target"
    else:
        az = abs(ms.target_z)
        if az < tz_cuts[0]:
            tz_bucket = "near"
        elif az < tz_cuts[1]:
            tz_bucket = "moderate"
        elif az < tz_cuts[2]:
            tz_bucket = "extended"
        else:
            tz_bucket = "far"

    atm_cuts = thresholds["atmfsratio"]
    if ms.atmfsratio is None or ms.atmfsratio < atm_cuts[0]:
        atm_bucket = "low"
    elif ms.atmfsratio < atm_cuts[1]:
        atm_bucket = "medium"
    else:
        atm_bucket = "high"

    carry_alignment = f"with_{atm_bucket}" if ms.with_carry else f"counter_{atm_bucket}"

    return {
        "target_z_abs": tz_bucket,
        "carry_regime": str(ms.carry_regime),
        "atmfsratio": atm_bucket,
        "carry_alignment": carry_alignment,
    }


def _make_item(
    rank: int,
    score: int,
    struct_id: str,
    profile: dict,
    score_cfg: dict,
    buckets: dict,
    ms: MarketState,
) -> StructureShortlistItem:
    breakdown = {
        dim: (buckets[dim], score_cfg.get(dim, {}).get(buckets[dim], 0))
        for dim in ("target_z_abs", "carry_regime", "atmfsratio", "carry_alignment")
    }
    rationale = _build_rationale(profile, breakdown)
    return StructureShortlistItem(
        structure_id=struct_id,
        display_name=profile["display_name"],
        rank=rank,
        rationale=rationale,
        rule_id="SCORER",
        sizing_modifier=None,
        caution=profile.get("major_risk"),
        optimised_for=profile.get("optimal_for", ""),
        is_exotic=profile.get("overlay_only", False),
    )


def _build_rationale(profile: dict, breakdown: dict) -> str:
    drivers = [
        dim.replace("_", " ")
        for dim, (bucket, s) in breakdown.items()
        if s >= 2 and bucket not in ("no_target", "no_carry")
    ]
    penalties = [
        dim.replace("_", " ")
        for dim, (bucket, s) in breakdown.items()
        if s <= -1
    ]
    base = profile.get("optimal_for", "")
    parts = []
    if drivers:
        parts.append(f"scores on: {', '.join(drivers)}")
    if penalties:
        parts.append(f"penalised by: {', '.join(penalties)}")
    return f"{base} [{'; '.join(parts)}]" if parts else base
