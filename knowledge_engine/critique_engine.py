"""
Critique engine — evaluates a PM-supplied structure against their stated view.

Reads the evaluation framework from knowledge/defaults/critique_defaults.json.
Returns a CritiqueOutput with a verdict, dimension scores, and specific notes.

The verdict and notes are skeletal at this stage — the LLM does the
substantive narration using these as structured inputs. The engine's job is
to:
  1. Identify which recommended structure the PM's structure should be
     compared against (from the structure selector output).
  2. Populate each evaluation dimension with a preliminary score based on
     structural properties of the PM's structure vs. the view parameters.
  3. Identify specific scenario weaknesses given the view and vol regime.

This is deliberately conservative — the engine flags potential issues;
the LLM contextualises them. False negatives (missing a real problem) are
worse than false positives (raising something the PM already knows).
"""

from __future__ import annotations

from knowledge_engine.loader import get_critique_dimensions, get_structure_info
from knowledge_engine.models import (
    CritiqueOutput,
    SizingOutput,
    StructureSelectionResult,
    TradeView,
)


# Structures that have unlimited loss exposure on the sold leg
_UNLIMITED_LOSS_STRUCTURES = {"risk_reversal"}

# Structures with path dependency (barrier knock-out)
_PATH_DEPENDENT_STRUCTURES = {"rko_call", "rko_put", "european_digital_rko"}

# Structures with binary gamma near expiry
_BINARY_GAMMA_STRUCTURES = {"european_digital", "european_digital_rko"}


def evaluate_structure(
    view: TradeView,
    pm_structure_id: str,
    selector_result: StructureSelectionResult,
    sizing: SizingOutput | None = None,
) -> CritiqueOutput:
    """
    Evaluate the PM's structure against their view.

    Args:
        view:             The PM's trade view.
        pm_structure_id:  The structure the PM wants to use (from catalog keys).
        selector_result:  The tool's recommended shortlist for this view.
        sizing:           Pre-computed sizing (applied to PM's structure).

    Returns:
        CritiqueOutput with verdict and detailed dimension notes.
    """
    # Find recommended alternative
    recommended = selector_result.shortlist[0] if selector_result.shortlist else None
    recommended_id = recommended.structure_id if recommended else None

    # Load catalog info for PM's structure
    try:
        pm_cat = get_structure_info(pm_structure_id)
    except KeyError:
        pm_cat = {"optimised_for": "unknown", "max_loss": "unknown"}

    dimension_scores: dict[str, str] = {}

    # ------------------------------------------------------------------
    # EV-01: Expected value vs. view
    # ------------------------------------------------------------------
    ev_note, ev_score = _assess_ev(view, pm_structure_id, recommended_id, pm_cat)
    dimension_scores["EV-01"] = ev_score

    # ------------------------------------------------------------------
    # EV-02: Scenario underperformance
    # ------------------------------------------------------------------
    scenario_note, scenario_score = _assess_scenario_weakness(view, pm_structure_id)
    dimension_scores["EV-02"] = scenario_score

    # ------------------------------------------------------------------
    # EV-03: Execution considerations
    # ------------------------------------------------------------------
    exec_note = _assess_execution(pm_structure_id)
    dimension_scores["EV-03"] = "acceptable"

    # ------------------------------------------------------------------
    # EV-04: Hedge effectiveness
    # ------------------------------------------------------------------
    hedge_note = _assess_hedge_effectiveness(pm_structure_id, view, pm_cat)
    dimension_scores["EV-04"] = "acceptable"

    # ------------------------------------------------------------------
    # EV-05: Gamma management
    # ------------------------------------------------------------------
    gamma_note = _assess_gamma(pm_structure_id)
    dimension_scores["EV-05"] = (
        "weak" if pm_structure_id in _BINARY_GAMMA_STRUCTURES else "acceptable"
    )

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    verdict = _determine_verdict(dimension_scores, pm_structure_id, recommended_id)

    return CritiqueOutput(
        verdict                  = verdict,
        pm_structure             = pm_structure_id,
        recommended_alternative  = recommended_id,
        ev_comparison_note       = ev_note,
        scenario_weakness        = scenario_note,
        execution_notes          = exec_note,
        gamma_notes              = gamma_note,
        hedge_effectiveness      = hedge_note,
        sizing                   = sizing,
        dimension_scores         = dimension_scores,
    )


# ---------------------------------------------------------------------------
# Dimension assessors
# ---------------------------------------------------------------------------

def _assess_ev(
    view: TradeView,
    pm_id: str,
    rec_id: str | None,
    pm_cat: dict,
) -> tuple[str, str]:
    """Returns (note, score)."""
    if pm_id == rec_id:
        return ("PM's structure matches the tool recommendation.", "strong")

    if pm_id in _UNLIMITED_LOSS_STRUCTURES:
        return (
            "Risk reversal: unlimited loss on sold leg. EV is theoretically positive "
            "for the view but the tail loss is unbounded. Max loss of recommended "
            "alternatives is premium only.",
            "acceptable",
        )

    return (
        f"Structure {pm_id} vs recommended {rec_id}: "
        "detailed EV comparison requires pricing both structures against the stated view probabilities.",
        "acceptable",
    )


def _assess_scenario_weakness(
    view: TradeView,
    pm_id: str,
) -> tuple[str, str]:
    """Returns (note, score)."""
    weaknesses: list[str] = []

    if pm_id in _PATH_DEPENDENT_STRUCTURES:
        weaknesses.append(
            f"{pm_id} knocks out if spot touches the barrier during the trade's life. "
            "The scenario where spot overshoots and then comes back to target — "
            "which is common in EM currencies — results in zero payout."
        )

    if pm_id in _BINARY_GAMMA_STRUCTURES:
        weaknesses.append(
            "Near expiry, if spot is near the strike, the position's P&L is highly "
            "sensitive to exact fixing — binary gamma risk is significant."
        )

    if not weaknesses:
        weaknesses.append(
            f"No major structural scenario weaknesses identified for {pm_id} given this view. "
            "Detailed scenario P&L matrix will quantify performance across spot/vol outcomes."
        )
        return (" ".join(weaknesses), "strong")

    return (" ".join(weaknesses), "acceptable" if len(weaknesses) == 1 else "weak")


def _assess_execution(pm_id: str) -> str:
    notes: list[str] = []

    if pm_id in _PATH_DEPENDENT_STRUCTURES:
        notes.append(
            "Barrier products require monitoring and operational setup "
            "for barrier event handling."
        )

    if pm_id == "risk_reversal":
        notes.append(
            "Risk reversals require margin for the sold leg. "
            "Ensure credit lines and margin capacity are in place."
        )

    return " ".join(notes) if notes else "No specific execution concerns for this structure."


def _assess_hedge_effectiveness(pm_id: str, view: TradeView, pm_cat: dict) -> str:
    max_loss = pm_cat.get("max_loss", "")
    path_dep = pm_cat.get("path_dependency", "none")

    if path_dep == "barrier_touch":
        return (
            f"Hedge effectiveness is conditional on the path. {pm_id} does not hedge "
            "the scenario where spot briefly touches the barrier before reversing to target. "
            "This is a real path in EM currencies."
        )
    if max_loss == "unlimited_on_sold_leg":
        return (
            "The sold leg of the risk reversal creates an open-ended liability. "
            "The hedge is effective in the primary direction but adds exposure in the opposite direction."
        )
    return "Standard directional hedge — full participation in the stated direction."


def _assess_gamma(pm_id: str) -> str:
    if pm_id in _BINARY_GAMMA_STRUCTURES:
        return (
            f"{pm_id} develops extreme gamma near expiry when spot is close to the strike. "
            "The position becomes difficult to manage in the final days before expiry. "
            "A small spot move around the strike can change P&L from full payout to zero."
        )
    if pm_id in _PATH_DEPENDENT_STRUCTURES:
        return (
            f"{pm_id} develops elevated negative gamma as spot approaches the barrier. "
            "The delta flips sharply near the barrier — hedging becomes costly."
        )
    return "Standard convexity profile. Gamma management is straightforward."


def _determine_verdict(
    dimension_scores: dict[str, str],
    pm_id: str,
    rec_id: str | None,
) -> str:
    weak_count = sum(1 for s in dimension_scores.values() if s == "weak")

    if pm_id == rec_id:
        return "appropriate_for_view"
    if weak_count >= 2:
        return "materially_misaligned"
    if weak_count == 1 or pm_id in _PATH_DEPENDENT_STRUCTURES:
        return "suboptimal_but_defensible"
    return "appropriate_for_view"
