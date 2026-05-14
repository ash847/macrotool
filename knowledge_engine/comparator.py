"""
Comparator pack MVP.

This module defines the explanation-pack schema plus a first-pass deterministic
comparator that can populate:
  - recommendation summary reasons
  - recommendation risk reasons
  - one or two pairwise comparisons

Construction reasons are wired into the schema but intentionally left
unpopulated in this MVP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from analytics.market_state import MarketState
from knowledge_engine.models import StructureSelectionResult, StructureShortlistItem
from knowledge_engine.scenario_scorer import ScoreResult


Polarity = Literal["chosen_edge", "challenger_edge", "caveat", "counterfactual"]
Materiality = Literal["high", "medium", "low"]

_CAPPED_STRUCTURES = {"1x1_spread", "1x1.5_spread", "1x2_spread", "seagull"}
_BINARY_STRUCTURES = {"european_digital", "european_digital_rko"}
_BARRIER_STRUCTURES = {"rko", "european_rko", "european_digital_rko"}
_SPREAD_STRUCTURES = {"1x1_spread", "1x1.5_spread", "1x2_spread"}


@dataclass
class Reason:
    code: str
    plain: str
    detail: str | None
    polarity: Polarity
    materiality: Materiality


@dataclass
class ConstructionReason:
    code: str
    plain: str
    strike_label: str
    strike_value: float
    reference: str
    relation: str


@dataclass
class PairwiseComparison:
    chosen_id: str
    challenger_id: str
    verdict: Literal["chosen_preferred", "challenger_preferred", "close_call", "not_comparable"]
    confidence: Literal["strong", "moderate", "close"]
    headline: str
    chosen_edges: list[Reason] = field(default_factory=list)
    challenger_edges: list[Reason] = field(default_factory=list)
    caveats: list[Reason] = field(default_factory=list)
    counterfactuals: list[Reason] = field(default_factory=list)


@dataclass
class RecommendationExplanationPack:
    chosen_id: str
    chosen_display_name: str
    summary_reasons: list[Reason] = field(default_factory=list)
    construction_reasons: list[ConstructionReason] = field(default_factory=list)
    risk_reasons: list[Reason] = field(default_factory=list)
    comparisons: dict[str, PairwiseComparison] = field(default_factory=dict)


@dataclass(frozen=True)
class PMPreferences:
    primary_objective: str = "Balanced"
    trade_management: str = "Standard hold"
    structure_constraint: str = "No restriction"


@dataclass(frozen=True)
class StructureScorePair:
    base: ScoreResult | None
    pm: ScoreResult | None


@dataclass(frozen=True)
class VariantEvaluation:
    variant: object
    rows: list[dict]
    base_score: ScoreResult
    pm_score: ScoreResult


@dataclass(frozen=True)
class ComparatorInputs:
    scenarios: list[dict]
    priced_variants_by_structure: dict[str, list[object]]
    scenario_rows_by_structure: dict[str, list[dict]]
    base_scores_by_structure: dict[str, ScoreResult]
    pm_scores_by_structure: dict[str, ScoreResult]
    score_pairs_by_structure: dict[str, StructureScorePair]
    variant_evaluations_by_structure: dict[str, list[VariantEvaluation]]


@dataclass(frozen=True)
class ReasonTemplate:
    plain: str


REASON_CATALOG: dict[str, ReasonTemplate] = {
    "scenario_fit.better_weighted_pnl": ReasonTemplate(
        plain="Better weighted P&L across the scenario set that matters most for this trade",
    ),
    "scenario_fit.better_slow_path": ReasonTemplate(
        plain="Holds up better if the move is slower or noisier than expected",
    ),
    "scenario_fit.better_correct_path": ReasonTemplate(
        plain="Delivers a better payoff if the trade develops along the expected path",
    ),
    "scenario_fit.weaker_wrong_way": ReasonTemplate(
        plain="Loses more if the market moves clearly against the view",
    ),
    "selection_fit.target_supports_spread": ReasonTemplate(
        plain="A defined target makes the spread structure a more natural fit",
    ),
    "selection_fit.vanilla_preserves_upside": ReasonTemplate(
        plain="Keeps full upside if the move runs well beyond the target",
    ),
    "premium.cheaper_upfront": ReasonTemplate(
        plain="Lower upfront premium",
    ),
    "premium.higher_premium_cleaner_risk": ReasonTemplate(
        plain="Higher upfront premium, but a cleaner and more controlled risk profile",
    ),
    "risk.capped_upside": ReasonTemplate(
        plain="Upside is capped if the move carries well beyond the target",
    ),
    "risk.binary_expiry_risk": ReasonTemplate(
        plain="Outcome depends heavily on where spot finishes at expiry",
    ),
    "risk.barrier_path_risk": ReasonTemplate(
        plain="The trade can be disrupted by the path of the move, not just the final outcome",
    ),
    "counterfactual.vanilla_if_uncapped_upside_dominates": ReasonTemplate(
        plain="Vanilla would be more attractive if keeping full upside mattered more than reducing premium",
    ),
}


def make_reason(
    code: str,
    *,
    polarity: Polarity,
    materiality: Materiality = "medium",
    detail: str | None = None,
) -> Reason:
    template = REASON_CATALOG[code]
    return Reason(
        code=code,
        plain=template.plain,
        detail=detail,
        polarity=polarity,
        materiality=materiality,
    )


def compare_structures(
    chosen: StructureShortlistItem,
    challenger: StructureShortlistItem,
    market_state: MarketState,
    scenario_scores_by_structure: dict[str, object],
    priced_variants_by_structure: dict[str, list[object]],
    preferences: PMPreferences | None = None,
) -> PairwiseComparison:
    prefs = preferences or PMPreferences()
    chosen_score = _score_pct(scenario_scores_by_structure.get(chosen.structure_id))
    challenger_score = _score_pct(scenario_scores_by_structure.get(challenger.structure_id))
    score_delta = chosen_score - challenger_score

    chosen_edges: list[Reason] = []
    challenger_edges: list[Reason] = []
    caveats: list[Reason] = []
    counterfactuals: list[Reason] = []

    verdict: Literal["chosen_preferred", "challenger_preferred", "close_call", "not_comparable"]
    confidence: Literal["strong", "moderate", "close"]

    if abs(score_delta) < 0.0025:
        verdict = "close_call"
        confidence = "close"
    elif score_delta > 0:
        verdict = "chosen_preferred"
        confidence = "strong" if abs(score_delta) >= 0.01 else "moderate"
    else:
        verdict = "challenger_preferred"
        confidence = "strong" if abs(score_delta) >= 0.01 else "moderate"

    if score_delta > 0.001:
        chosen_edges.append(make_reason(
            "scenario_fit.better_weighted_pnl",
            polarity="chosen_edge",
            materiality=_gap_materiality(score_delta),
        ))
    elif score_delta < -0.001:
        challenger_edges.append(make_reason(
            "scenario_fit.better_weighted_pnl",
            polarity="challenger_edge",
            materiality=_gap_materiality(abs(score_delta)),
        ))

    if market_state.target_z is not None and chosen.structure_id in _SPREAD_STRUCTURES:
        chosen_edges.append(make_reason(
            "selection_fit.target_supports_spread",
            polarity="chosen_edge",
            materiality="medium",
        ))

    premium_delta = _premium_pct(priced_variants_by_structure.get(chosen.structure_id, [])) - _premium_pct(
        priced_variants_by_structure.get(challenger.structure_id, [])
    )
    if abs(premium_delta) >= 0.0005:
        premium_materiality = _premium_materiality(abs(premium_delta), prefs)
        if premium_delta < 0:
            chosen_edges.append(make_reason(
                "premium.cheaper_upfront",
                polarity="chosen_edge",
                materiality=premium_materiality,
            ))
        else:
            challenger_edges.append(make_reason(
                "premium.cheaper_upfront",
                polarity="challenger_edge",
                materiality=premium_materiality,
            ))

    if challenger.structure_id == "vanilla" and chosen.structure_id in _CAPPED_STRUCTURES:
        challenger_edges.append(make_reason(
            "selection_fit.vanilla_preserves_upside",
            polarity="challenger_edge",
            materiality="medium",
        ))
        counterfactuals.append(make_reason(
            "counterfactual.vanilla_if_uncapped_upside_dominates",
            polarity="counterfactual",
            materiality="medium",
        ))

    if challenger.structure_id in _BINARY_STRUCTURES and chosen.structure_id not in _BINARY_STRUCTURES:
        chosen_edges.append(make_reason(
            "scenario_fit.better_slow_path",
            polarity="chosen_edge",
            materiality=_slow_path_materiality(prefs),
        ))

    caveats.extend(_risk_reasons_for_structure(chosen.structure_id))

    headline = _headline_for(verdict, chosen.display_name, challenger.display_name)
    return PairwiseComparison(
        chosen_id=chosen.structure_id,
        challenger_id=challenger.structure_id,
        verdict=verdict,
        confidence=confidence,
        headline=headline,
        chosen_edges=_dedupe_reasons(chosen_edges)[:3],
        challenger_edges=_dedupe_reasons(challenger_edges)[:2],
        caveats=_dedupe_reasons(caveats)[:2],
        counterfactuals=_dedupe_reasons(counterfactuals)[:2],
    )


def build_recommendation_pack(
    market_state: MarketState,
    selector_result: StructureSelectionResult,
    priced_variants_by_structure: dict[str, list[object]],
    scenario_scores_by_structure: dict[str, object],
    preferences: PMPreferences | None = None,
) -> RecommendationExplanationPack:
    if not selector_result.shortlist:
        raise ValueError("Cannot build explanation pack without a shortlisted chosen structure")

    prefs = preferences or PMPreferences()
    chosen = selector_result.shortlist[0]

    summary_reasons: list[Reason] = []
    risk_reasons = _risk_reasons_for_structure(chosen.structure_id)

    comparison_targets = _pick_comparison_targets(selector_result, scenario_scores_by_structure)
    if comparison_targets:
        first_challenger = comparison_targets[0]
        chosen_score = _score_pct(scenario_scores_by_structure.get(chosen.structure_id))
        challenger_score = _score_pct(scenario_scores_by_structure.get(first_challenger.structure_id))
        delta = chosen_score - challenger_score
        if delta > 0.001:
            summary_reasons.append(make_reason(
                "scenario_fit.better_weighted_pnl",
                polarity="chosen_edge",
                materiality=_gap_materiality(delta),
            ))

    if market_state.target_z is not None and chosen.structure_id in _SPREAD_STRUCTURES:
        summary_reasons.append(make_reason(
            "selection_fit.target_supports_spread",
            polarity="chosen_edge",
            materiality="medium",
        ))

    comparisons = {
        challenger.structure_id: compare_structures(
            chosen,
            challenger,
            market_state,
            scenario_scores_by_structure,
            priced_variants_by_structure,
            prefs,
        )
        for challenger in comparison_targets
    }

    return RecommendationExplanationPack(
        chosen_id=chosen.structure_id,
        chosen_display_name=chosen.display_name,
        summary_reasons=_dedupe_reasons(summary_reasons),
        construction_reasons=[],
        risk_reasons=_dedupe_reasons(risk_reasons),
        comparisons=comparisons,
    )


def build_comparator_inputs(
    market_state: MarketState,
    selector_result: StructureSelectionResult,
    *,
    target: float,
    is_call: bool,
    stop_price: float | None,
    loss_budget: float | None,
    preferences: PMPreferences | None = None,
) -> ComparatorInputs:
    """
    Build real pricing/scenario inputs for the comparator from existing engines.

    The comparator should remain an explanation layer, so this function is only
    a thin adapter around the existing variant pricer, scenario generator,
    scenario pricer, scenario weighter, and scenario scorer.
    """
    from analytics.scenario_generator import generate_scenarios
    from analytics.scenario_pricer import price_scenarios
    from analytics.structure_pricer import price_variants
    from knowledge_engine.scenario_scorer import score_structure
    from knowledge_engine.scenario_weighter import compute_family_weights

    prefs = preferences or PMPreferences()
    trade_inputs = {
        "spot": market_state.spot,
        "forward": market_state.fwd,
        "implied_vol": market_state.vol,
        "tenor_years": market_state.T,
        "target": target,
        "r_d": market_state.r_d,
        "r_f": market_state.r_f,
    }
    scenarios = generate_scenarios(trade_inputs)
    weighter = compute_family_weights(
        market_state,
        primary_objective=prefs.primary_objective,
        trade_management=prefs.trade_management,
    )
    base_weights = _base_weights_from_weighter(weighter)

    priced_variants_by_structure: dict[str, list[object]] = {}
    scenario_rows_by_structure: dict[str, list[dict]] = {}
    base_scores_by_structure: dict[str, ScoreResult] = {}
    pm_scores_by_structure: dict[str, ScoreResult] = {}
    score_pairs_by_structure: dict[str, StructureScorePair] = {}
    variant_evaluations_by_structure: dict[str, list[VariantEvaluation]] = {}

    for item in selector_result.shortlist:
        try:
            variants = price_variants(
                market_state,
                item.structure_id,
                target=target,
                is_call=is_call,
                stop_price=stop_price,
                loss_budget=loss_budget,
            )
        except Exception:
            variants = []
        if not variants:
            continue

        priced_variants_by_structure[item.structure_id] = variants
        evaluations: list[VariantEvaluation] = []
        for variant in variants:
            rows = price_scenarios(
                variant,
                item.structure_id,
                scenarios,
                trade_inputs,
                is_call,
            )
            base_score = score_structure(rows, base_weights)
            pm_score = score_structure(rows, weighter.weights)
            evaluations.append(VariantEvaluation(
                variant=variant,
                rows=rows,
                base_score=base_score,
                pm_score=pm_score,
            ))

        if not evaluations:
            continue

        primary = evaluations[0]
        scenario_rows_by_structure[item.structure_id] = primary.rows
        base_scores_by_structure[item.structure_id] = primary.base_score
        pm_scores_by_structure[item.structure_id] = primary.pm_score
        score_pairs_by_structure[item.structure_id] = StructureScorePair(
            base=primary.base_score,
            pm=primary.pm_score,
        )
        variant_evaluations_by_structure[item.structure_id] = evaluations

    return ComparatorInputs(
        scenarios=scenarios,
        priced_variants_by_structure=priced_variants_by_structure,
        scenario_rows_by_structure=scenario_rows_by_structure,
        base_scores_by_structure=base_scores_by_structure,
        pm_scores_by_structure=pm_scores_by_structure,
        score_pairs_by_structure=score_pairs_by_structure,
        variant_evaluations_by_structure=variant_evaluations_by_structure,
    )


def _pick_comparison_targets(
    selector_result: StructureSelectionResult,
    scenario_scores_by_structure: dict[str, object],
) -> list[StructureShortlistItem]:
    if not selector_result.shortlist:
        return []

    chosen = selector_result.shortlist[0]
    targets: list[StructureShortlistItem] = []

    vanilla = next(
        (
            item for item in selector_result.shortlist
            if item.structure_id == "vanilla" and item.structure_id != chosen.structure_id
        ),
        None,
    )
    if vanilla is not None:
        targets.append(vanilla)

    for item in selector_result.shortlist[1:]:
        if item.structure_id == chosen.structure_id:
            continue
        if any(existing.structure_id == item.structure_id for existing in targets):
            continue
        targets.append(item)
        if len(targets) >= 2:
            break

    return targets[:2]


def _risk_reasons_for_structure(structure_id: str) -> list[Reason]:
    reasons: list[Reason] = []
    if structure_id in _CAPPED_STRUCTURES:
        reasons.append(make_reason("risk.capped_upside", polarity="caveat", materiality="medium"))
    if structure_id in _BINARY_STRUCTURES:
        reasons.append(make_reason("risk.binary_expiry_risk", polarity="caveat", materiality="high"))
    if structure_id in _BARRIER_STRUCTURES:
        reasons.append(make_reason("risk.barrier_path_risk", polarity="caveat", materiality="high"))
    return reasons


def _score_pct(score: object | None) -> float:
    if score is None:
        return 0.0
    if hasattr(score, "score_pct"):
        return float(getattr(score, "score_pct"))
    return float(score)


def _premium_pct(variants: list[object]) -> float:
    if not variants:
        return 0.0
    return float(getattr(variants[0], "net_premium_pct", 0.0))


def _gap_materiality(delta: float) -> Materiality:
    if delta >= 0.015:
        return "high"
    if delta >= 0.005:
        return "medium"
    return "low"


def _premium_materiality(delta: float, preferences: PMPreferences) -> Materiality:
    if preferences.primary_objective == "Keep cost low":
        return "high"
    if delta >= 0.01:
        return "high"
    if delta >= 0.003:
        return "medium"
    return "low"


def _slow_path_materiality(preferences: PMPreferences) -> Materiality:
    if preferences.trade_management == "Need defendable mark-to-market":
        return "high"
    return "medium"


def _headline_for(
    verdict: Literal["chosen_preferred", "challenger_preferred", "close_call", "not_comparable"],
    chosen_display_name: str,
    challenger_display_name: str,
) -> str:
    if verdict == "close_call":
        return f"{chosen_display_name} and {challenger_display_name} are a close call"
    if verdict == "challenger_preferred":
        return f"{challenger_display_name} looks stronger than {chosen_display_name} on this comparison"
    if verdict == "not_comparable":
        return f"{chosen_display_name} and {challenger_display_name} are not directly comparable here"
    return f"{chosen_display_name} is preferred to {challenger_display_name}"


def _dedupe_reasons(reasons: list[Reason]) -> list[Reason]:
    seen: set[tuple[str, Polarity]] = set()
    result: list[Reason] = []
    for reason in reasons:
        key = (reason.code, reason.polarity)
        if key in seen:
            continue
        seen.add(key)
        result.append(reason)
    return result


def _base_weights_from_weighter(weighter: object) -> dict[str, float]:
    base_fired = getattr(weighter, "base_fired", None)
    if base_fired is None:
        return dict(getattr(weighter, "weights"))
    multipliers = dict(base_fired.multipliers)
    total = sum(multipliers.values())
    if total <= 0:
        return {cid: 1.0 / len(multipliers) for cid in multipliers}
    return {cid: value / total for cid, value in multipliers.items()}
