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
_KEY_CHALLENGERS = {
    "vanilla": "Vanilla Option",
    "european_digital": "European Digital",
    "european_digital_rko": "European Digital with RKO",
}


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
class VariantComparison:
    chosen: "RankedVariant"
    challenger: "RankedVariant"
    verdict: Literal["chosen_preferred", "challenger_preferred", "close_call", "not_comparable"]
    confidence: Literal["strong", "moderate", "close"]
    headline: str


@dataclass
class UnavailableComparison:
    challenger_id: str
    challenger_display_name: str
    reason: Literal["not_shortlisted", "not_priceable", "no_scenario_rows"]
    plain: str


@dataclass
class RecommendationExplanationPack:
    chosen_id: str
    chosen_display_name: str
    chosen_variant_label: str | None = None
    recommendation_basis: Literal["scenario_weighted_pnl", "affinity_rank"] = "scenario_weighted_pnl"
    is_call: bool = True
    ranked_variants: list["RankedVariant"] = field(default_factory=list)
    summary_reasons: list[Reason] = field(default_factory=list)
    construction_reasons: list[ConstructionReason] = field(default_factory=list)
    risk_reasons: list[Reason] = field(default_factory=list)
    comparisons: dict[str, PairwiseComparison] = field(default_factory=dict)
    variant_comparisons: list[VariantComparison] = field(default_factory=list)
    unavailable_comparisons: list[UnavailableComparison] = field(default_factory=list)
    active_context_ids: list[str] = field(default_factory=list)
    active_context_comments: list[str] = field(default_factory=list)


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
class RankedStructure:
    structure_id: str
    display_name: str
    affinity_rank: int
    scenario_rank: int
    base_score_ccy: float | None
    pm_score_ccy: float | None


@dataclass(frozen=True)
class RankedVariant:
    structure_id: str
    structure_display_name: str
    variant_label: str
    affinity_rank: int
    scenario_rank: int
    strikes: tuple[float, ...]
    barrier: float | None
    notional: float | None
    base_score_ccy: float
    pm_score_ccy: float


@dataclass(frozen=True)
class ScenarioAggregate:
    scenario_ids: tuple[str, ...]
    avg_pnl_pct: float | None
    worst_pnl_pct: float | None
    best_pnl_pct: float | None


@dataclass(frozen=True)
class ScenarioAggregates:
    slow_path: ScenarioAggregate
    correct_path: ScenarioAggregate
    wrong_way: ScenarioAggregate
    overshoot: ScenarioAggregate
    vol_sensitivity: ScenarioAggregate
    expiry_target_price_pct: float | None
    expiry_target_pnl_pct: float | None


@dataclass(frozen=True)
class VariantEvaluation:
    variant: object
    rows: list[dict]
    base_score: ScoreResult
    pm_score: ScoreResult
    aggregates: ScenarioAggregates


@dataclass(frozen=True)
class ComparatorInputs:
    scenarios: list[dict]
    priced_variants_by_structure: dict[str, list[object]]
    scenario_rows_by_structure: dict[str, list[dict]]
    base_scores_by_structure: dict[str, ScoreResult]
    pm_scores_by_structure: dict[str, ScoreResult]
    score_pairs_by_structure: dict[str, StructureScorePair]
    scenario_aggregates_by_structure: dict[str, ScenarioAggregates]
    variant_evaluations_by_structure: dict[str, list[VariantEvaluation]]
    active_context: str | None = None   # fired base-weighting context id (for commentary)
    weights: dict[str, float] = field(default_factory=dict)   # resolved PM scenario weights
                                                              # (reused by Tier-2 to score a PM structure)


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
    chosen_score = _score_ccy(scenario_scores_by_structure.get(chosen.structure_id))
    challenger_score = _score_ccy(scenario_scores_by_structure.get(challenger.structure_id))
    score_delta = chosen_score - challenger_score

    chosen_edges: list[Reason] = []
    challenger_edges: list[Reason] = []
    caveats: list[Reason] = []
    counterfactuals: list[Reason] = []

    verdict: Literal["chosen_preferred", "challenger_preferred", "close_call", "not_comparable"]
    confidence: Literal["strong", "moderate", "close"]

    if abs(score_delta) < 0.25:
        verdict = "close_call"
        confidence = "close"
    elif score_delta > 0:
        verdict = "chosen_preferred"
        confidence = "strong" if abs(score_delta) >= 1.0 else "moderate"
    else:
        verdict = "challenger_preferred"
        confidence = "strong" if abs(score_delta) >= 1.0 else "moderate"

    if score_delta > 0.25:
        chosen_edges.append(make_reason(
            "scenario_fit.better_weighted_pnl",
            polarity="chosen_edge",
            materiality=_gap_materiality(score_delta),
        ))
    elif score_delta < -0.25:
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
    variant_evaluations_by_structure: dict[str, list[VariantEvaluation]] | None = None,
    user_email: str | None = None,
) -> RecommendationExplanationPack:
    if not selector_result.shortlist:
        raise ValueError("Cannot build explanation pack without a shortlisted chosen structure")

    prefs = preferences or PMPreferences()
    ranked_variants = rank_variants_by_scenario_ccy(selector_result, variant_evaluations_by_structure or {})
    ranked_structures = rank_structures_by_scenario_score(selector_result, scenario_scores_by_structure)
    chosen_variant = ranked_variants[0] if ranked_variants else None
    chosen_rank = ranked_structures[0] if ranked_structures else None
    chosen = _item_for_structure_id(
        selector_result,
        (
            chosen_variant.structure_id
            if chosen_variant is not None
            else chosen_rank.structure_id if chosen_rank is not None
            else selector_result.shortlist[0].structure_id
        ),
    )

    summary_reasons: list[Reason] = []
    risk_reasons = _risk_reasons_for_structure(chosen.structure_id)

    comparison_targets = _pick_comparison_targets(
        selector_result,
        scenario_scores_by_structure,
        chosen.structure_id,
    )
    if len(ranked_variants) >= 2:
        delta = ranked_variants[0].pm_score_ccy - ranked_variants[1].pm_score_ccy
    elif comparison_targets:
        first_challenger = comparison_targets[0]
        delta = _score_ccy(scenario_scores_by_structure.get(chosen.structure_id)) - _score_ccy(
            scenario_scores_by_structure.get(first_challenger.structure_id)
        )
    else:
        delta = 0.0

    if delta > 0.25:
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
    variant_comparisons = _variant_comparisons(ranked_variants)
    unavailable_comparisons = _unavailable_key_comparisons(
        selector_result,
        chosen.structure_id,
        priced_variants_by_structure,
        scenario_scores_by_structure,
    )

    from knowledge_engine.scenario_weighter import compute_family_weights as _cwf
    _weighter = _cwf(
        market_state,
        primary_objective=prefs.primary_objective,
        trade_management=prefs.trade_management,
        user_email=user_email,
    )
    _active_ids: list[str] = []
    _active_comments: list[str] = []
    for _ctx in getattr(_weighter, "fired", []):
        _active_ids.append(_ctx.id)
        _active_comments.append(_ctx.comment)

    return RecommendationExplanationPack(
        chosen_id=chosen.structure_id,
        chosen_display_name=(
            _variant_name(chosen_variant)
            if chosen_variant is not None else chosen.display_name
        ),
        chosen_variant_label=chosen_variant.variant_label if chosen_variant is not None else None,
        recommendation_basis="scenario_weighted_pnl" if (ranked_variants or ranked_structures) else "affinity_rank",
        is_call=market_state.put_call == "Call",
        ranked_variants=ranked_variants,
        summary_reasons=_dedupe_reasons(summary_reasons),
        construction_reasons=[],
        risk_reasons=_dedupe_reasons(risk_reasons),
        comparisons=comparisons,
        variant_comparisons=variant_comparisons,
        unavailable_comparisons=unavailable_comparisons,
        active_context_ids=_active_ids,
        active_context_comments=_active_comments,
    )


def rank_structures_by_scenario_score(
    selector_result: StructureSelectionResult,
    scenario_scores_by_structure: dict[str, object],
    base_scores_by_structure: dict[str, object] | None = None,
) -> list[RankedStructure]:
    """Rank priceable shortlisted structures by scenario USD P&L, preserving affinity rank."""
    by_score: list[tuple[float, int, StructureShortlistItem]] = []
    for item in selector_result.shortlist:
        score = scenario_scores_by_structure.get(item.structure_id)
        if score is None:
            continue
        by_score.append((_score_ccy(score), item.rank, item))

    by_score.sort(key=lambda x: (-x[0], x[1]))

    ranked: list[RankedStructure] = []
    for scenario_rank, (_, _, item) in enumerate(by_score, start=1):
        base_score = (
            _score_ccy(base_scores_by_structure.get(item.structure_id))
            if base_scores_by_structure and item.structure_id in base_scores_by_structure
            else None
        )
        pm_score = _score_ccy(scenario_scores_by_structure.get(item.structure_id))
        ranked.append(RankedStructure(
            structure_id=item.structure_id,
            display_name=item.display_name,
            affinity_rank=item.rank,
            scenario_rank=scenario_rank,
            base_score_ccy=base_score,
            pm_score_ccy=pm_score,
        ))
    return ranked


def rank_variants_by_scenario_ccy(
    selector_result: StructureSelectionResult,
    variant_evaluations_by_structure: dict[str, list[VariantEvaluation]],
) -> list[RankedVariant]:
    """Rank concrete priced variants by PM-overlay weighted P&L in base currency."""
    items = {item.structure_id: item for item in selector_result.shortlist}
    by_score: list[tuple[float, int, int, str, VariantEvaluation]] = []
    for structure_id, evaluations in variant_evaluations_by_structure.items():
        item = items.get(structure_id)
        if item is None:
            continue
        for variant_index, evaluation in enumerate(evaluations):
            if evaluation.pm_score.score_ccy is None or evaluation.base_score.score_ccy is None:
                continue
            by_score.append((
                evaluation.pm_score.score_ccy,
                item.rank,
                variant_index,
                structure_id,
                evaluation,
            ))

    by_score.sort(key=lambda x: (-x[0], x[1], x[2]))

    ranked: list[RankedVariant] = []
    for scenario_rank, (_, _, _, structure_id, evaluation) in enumerate(by_score, start=1):
        item = items[structure_id]
        variant = evaluation.variant
        ranked.append(RankedVariant(
            structure_id=structure_id,
            structure_display_name=item.display_name,
            variant_label=_variant_label_with_details(variant),
            affinity_rank=item.rank,
            scenario_rank=scenario_rank,
            strikes=tuple(float(k) for k in getattr(variant, "strikes", [])),
            barrier=getattr(variant, "barrier", None),
            notional=getattr(variant, "structure_notional", None),
            base_score_ccy=float(evaluation.base_score.score_ccy),
            pm_score_ccy=float(evaluation.pm_score.score_ccy),
        ))
    return ranked


def build_comparator_inputs(
    market_state: MarketState,
    selector_result: StructureSelectionResult,
    *,
    target: float,
    is_call: bool,
    stop_price: float | None,
    loss_budget: float | None,
    preferences: PMPreferences | None = None,
    smile: object | None = None,
    user_email: str | None = None,
) -> ComparatorInputs:
    """
    Build real pricing/scenario inputs for the comparator from existing engines.

    The comparator should remain an explanation layer, so this function is only
    a thin adapter around the existing variant pricer, scenario generator,
    scenario pricer, scenario weighter, and scenario scorer.

    ``smile`` is an optional ``VolSurface`` passed straight through to both the
    variant pricer (entry pricing per strike) and the scenario pricer (sticky-
    delta MtM on the vanilla legs). When None, both fall back to flat ATM vol
    (legacy behaviour). Digital / RKO / european_rko legs stay flat regardless.
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
        user_email=user_email,
    )
    base_weights = _base_weights_from_weighter(weighter)

    priced_variants_by_structure: dict[str, list[object]] = {}
    scenario_rows_by_structure: dict[str, list[dict]] = {}
    base_scores_by_structure: dict[str, ScoreResult] = {}
    pm_scores_by_structure: dict[str, ScoreResult] = {}
    score_pairs_by_structure: dict[str, StructureScorePair] = {}
    scenario_aggregates_by_structure: dict[str, ScenarioAggregates] = {}
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
                smile=smile,
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
                surface=smile,
            )
            base_score = score_structure(rows, base_weights)
            pm_score = score_structure(rows, weighter.weights)
            aggregates = summarize_scenario_rows(rows)
            evaluations.append(VariantEvaluation(
                variant=variant,
                rows=rows,
                base_score=base_score,
                pm_score=pm_score,
                aggregates=aggregates,
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
        scenario_aggregates_by_structure[item.structure_id] = primary.aggregates
        variant_evaluations_by_structure[item.structure_id] = evaluations

    return ComparatorInputs(
        scenarios=scenarios,
        priced_variants_by_structure=priced_variants_by_structure,
        scenario_rows_by_structure=scenario_rows_by_structure,
        base_scores_by_structure=base_scores_by_structure,
        pm_scores_by_structure=pm_scores_by_structure,
        score_pairs_by_structure=score_pairs_by_structure,
        scenario_aggregates_by_structure=scenario_aggregates_by_structure,
        variant_evaluations_by_structure=variant_evaluations_by_structure,
        active_context=(weighter.base_fired.id if getattr(weighter, "base_fired", None) else None),
        weights=dict(weighter.weights),
    )


def summarize_scenario_rows(rows: list[dict]) -> ScenarioAggregates:
    return ScenarioAggregates(
        slow_path=_aggregate_rows(rows, ("25%T|S", "50%T|S", "25%T|t%→K", "50%T|t%→K")),
        correct_path=_aggregate_rows(rows, ("25%T|t%→K", "50%T|t%→K", "Expiry|K")),
        wrong_way=_aggregate_rows(rows, ("25%T|−1σ", "50%T|−1σ", "Expiry|−1σ")),
        overshoot=_aggregate_rows(rows, ("25%T|K+½σ", "50%T|K+½σ", "Expiry|K+½σ")),
        vol_sensitivity=_aggregate_rows(rows, ("2w|Δvol", "25%T|Δvol", "50%T|Δvol")),
        expiry_target_price_pct=_row_value(rows, "Expiry|K", "price_pct"),
        expiry_target_pnl_pct=_row_value(rows, "Expiry|K", "pnl_pct"),
    )


def _pick_comparison_targets(
    selector_result: StructureSelectionResult,
    scenario_scores_by_structure: dict[str, object],
    chosen_structure_id: str | None = None,
) -> list[StructureShortlistItem]:
    if not selector_result.shortlist:
        return []

    chosen_id = chosen_structure_id or selector_result.shortlist[0].structure_id
    targets: list[StructureShortlistItem] = []

    vanilla = next(
        (
            item for item in selector_result.shortlist
            if item.structure_id == "vanilla" and item.structure_id != chosen_id
        ),
        None,
    )
    if vanilla is not None:
        targets.append(vanilla)

    for item in selector_result.shortlist[1:]:
        if item.structure_id == chosen_id:
            continue
        if any(existing.structure_id == item.structure_id for existing in targets):
            continue
        targets.append(item)
        if len(targets) >= 2:
            break

    for item in _scenario_ranked_items(selector_result, scenario_scores_by_structure):
        if item.structure_id == chosen_id:
            continue
        if any(existing.structure_id == item.structure_id for existing in targets):
            continue
        targets.append(item)
        if len(targets) >= 5:
            break

    return targets[:5]


def _scenario_ranked_items(
    selector_result: StructureSelectionResult,
    scenario_scores_by_structure: dict[str, object],
) -> list[StructureShortlistItem]:
    by_id = {item.structure_id: item for item in selector_result.shortlist}
    ranked = rank_structures_by_scenario_score(selector_result, scenario_scores_by_structure)
    return [by_id[r.structure_id] for r in ranked if r.structure_id in by_id]


def _unavailable_key_comparisons(
    selector_result: StructureSelectionResult,
    chosen_structure_id: str,
    priced_variants_by_structure: dict[str, list[object]],
    scenario_scores_by_structure: dict[str, object],
) -> list[UnavailableComparison]:
    shortlisted_ids = {item.structure_id for item in selector_result.shortlist}
    result: list[UnavailableComparison] = []
    for structure_id, display_name in _KEY_CHALLENGERS.items():
        if structure_id == chosen_structure_id:
            continue
        if structure_id not in shortlisted_ids:
            result.append(UnavailableComparison(
                challenger_id=structure_id,
                challenger_display_name=display_name,
                reason="not_shortlisted",
                plain="Not scenario-priced because it did not make the affinity shortlist.",
            ))
            continue
        if structure_id not in priced_variants_by_structure:
            result.append(UnavailableComparison(
                challenger_id=structure_id,
                challenger_display_name=display_name,
                reason="not_priceable",
                plain="Shortlisted, but no priceable variant was available for the scenario grid.",
            ))
            continue
        if structure_id not in scenario_scores_by_structure:
            result.append(UnavailableComparison(
                challenger_id=structure_id,
                challenger_display_name=display_name,
                reason="no_scenario_rows",
                plain="Shortlisted and priceable, but no scenario-ranked comparison was available.",
            ))
    return result


def _variant_comparisons(ranked_variants: list[RankedVariant]) -> list[VariantComparison]:
    if not ranked_variants:
        return []
    chosen = ranked_variants[0]
    comparisons: list[VariantComparison] = []
    for challenger in ranked_variants[1:6]:
        delta = chosen.pm_score_ccy - challenger.pm_score_ccy
        if abs(delta) < 0.25:
            verdict: Literal["chosen_preferred", "challenger_preferred", "close_call", "not_comparable"] = "close_call"
            confidence: Literal["strong", "moderate", "close"] = "close"
        elif delta > 0:
            verdict = "chosen_preferred"
            confidence = "strong" if abs(delta) >= 1.0 else "moderate"
        else:
            verdict = "challenger_preferred"
            confidence = "strong" if abs(delta) >= 1.0 else "moderate"
        comparisons.append(VariantComparison(
            chosen=chosen,
            challenger=challenger,
            verdict=verdict,
            confidence=confidence,
            headline=(
                f"{_variant_name(chosen)} is preferred to {_variant_name(challenger)} "
                "on PM weighted P&L"
                if verdict == "chosen_preferred"
                else f"{_variant_name(chosen)} and {_variant_name(challenger)} are a close call"
            ),
        ))
    return comparisons


def _variant_name(variant: RankedVariant) -> str:
    return f"{variant.structure_display_name} - {variant.variant_label}"


def _variant_label_with_details(variant: object) -> str:
    label = str(getattr(variant, "variant_label", "")).strip() or "Variant"
    details: list[str] = []
    strikes = [float(k) for k in getattr(variant, "strikes", [])]
    if strikes:
        details.append(" / ".join(f"{k:.4f}" for k in strikes))
    barrier = getattr(variant, "barrier", None)
    if barrier is not None:
        details.append(f"KO {float(barrier):.4f}")
    if not details:
        return label
    return f"{label} ({'; '.join(details)})"


def _item_for_structure_id(
    selector_result: StructureSelectionResult,
    structure_id: str,
) -> StructureShortlistItem:
    for item in selector_result.shortlist:
        if item.structure_id == structure_id:
            return item
    raise KeyError(f"Structure {structure_id!r} is not in the shortlist")


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


def _score_ccy(score: object | None) -> float:
    if score is None:
        return 0.0
    if hasattr(score, "score_ccy") and getattr(score, "score_ccy") is not None:
        return float(getattr(score, "score_ccy"))
    if hasattr(score, "score_pct"):
        return float(getattr(score, "score_pct"))
    return float(score)


def _premium_pct(variants: list[object]) -> float:
    if not variants:
        return 0.0
    return float(getattr(variants[0], "net_premium_pct", 0.0))


def _gap_materiality(delta: float) -> Materiality:
    if delta >= 1.0:
        return "high"
    if delta >= 0.25:
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


def _aggregate_rows(rows: list[dict], scenario_ids: tuple[str, ...]) -> ScenarioAggregate:
    values = [
        float(row["pnl_pct"])
        for row in rows
        if row.get("scenario_id") in scenario_ids and row.get("pnl_pct") is not None
    ]
    if not values:
        return ScenarioAggregate(
            scenario_ids=scenario_ids,
            avg_pnl_pct=None,
            worst_pnl_pct=None,
            best_pnl_pct=None,
        )
    return ScenarioAggregate(
        scenario_ids=scenario_ids,
        avg_pnl_pct=sum(values) / len(values),
        worst_pnl_pct=min(values),
        best_pnl_pct=max(values),
    )


def _row_value(rows: list[dict], scenario_id: str, field: str) -> float | None:
    for row in rows:
        if row.get("scenario_id") != scenario_id:
            continue
        value = row.get(field)
        return float(value) if value is not None else None
    return None
