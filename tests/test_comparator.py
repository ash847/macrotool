"""
Comparator MVP tests.

Covers:
  - data model shape
  - catalog / make_reason behavior
  - first-pass pairwise comparator
  - first-pass recommendation pack builder
"""

from types import SimpleNamespace

import pytest

from analytics.market_state import MarketState
from analytics.structure_pricer import PricedVariant
from knowledge_engine.comparator import (
    ComparatorInputs,
    ConstructionReason,
    PMPreferences,
    PairwiseComparison,
    REASON_CATALOG,
    RecommendationExplanationPack,
    RankedVariant,
    Reason,
    ScenarioAggregates,
    StructureScorePair,
    UnavailableComparison,
    build_recommendation_pack,
    build_comparator_inputs,
    compare_structures,
    make_reason,
    rank_structures_by_scenario_score,
    rank_variants_by_scenario_ccy,
    summarize_scenario_rows,
)
from knowledge_engine.models import StructureSelectionResult, StructureShortlistItem


def _ms(target_z: float | None = 0.8) -> MarketState:
    return MarketState(
        spot=5.20,
        fwd=5.25,
        vol=0.12,
        T=0.25,
        r_d=0.05,
        r_f=0.10,
        c=0.0,
        carry_regime=1,
        target_z=target_z,
        atmfsratio=1.2,
        put_call="put",
        with_carry=True,
    )


def _item(structure_id: str, rank: int, display_name: str | None = None) -> StructureShortlistItem:
    return StructureShortlistItem(
        structure_id=structure_id,
        display_name=display_name or structure_id,
        rank=rank,
        rationale="test rationale",
        rule_id="rule.test",
        sizing_modifier=None,
        caution=None,
        optimised_for="test",
    )


def _selection_result(*items: StructureShortlistItem) -> StructureSelectionResult:
    return StructureSelectionResult(shortlist=list(items), rules_fired=["test"])


def _score(score_pct: float, score_ccy: float | None = None):
    return SimpleNamespace(score_pct=score_pct, score_ccy=score_ccy)


def _variant(net_premium_pct: float) -> PricedVariant:
    return PricedVariant(
        variant_label="test",
        strikes=[5.60],
        barrier=None,
        net_premium_pct=net_premium_pct,
        breakeven=None,
        payoff_at_target_pct=None,
        rr_at_target=None,
        max_loss_pct=max(net_premium_pct, 0.01),
        wing_ratio=None,
        is_zero_cost=False,
    )


class TestSchema:
    def test_construction_reason_exists_but_can_be_left_empty(self):
        pack = RecommendationExplanationPack(
            chosen_id="1x1_spread",
            chosen_display_name="1x1 Spread",
        )
        assert pack.construction_reasons == []

    def test_construction_reason_can_hold_strike_metadata(self):
        r = ConstructionReason(
            code="construction.sold_leg_at_target",
            plain="Sold leg set at the target level",
            strike_label="sold leg",
            strike_value=5.60,
            reference="target",
            relation="at",
        )
        assert r.reference == "target"
        assert r.relation == "at"


class TestReasonCatalog:
    def test_catalog_contains_only_mvp_codes(self):
        assert set(REASON_CATALOG) == {
            "scenario_fit.better_weighted_pnl",
            "scenario_fit.better_slow_path",
            "scenario_fit.better_correct_path",
            "scenario_fit.weaker_wrong_way",
            "selection_fit.target_supports_spread",
            "selection_fit.vanilla_preserves_upside",
            "premium.cheaper_upfront",
            "premium.higher_premium_cleaner_risk",
            "risk.capped_upside",
            "risk.binary_expiry_risk",
            "risk.barrier_path_risk",
            "counterfactual.vanilla_if_uncapped_upside_dominates",
        }

    def test_make_reason_requires_programmatic_polarity(self):
        with pytest.raises(TypeError):
            make_reason("risk.capped_upside")  # type: ignore[call-arg]

    def test_make_reason_allows_programmatic_materiality(self):
        r = make_reason(
            "premium.cheaper_upfront",
            polarity="challenger_edge",
            materiality="high",
        )
        assert isinstance(r, Reason)
        assert r.polarity == "challenger_edge"
        assert r.materiality == "high"


class TestPairwiseComparisonMVP:
    def test_spread_vs_vanilla_populates_balanced_reasons(self):
        chosen = _item("1x1_spread", 1, "1x1 Spread")
        challenger = _item("vanilla", 2, "Vanilla")

        comparison = compare_structures(
            chosen,
            challenger,
            _ms(),
            {
                "1x1_spread": _score(0.030, 3.0),
                "vanilla": _score(0.015, 1.5),
            },
            {
                "1x1_spread": [_variant(0.010)],
                "vanilla": [_variant(0.016)],
            },
        )

        assert isinstance(comparison, PairwiseComparison)
        assert comparison.verdict == "chosen_preferred"
        assert any(r.code == "scenario_fit.better_weighted_pnl" for r in comparison.chosen_edges)
        assert any(r.code == "selection_fit.target_supports_spread" for r in comparison.chosen_edges)
        assert any(r.code == "selection_fit.vanilla_preserves_upside" for r in comparison.challenger_edges)
        assert any(r.code == "risk.capped_upside" for r in comparison.caveats)
        assert any(
            r.code == "counterfactual.vanilla_if_uncapped_upside_dominates"
            for r in comparison.counterfactuals
        )

    def test_small_score_gap_produces_close_call(self):
        chosen = _item("1x1_spread", 1, "1x1 Spread")
        challenger = _item("european_digital", 2, "Digital")
        comparison = compare_structures(
            chosen,
            challenger,
            _ms(),
            {
                "1x1_spread": _score(0.0200, 2.0),
                "european_digital": _score(0.0188, 1.88),
            },
            {
                "1x1_spread": [_variant(0.011)],
                "european_digital": [_variant(0.007)],
            },
        )
        assert comparison.verdict == "close_call"
        assert comparison.confidence == "close"

    def test_cost_preference_promotes_cheaper_upfront(self):
        chosen = _item("1x1_spread", 1, "1x1 Spread")
        challenger = _item("european_digital", 2, "Digital")
        comparison = compare_structures(
            chosen,
            challenger,
            _ms(),
            {
                "1x1_spread": _score(0.028, 2.8),
                "european_digital": _score(0.020, 2.0),
            },
            {
                "1x1_spread": [_variant(0.020)],
                "european_digital": [_variant(0.006)],
            },
            PMPreferences(primary_objective="Keep cost low"),
        )
        cheaper = next(r for r in comparison.challenger_edges if r.code == "premium.cheaper_upfront")
        assert cheaper.materiality == "high"


class TestRecommendationPackMVP:
    def test_build_pack_populates_summary_risk_and_two_comparisons(self):
        chosen = _item("1x1_spread", 1, "1x1 Spread")
        vanilla = _item("vanilla", 2, "Vanilla")
        digital = _item("european_digital", 3, "Digital")

        pack = build_recommendation_pack(
            _ms(),
            _selection_result(chosen, vanilla, digital),
            {
                "1x1_spread": [_variant(0.010)],
                "vanilla": [_variant(0.016)],
                "european_digital": [_variant(0.007)],
            },
            {
                "1x1_spread": _score(0.030, 3.0),
                "vanilla": _score(0.015, 1.5),
                "european_digital": _score(0.020, 2.0),
            },
        )

        assert pack.chosen_id == "1x1_spread"
        assert pack.construction_reasons == []
        assert any(r.code == "scenario_fit.better_weighted_pnl" for r in pack.summary_reasons)
        assert any(r.code == "selection_fit.target_supports_spread" for r in pack.summary_reasons)
        assert any(r.code == "risk.capped_upside" for r in pack.risk_reasons)
        assert set(pack.comparisons) == {"vanilla", "european_digital"}
        assert any(item.challenger_id == "european_digital_rko" for item in pack.unavailable_comparisons)

    def test_pack_omits_target_support_reason_when_no_target(self):
        chosen = _item("1x1_spread", 1, "1x1 Spread")
        vanilla = _item("vanilla", 2, "Vanilla")
        pack = build_recommendation_pack(
            _ms(target_z=None),
            _selection_result(chosen, vanilla),
            {
                "1x1_spread": [_variant(0.010)],
                "vanilla": [_variant(0.016)],
            },
            {
                "1x1_spread": _score(0.030),
                "vanilla": _score(0.015),
            },
        )
        assert all(r.code != "selection_fit.target_supports_spread" for r in pack.summary_reasons)

    def test_build_pack_chooses_best_scenario_score_not_affinity_rank(self):
        affinity_winner = _item("1x1_spread", 1, "1x1 Spread")
        scenario_winner = _item("vanilla", 2, "Vanilla")

        pack = build_recommendation_pack(
            _ms(),
            _selection_result(affinity_winner, scenario_winner),
            {
                "1x1_spread": [_variant(0.010)],
                "vanilla": [_variant(0.016)],
            },
            {
                "1x1_spread": _score(0.010),
                "vanilla": _score(0.030),
            },
        )

        assert pack.chosen_id == "vanilla"
        assert pack.chosen_display_name == "Vanilla"
        assert pack.recommendation_basis == "scenario_weighted_pnl"
        assert pack.ranked_variants == []

    def test_unavailable_key_challengers_capture_not_shortlisted(self):
        chosen = _item("1x1_spread", 1, "1x1 Spread")
        pack = build_recommendation_pack(
            _ms(),
            _selection_result(chosen),
            {"1x1_spread": [_variant(0.010)]},
            {"1x1_spread": _score(0.030)},
        )

        unavailable = {item.challenger_id: item for item in pack.unavailable_comparisons}

        assert isinstance(unavailable["vanilla"], UnavailableComparison)
        assert unavailable["vanilla"].reason == "not_shortlisted"
        assert unavailable["european_digital"].reason == "not_shortlisted"
        assert unavailable["european_digital_rko"].reason == "not_shortlisted"

    def test_unavailable_key_challengers_capture_not_priceable(self):
        chosen = _item("1x1_spread", 1, "1x1 Spread")
        digital = _item("european_digital", 2, "European Digital")
        pack = build_recommendation_pack(
            _ms(),
            _selection_result(chosen, digital),
            {"1x1_spread": [_variant(0.010)]},
            {"1x1_spread": _score(0.030)},
        )

        unavailable = {item.challenger_id: item for item in pack.unavailable_comparisons}

        assert unavailable["european_digital"].reason == "not_priceable"

    def test_pack_builds_up_to_five_comparisons_from_scenario_ranked_targets(self):
        chosen = _item("vanilla", 1, "Vanilla")
        items = [
            chosen,
            _item("1x1_spread", 2, "1x1 Spread"),
            _item("european_digital", 3, "European Digital"),
            _item("european_digital_rko", 4, "European Digital with RKO"),
            _item("seagull", 5, "Seagull"),
            _item("1x2_spread", 6, "1x2 Spread"),
        ]
        variants = {item.structure_id: [_variant(0.010 + item.rank / 1000)] for item in items}
        scores = {
            "vanilla": _score(0.060),
            "1x1_spread": _score(0.050),
            "european_digital": _score(0.040),
            "european_digital_rko": _score(0.030),
            "seagull": _score(0.020),
            "1x2_spread": _score(0.010),
        }

        pack = build_recommendation_pack(
            _ms(),
            _selection_result(*items),
            variants,
            scores,
        )

        assert len(pack.comparisons) == 5
        assert set(pack.comparisons) == {
            "1x1_spread",
            "european_digital",
            "european_digital_rko",
            "seagull",
            "1x2_spread",
        }

    def test_rank_structures_by_scenario_score_preserves_affinity_rank_metadata(self):
        first = _item("1x1_spread", 1, "1x1 Spread")
        second = _item("vanilla", 2, "Vanilla")
        ranked = rank_structures_by_scenario_score(
            _selection_result(first, second),
            {
                "1x1_spread": _score(0.010, 1.0),
                "vanilla": _score(0.030, 3.0),
            },
            {
                "1x1_spread": _score(0.012, 1.2),
                "vanilla": _score(0.025, 2.5),
            },
        )

        assert [r.structure_id for r in ranked] == ["vanilla", "1x1_spread"]
        assert ranked[0].scenario_rank == 1
        assert ranked[0].affinity_rank == 2
        assert ranked[0].base_score_ccy == pytest.approx(2.5)
        assert ranked[0].pm_score_ccy == pytest.approx(3.0)

    def test_rank_variants_by_scenario_ccy_uses_specific_variant_labels(self):
        chosen = _item("vanilla", 1, "Vanilla")
        digital = _item("european_digital", 2, "European Digital")
        selector_result = _selection_result(chosen, digital)

        inputs = build_comparator_inputs(
            _ms(),
            selector_result,
            target=5.00,
            is_call=False,
            stop_price=5.35,
            loss_budget=2.0,
        )
        ranked = rank_variants_by_scenario_ccy(
            selector_result,
            inputs.variant_evaluations_by_structure,
        )

        assert ranked
        assert isinstance(ranked[0], RankedVariant)
        assert "(" in ranked[0].variant_label
        assert any(ch.isdigit() for ch in ranked[0].variant_label)


class TestComparatorInputBuilder:
    def test_build_comparator_inputs_uses_real_pricing_and_scenarios(self):
        chosen = _item("vanilla", 1, "Vanilla")
        digital = _item("european_digital", 2, "European Digital")
        selector_result = _selection_result(chosen, digital)

        inputs = build_comparator_inputs(
            _ms(),
            selector_result,
            target=5.00,
            is_call=False,
            stop_price=5.35,
            loss_budget=2.0,
        )

        assert isinstance(inputs, ComparatorInputs)
        assert set(inputs.priced_variants_by_structure) == {"vanilla", "european_digital"}
        assert set(inputs.scenario_rows_by_structure) == {"vanilla", "european_digital"}
        assert set(inputs.base_scores_by_structure) == {"vanilla", "european_digital"}
        assert set(inputs.pm_scores_by_structure) == {"vanilla", "european_digital"}
        assert isinstance(inputs.score_pairs_by_structure["vanilla"], StructureScorePair)
        assert inputs.score_pairs_by_structure["vanilla"].base is inputs.base_scores_by_structure["vanilla"]
        assert inputs.score_pairs_by_structure["vanilla"].pm is inputs.pm_scores_by_structure["vanilla"]
        assert isinstance(inputs.scenario_aggregates_by_structure["vanilla"], ScenarioAggregates)
        assert len(inputs.scenarios) == 20
        assert len(inputs.scenario_rows_by_structure["vanilla"]) == 20
        assert inputs.variant_evaluations_by_structure["vanilla"]
        assert inputs.variant_evaluations_by_structure["vanilla"][0].rows
        assert inputs.variant_evaluations_by_structure["vanilla"][0].aggregates.correct_path.avg_pnl_pct is not None

    def test_real_inputs_feed_existing_pack_builder(self):
        chosen = _item("vanilla", 1, "Vanilla")
        digital = _item("european_digital", 2, "European Digital")
        selector_result = _selection_result(chosen, digital)

        inputs = build_comparator_inputs(
            _ms(),
            selector_result,
            target=5.00,
            is_call=False,
            stop_price=5.35,
            loss_budget=2.0,
        )
        pack = build_recommendation_pack(
            _ms(),
            selector_result,
            inputs.priced_variants_by_structure,
            inputs.pm_scores_by_structure,
            variant_evaluations_by_structure=inputs.variant_evaluations_by_structure,
        )

        assert pack.ranked_variants
        assert pack.chosen_id == pack.ranked_variants[0].structure_id
        assert pack.chosen_variant_label == pack.ranked_variants[0].variant_label
        assert "european_digital" in pack.comparisons


class TestScenarioAggregates:
    def test_summarize_scenario_rows_groups_current_grid_cells(self):
        rows = [
            {"scenario_id": "25%T|F", "pnl_pct": -0.01, "price_pct": 0.01},
            {"scenario_id": "50%T|F", "pnl_pct": -0.02, "price_pct": 0.01},
            {"scenario_id": "25%T|t%→K", "pnl_pct": 0.03, "price_pct": 0.04},
            {"scenario_id": "50%T|t%→K", "pnl_pct": 0.05, "price_pct": 0.06},
            {"scenario_id": "Expiry|K", "pnl_pct": 0.20, "price_pct": 0.25},
            {"scenario_id": "25%T|−1σ", "pnl_pct": -0.04, "price_pct": 0.00},
            {"scenario_id": "50%T|−1σ", "pnl_pct": -0.06, "price_pct": 0.00},
            {"scenario_id": "Expiry|−1σ", "pnl_pct": -0.10, "price_pct": 0.00},
            {"scenario_id": "25%T|K+½σ", "pnl_pct": 0.07, "price_pct": 0.08},
            {"scenario_id": "50%T|K+½σ", "pnl_pct": 0.08, "price_pct": 0.09},
            {"scenario_id": "Expiry|K+½σ", "pnl_pct": 0.09, "price_pct": 0.10},
            {"scenario_id": "1w|Δvol", "pnl_pct": -0.005, "price_pct": 0.02},
            {"scenario_id": "25%T|Δvol", "pnl_pct": 0.01, "price_pct": 0.03},
            {"scenario_id": "50%T|Δvol", "pnl_pct": 0.02, "price_pct": 0.04},
        ]

        aggregates = summarize_scenario_rows(rows)

        assert aggregates.slow_path.avg_pnl_pct == pytest.approx(0.0125)
        assert aggregates.correct_path.avg_pnl_pct == pytest.approx((0.03 + 0.05 + 0.20) / 3)
        assert aggregates.wrong_way.worst_pnl_pct == pytest.approx(-0.10)
        assert aggregates.overshoot.best_pnl_pct == pytest.approx(0.09)
        assert aggregates.vol_sensitivity.worst_pnl_pct == pytest.approx(-0.005)
        assert aggregates.expiry_target_price_pct == pytest.approx(0.25)
        assert aggregates.expiry_target_pnl_pct == pytest.approx(0.20)

    def test_missing_aggregate_cells_return_none_values(self):
        aggregates = summarize_scenario_rows([])

        assert aggregates.slow_path.avg_pnl_pct is None
        assert aggregates.correct_path.worst_pnl_pct is None
        assert aggregates.expiry_target_price_pct is None
