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
    ConstructionReason,
    PMPreferences,
    PairwiseComparison,
    REASON_CATALOG,
    RecommendationExplanationPack,
    Reason,
    build_recommendation_pack,
    compare_structures,
    make_reason,
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


def _score(score_pct: float):
    return SimpleNamespace(score_pct=score_pct)


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
                "1x1_spread": _score(0.030),
                "vanilla": _score(0.015),
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
                "1x1_spread": _score(0.0200),
                "european_digital": _score(0.0188),
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
                "1x1_spread": _score(0.028),
                "european_digital": _score(0.020),
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
                "1x1_spread": _score(0.030),
                "vanilla": _score(0.015),
                "european_digital": _score(0.020),
            },
        )

        assert pack.chosen_id == "1x1_spread"
        assert pack.construction_reasons == []
        assert any(r.code == "scenario_fit.better_weighted_pnl" for r in pack.summary_reasons)
        assert any(r.code == "selection_fit.target_supports_spread" for r in pack.summary_reasons)
        assert any(r.code == "risk.capped_upside" for r in pack.risk_reasons)
        assert set(pack.comparisons) == {"vanilla", "european_digital"}

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
