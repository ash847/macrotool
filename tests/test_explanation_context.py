"""
Tests for PM-safe comparator pack rendering.
"""

from types import SimpleNamespace

from analytics.market_state import MarketState
from conversation.explanation_context import render_explanation_pack
from knowledge_engine.comparator import (
    PMPreferences,
    build_comparator_inputs,
    build_recommendation_pack,
    make_reason,
)
from knowledge_engine.models import StructureSelectionResult, StructureShortlistItem


def _ms() -> MarketState:
    return MarketState(
        spot=5.20,
        fwd=5.25,
        vol=0.12,
        T=0.25,
        r_d=0.05,
        r_f=0.10,
        c=0.0,
        carry_regime=1,
        target_z=0.8,
        atmfsratio=1.2,
        put_call="put",
        with_carry=True,
    )


def _item(structure_id: str, rank: int, display_name: str) -> StructureShortlistItem:
    return StructureShortlistItem(
        structure_id=structure_id,
        display_name=display_name,
        rank=rank,
        rationale="test rationale",
        rule_id="rule.test",
        sizing_modifier=None,
        caution=None,
        optimised_for="test",
    )


def test_render_explanation_pack_includes_sections_and_disclosure():
    chosen = _item("1x1_spread", 1, "1x1 Spread")
    challenger = _item("vanilla", 2, "Vanilla")
    pack = build_recommendation_pack(
        _ms(),
        StructureSelectionResult(shortlist=[chosen, challenger], rules_fired=["test"]),
        {
            "1x1_spread": [SimpleNamespace(net_premium_pct=0.01)],
            "vanilla": [SimpleNamespace(net_premium_pct=0.02)],
        },
        {
            "1x1_spread": SimpleNamespace(score_pct=0.03),
            "vanilla": SimpleNamespace(score_pct=0.02),
        },
    )

    rendered = render_explanation_pack(pack)

    assert "RECOMMENDATION EXPLANATION PACK" in rendered
    assert "Chosen: 1x1 Spread" in rendered
    assert "Recommendation basis: scenario_weighted_pnl" in rendered
    assert "Scenario ranking:" in rendered
    assert "Summary:" in rendered
    assert "Comparison: 1x1_spread vs vanilla" in rendered
    assert "Disclosure:" in rendered
    assert "Do not reveal raw weights" in rendered


def test_render_explanation_pack_avoids_raw_internal_artifacts():
    chosen = _item("vanilla", 1, "Vanilla")
    digital = _item("european_digital", 2, "European Digital")
    selector_result = StructureSelectionResult(shortlist=[chosen, digital], rules_fired=["test"])
    inputs = build_comparator_inputs(
        _ms(),
        selector_result,
        target=5.00,
        is_call=False,
        stop_price=5.35,
        loss_budget=2.0,
        preferences=PMPreferences(),
    )
    pack = build_recommendation_pack(
        _ms(),
        selector_result,
        inputs.priced_variants_by_structure,
        inputs.pm_scores_by_structure,
    )

    rendered = render_explanation_pack(pack)

    assert "REASON_CATALOG" not in rendered
    assert "scenario_definitions" not in rendered
    assert "target_z_abs" not in rendered
    assert "multipliers" not in rendered


def test_high_materiality_reason_is_marked_without_numeric_score():
    reason = make_reason(
        "premium.cheaper_upfront",
        polarity="chosen_edge",
        materiality="high",
    )
    chosen = _item("vanilla", 1, "Vanilla")
    pack = build_recommendation_pack(
        _ms(),
        StructureSelectionResult(shortlist=[chosen], rules_fired=["test"]),
        {"vanilla": [SimpleNamespace(net_premium_pct=0.01)]},
        {"vanilla": SimpleNamespace(score_pct=0.03)},
    )
    pack.summary_reasons.append(reason)

    rendered = render_explanation_pack(pack)

    assert "Lower upfront premium [high]" in rendered
    assert "0.03" not in rendered
