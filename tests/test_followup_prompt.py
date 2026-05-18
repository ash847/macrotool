from types import SimpleNamespace

from conversation.context_builder import build_followup_prompt
from knowledge_engine.models import SizingOutput, StructureSelectionResult, StructureShortlistItem, TradeView


def _view() -> TradeView:
    return TradeView(
        pair="USDBRL",
        direction="base_lower",
        direction_conviction="high",
        horizon_days=90,
        magnitude_pct=4.0,
        mode="recommend",
    )


def _selector_result() -> StructureSelectionResult:
    return StructureSelectionResult(shortlist=[
        StructureShortlistItem(
            structure_id="vanilla",
            display_name="Vanilla",
            rank=1,
            rationale="test",
            rule_id="rule.test",
            sizing_modifier=None,
            caution=None,
            optimised_for="test",
        ),
        StructureShortlistItem(
            structure_id="1x1_spread",
            display_name="1x1 Spread",
            rank=2,
            rationale="test",
            rule_id="rule.test",
            sizing_modifier=None,
            caution=None,
            optimised_for="test",
        ),
    ], rules_fired=["rule.test"])


def _sizing() -> SizingOutput:
    return SizingOutput(
        kelly_fraction=0.25,
        kelly_conviction_used="high",
        kelly_source="default",
        vol_adjustment=1.0,
        adjusted_kelly=0.25,
        base_notional_usd=1_000_000,
        kelly_notional_usd=250_000,
        budget_type="from_budget",
        stop_level=5.45,
        stop_distance_pct=2.0,
        daily_range_est=0.05,
        tranche_schedule=None,
        tranche_count=None,
        take_profit_levels=[],
        notes=["Kelly-adjusted notional: $250,000"],
    )


def test_followup_prompt_includes_explanation_pack_when_present():
    prompt = build_followup_prompt(
        _view(),
        SimpleNamespace(spot=5.20),
        _selector_result(),
        _sizing(),
        explanation_context="RECOMMENDATION EXPLANATION PACK\nChosen: Vanilla - 25D (5.1000)",
    )

    assert "Use the recommendation explanation pack first" in prompt
    assert "[RECOMMENDATION EXPLANATION PACK]" in prompt
    assert "Chosen: Vanilla - 25D (5.1000)" in prompt


def test_followup_prompt_omits_explanation_pack_when_absent():
    prompt = build_followup_prompt(
        _view(),
        SimpleNamespace(spot=5.20),
        _selector_result(),
        _sizing(),
        explanation_context=None,
    )

    assert "[RECOMMENDATION EXPLANATION PACK]" not in prompt
    assert "Use the recommendation explanation pack first" not in prompt
