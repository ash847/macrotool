from dataclasses import replace

import pytest

from agentic.agent_flow import AgentFlow, build_system_prompt
from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
from agentic.session import AgentSession
from agentic.shortlist import _money, render_shortlist, shortlist_reference
from agentic.standard_pack import build_pack
from agentic.tools import dispatch
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.loader import load_agent_vocabulary
from knowledge_engine.models import TradeView
from knowledge_engine.payoff_profile import payoff_profile


@pytest.fixture(scope="module", params=["base_higher", "base_lower"])
def context(request):
    snapshot = load_snapshot()
    config = load_config()
    view = TradeView(pair="USDBRL", direction=request.param, direction_conviction="medium",
                     horizon_days=90, magnitude_pct=6.0, mode="recommend")
    pack = build_pack(view, snapshot.get(view.pair), config, linear_notional=1_000_000)
    return AgentSession(snapshot=snapshot, cfg=config, view=view, pack=pack)


@pytest.mark.parametrize("amount, expected", [
    (1_000, "1,000.00"), (1_000_000, "1,000,000.00"),
    (1_000_000_000, "1,000,000,000.00"), (-1_000, "-1,000.00"),
    (-0.0, "0.00"), (0.001, "<0.01"),
])
@pytest.mark.parametrize("currency", ["USD", "EUR", "GBP"])
def test_money_retains_units_and_currency(amount, expected, currency):
    assert _money(amount, currency) == f"{expected} {currency}"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_money_is_not_zero(value):
    assert _money(value, "USD") == "Unavailable"


def test_approved_vocabulary_is_used_by_prompt_and_renderer(context):
    vocabulary = load_agent_vocabulary()
    prompt = build_system_prompt([context.view.pair])
    for rule in vocabulary["narration_rules"]:
        assert rule in prompt
    table = render_shortlist(context.pack, context.view)
    assert vocabulary["shortlist_scope"] in table
    assert vocabulary["premium_risk_note"] in table


@pytest.mark.parametrize("is_call,strike,barrier,first,second", [
    (True, 5.8, 6.2, "5.8000 strike", "6.2000 knock-out"),
    (False, 4.21, 4.0283, "4.0283 knock-out", "4.2100 strike"),
])
def test_barrier_range_is_ascending_without_swapping_roles(is_call, strike, barrier, first, second):
    profile = payoff_profile("european_rko", [], net_premium_pct=0.01,
                             is_zero_cost=False, is_call=is_call, strikes=[strike], barrier=barrier)
    assert profile.value_region.startswith(f"between the {first} and the {second}")
    assert profile.product_nature == "expiry_only"


@pytest.mark.parametrize("premium,flow", [(0.01, "Pay"), (-0.01, "Receive"), (0.0, "Zero")])
def test_premium_flow_target_ratio_and_zero_allocation(context, premium, flow):
    original = context.pack.recommended[0]
    economics = replace(original.variant.economics,
                        target_return_on_premium=2.0 if premium > 0 else None,
                        ratio_status="available" if premium > 0 else "not_applicable")
    variant = replace(original.variant, structure_notional=0.0, net_premium_ccy=0.0,
                      net_premium_pct=premium, economics=economics)
    pack = replace(context.pack, recommended=[replace(original, variant=variant)])
    table = render_shortlist(pack, context.view)
    assert f"| {flow}" in table
    assert "0.00 USD" in table
    assert ("2.00×" if premium > 0 else "N/A — no premium outlay") in table


def test_comparison_followed_by_sizing_uses_same_pack(context, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Follow-up must not rebuild or reprice")
    monkeypatch.setattr("agentic.tools.build_pack", fail)
    monkeypatch.setattr("agentic.tools.price_structure", fail)
    reference = shortlist_reference(context.pack, context.view)
    llm = FakeToolLLM(script=[
        LLMTurn(text="", tool_calls=[ToolCall("compare", "inspect_recommendations",
            {"shortlist_ref": reference, "ranks": [1, 3]})], stop_reason="tool_use"),
        LLMTurn(text="Comparison from the retained trades.", tool_calls=[], stop_reason="end_turn"),
        LLMTurn(text="", tool_calls=[ToolCall("size", "inspect_recommendations",
            {"shortlist_ref": reference, "ranks": [2]})], stop_reason="tool_use"),
        LLMTurn(text="Sizing follows the recorded audit.", tool_calls=[], stop_reason="end_turn"),
    ])
    session = AgentSession(snapshot=context.snapshot, cfg=context.cfg, view=context.view, pack=context.pack)
    flow = AgentFlow(llm, session)
    comparison = flow.advance("Compare 1 and 3")
    assert "| 1 |" in comparison and "| 3 |" in comparison and "| 2 |" not in comparison
    assert flow.advance("Why this size for 2?") == "Sizing follows the recorded audit."
    assert session.pack is context.pack


def test_changed_market_or_weights_reject_old_rank_reference(context):
    reference = shortlist_reference(context.pack, context.view)
    changed_market = replace(context.pack.market_state, spot=context.pack.market_state.spot * 1.01)
    for changed in (replace(context.pack, market_state=changed_market),
                    replace(context.pack, scenario_weights={"changed": 1.0})):
        session = AgentSession(snapshot=context.snapshot, cfg=context.cfg, view=context.view, pack=changed)
        text, error = dispatch(session, "inspect_recommendations", {"shortlist_ref": reference, "ranks": [1]})
        assert error and "no longer current" in text
