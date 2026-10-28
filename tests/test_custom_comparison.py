"""Custom scenario evidence, attribution, context guards and agent routing."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from agentic.scenario_comparison import ScenarioEvidence, compare_evidence, public_attribution, validate_evidence
from agentic.session import AgentSession
from agentic.shortlist import shortlist_reference
from agentic.standard_pack import build_pack
from agentic.tools import dispatch
from config.loader import load_config
from knowledge_engine.models import TradeView
from knowledge_engine.scenario_scorer import score_structure
from tests.test_product_model_parity import _frozen_snapshot


def _session(direction="base_higher", kelly=False, magnitude=12):
    snapshot = _frozen_snapshot()
    session = AgentSession(snapshot=snapshot, cfg=load_config(), linear_notional=10000,
                           sizing_method="kelly" if kelly else "fixed_loss")
    view = TradeView(pair="USDBRL", direction=direction, horizon_days=90,
                     magnitude_pct=magnitude, direction_conviction="medium", mode="recommend")
    kwargs = {}
    if kelly:
        from tests._curves import stated_lognormal
        from analytics.distributions import interpolate_atm_vol
        from pricing.forwards import rate_context_for_snapshot
        ccy = snapshot.get("USDBRL")
        forward = rate_context_for_snapshot(ccy, 90/365).forward
        probs, bins = stated_lognormal(forward * (1.08 if direction == "base_higher" else 0.92),
                                       interpolate_atm_vol(ccy, 90), 90/365)
        kwargs = dict(kelly_probs=probs, kelly_bins=bins)
    session.view = view
    session.pack = build_pack(view, snapshot.get(view.pair), session.cfg, linear_notional=10000,
                              sizing_method=session.sizing_method, **kwargs)
    return session


def _price(session, request="1x1.3 ATMF/25d"):
    text, error = dispatch(session, "price_structure", {"request": request})
    assert not error, text
    assert "CUSTOM TRADE INDEX" in text
    return session.priced[-1]


def _args(session, ranks=None):
    return dict(shortlist_ref=shortlist_reference(session.pack, session.view),
                custom_index=len(session.priced), ranks=ranks or [session.pack.recommended[0].rank])


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
@pytest.mark.parametrize("kelly", [False, True])
def test_custom_retains_exact_grid_weights_and_comparison_never_reprices(direction, kelly, monkeypatch):
    session = _session(direction, kelly)
    pack = session.pack
    before = [(rec.rank, rec.score_pct, rec.score_ccy, rec.variant.structure_notional) for rec in pack.recommended]
    grid = deepcopy(pack.scenario_grid)

    def fail(*args, **kwargs):
        raise AssertionError("Must use retained evidence, not reconstruct")

    monkeypatch.setattr("analytics.scenario_generator.generate_scenarios", fail)
    custom = _price(session)
    assert custom.scenario_error is None
    evidence = custom.scenario_evidence
    assert [row["scenario_id"] for row in evidence.rows] == [sc["id"] for sc in grid]
    assert evidence.score == score_structure(evidence.rows, pack.scenario_weights)
    assert evidence.score.score_pct == pytest.approx(sum(cell.contrib_pct for cell in evidence.score.cells))
    assert evidence.score.score_ccy == pytest.approx(evidence.score.score_pct * custom.variant.structure_notional)
    for rec in pack.recommended:
        validate_evidence(rec.scenario_evidence)
        assert rec.scenario_evidence.score.score_pct == rec.score_pct
        assert rec.scenario_evidence.score.score_ccy == rec.score_ccy
    monkeypatch.setattr("analytics.scenario_pricer.price_scenarios", fail)
    monkeypatch.setattr("agentic.tools.price_structure", fail)
    ranks = [rec.rank for rec in pack.recommended[:3]]
    text, error = dispatch(session, "compare_custom_trade", _args(session, ranks))
    assert not error, text
    data = json.loads(text)
    assert [item["rank"] for item in data["comparisons"]] == ranks
    assert data["custom_notional_base"] == custom.variant.structure_notional
    assert '"score_pct"' not in text and '"normalized_weight"' not in text
    assert pack.scenario_grid == grid
    assert before == [(rec.rank, rec.score_pct, rec.score_ccy, rec.variant.structure_notional) for rec in pack.recommended]


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
def test_same_vanilla_as_recommendation_has_equal_evidence(direction):
    session = _session(direction)
    rec = next(rec for rec in session.pack.recommended if rec.structure_id == "vanilla")
    custom = _price(session, f"vanilla {rec.variant.strikes[0]:.17g}")
    assert custom.scenario_evidence.score.score_pct == pytest.approx(rec.score_pct, abs=1e-12)
    assert custom.variant.structure_notional == pytest.approx(rec.variant.structure_notional)
    text, error = dispatch(session, "compare_custom_trade", _args(session, [rec.rank]))
    assert not error, text
    assert json.loads(text)["comparisons"][0]["custom_minus_recommendation"]["per_unit_performance"] == "equal"


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
def test_custom_seagull_uses_same_stop_and_sizing_as_pack(direction):
    from agentic.price_structure import price_structure
    session = _session(direction)
    custom = _price(session, "seagull ATMF/25d/25d")
    pack, ms = session.pack, session.pack.market_state
    distance = pack.loss_budget / pack.linear_notional
    expected = price_structure("seagull ATMF/25d/25d", ms, is_call=pack.is_call, target=pack.target,
                               stop_price=ms.fwd*(1-distance if pack.is_call else 1+distance),
                               loss_budget=pack.loss_budget, linear_notional=pack.linear_notional)
    assert custom.variant.max_loss_pct == pytest.approx(expected.variant.max_loss_pct)
    assert custom.variant.structure_notional == pytest.approx(expected.variant.structure_notional)
    assert custom.scenario_evidence is not None


def _evidence(pnls, n=100):
    rows = tuple(dict(scenario_id=str(i), row="Expiry", col=str(i), pnl_pct=pnl, pnl_ccy=pnl*n)
                 for i, pnl in enumerate(pnls))
    return ScenarioEvidence(rows, score_structure(rows, {str(i): 1 for i in range(len(pnls))}))


def test_attribution_denominator_and_size_reversal():
    custom, reference = _evidence([0.4, -0.1, 0.3, -0.2], 10), _evidence([0.1]*4, 100)
    result = compare_evidence(custom, reference, 10, 100)
    assert result["per_unit_performance"] == "equal"
    assert result["actual_sized_performance"] == "lower"
    assert result["size_effect_using_custom_per_unit_performance"] == "lower"
    shares = result["per_unit_gap_attribution"]
    assert shares["contributors"][0]["signed_share_of_total_absolute_contribution"] == pytest.approx(0.3)
    assert shares["detractors"][0]["signed_share_of_total_absolute_contribution"] == pytest.approx(-0.3)
    stronger = _evidence([0.2]*4, 10)
    reversal = compare_evidence(stronger, reference, 10, 100)
    assert reversal["per_unit_performance"] == "higher"
    assert reversal["actual_sized_performance"] == "lower"
    assert reversal["structure_effect_at_comparator_notional"] == "higher"
    # Showing only top three must not renormalize to those three.
    drivers = public_attribution(_evidence([1]*8))["contributors"]
    assert len(drivers) == 3
    assert sum(row["signed_share_of_total_absolute_contribution"] for row in drivers) == pytest.approx(3/8)


def test_zero_negative_and_missing_sizes():
    negative = _evidence([-0.2, -0.1])
    zero = _evidence([0, 0])
    assert public_attribution(zero) == {"contributors": [], "detractors": []}
    assert compare_evidence(negative, zero, None, 1)["actual_sized_performance"] == "unavailable"
    assert compare_evidence(negative, zero, 0, 0)["actual_sized_performance"] == "equal"
    assert public_attribution(_evidence([1e-16]))["contributors"][0]["signed_share_of_total_absolute_contribution"] is None
    with pytest.raises(ValueError, match="non-negative"):
        compare_evidence(negative, zero, float("nan"), 1)


def test_invalid_evidence_is_not_a_zero_score():
    evidence = _evidence([1, 2])
    with pytest.raises(ValueError, match="unavailable"):
        validate_evidence(None)
    with pytest.raises(ValueError, match="non-finite"):
        validate_evidence(_evidence([float("nan")]))
    with pytest.raises(ValueError, match="pricing failed"):
        validate_evidence(replace(evidence, rows=(dict(evidence.rows[0], pricing_error="ValueError"), evidence.rows[1])))
    changed = replace(evidence, score=score_structure(evidence.rows, {"0": 1, "1": 2}))
    with pytest.raises(ValueError, match="same scenario"):
        compare_evidence(evidence, changed, 1, 1)


def test_context_and_argument_guards():
    session = _session()
    assert dispatch(session, "compare_custom_trade", _args(session))[1]
    _price(session)
    args = _args(session)
    for overrides in ({"shortlist_ref": "old"}, {"custom_index": True}, {"custom_index": 99},
                      {"ranks": [1, 1]}, {"ranks": [True]}, {"ranks": [999]}, {"ranks": []}, {"score": 100}):
        assert dispatch(session, "compare_custom_trade", dict(args, **overrides))[1]
    session.pack.scenario_weights = dict(session.pack.scenario_weights)
    first = next(iter(session.pack.scenario_weights))
    session.pack.scenario_weights[first] += 1
    # Even using the new reference cannot relabel old custom evidence.
    assert dispatch(session, "compare_custom_trade", _args(session))[1]


def test_no_target_and_scenario_failure_are_explicit(monkeypatch):
    session = _session(magnitude=None)
    custom = _price(session, "vanilla 25d")
    assert custom.scenario_evidence is None
    assert "usable target" in custom.scenario_error
    assert dispatch(session, "compare_custom_trade", _args(session))[1]
    session = _session()
    def fail(*args, **kwargs):
        raise ValueError("broken scenario")
    monkeypatch.setattr("analytics.scenario_pricer._value_variant", fail)
    custom = _price(session)
    assert custom.scenario_evidence is None
    assert "unavailable" in custom.scenario_error
    assert dispatch(session, "compare_custom_trade", _args(session))[1]


def test_agent_can_explain_retained_comparison_without_changing_rankings():
    from agentic.agent_flow import AgentFlow
    from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
    session = _session()
    _price(session)
    llm = FakeToolLLM(script=[
        LLMTurn(text="", tool_calls=[ToolCall("compare", "compare_custom_trade", _args(session))], stop_reason="tool_use"),
        LLMTurn(text="The scenario differences explain this trade-off.", tool_calls=[], stop_reason="end_turn"),
    ])
    result = AgentFlow(llm, session).advance("Compare my custom trade to recommendation 1 and explain the drivers")
    assert result == "The scenario differences explain this trade-off."
    assert any("per_unit_gap_attribution" in str(message) for message in session.messages)


def test_restricted_custom_trade_is_inspection_only_even_when_compared():
    session = _session()
    session.tail_constraint = "higher_spot"
    session.pack = build_pack(session.view, session.snapshot.get("USDBRL"), session.cfg,
                              linear_notional=10000, tail_constraint="higher_spot")
    original_ranks = [(rec.rank, rec.structure_id) for rec in session.pack.recommended]
    _price(session)
    text, error = dispatch(session, "compare_custom_trade", _args(session))
    assert not error, text
    assert "higher-spot tail" in json.loads(text)["restriction_warning"]
    assert [(rec.rank, rec.structure_id) for rec in session.pack.recommended] == original_ranks


@pytest.mark.parametrize("request_text", ["digital 10%", "erko ATMF/25d"])
def test_supported_exotic_scenario_pricers_are_not_replaced_with_vanillas(request_text):
    session = _session()
    custom = _price(session, request_text)
    assert custom.variant.resolved_contract is None
    assert custom.scenario_evidence is not None
    assert custom.scenario_error is None
    assert not dispatch(session, "compare_custom_trade", _args(session))[1]
