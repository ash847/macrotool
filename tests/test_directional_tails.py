from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentic.agent_flow import AgentFlow
from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
from agentic.session import AgentSession
from agentic.standard_pack import build_pack, _filter_directional_tails
from agentic.shortlist import render_shortlist, shortlist_reference
from agentic.tools import dispatch
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.loader import load_directional_tail_risk
from knowledge_engine.models import TradeView
from knowledge_engine.tail_policy import (
    assign_variant_tails, construction_tails, resolved_tail_constraint, tail_exclusion_reason,
)
from analytics.structure_pricer import _load_variants
from workspace.settings import ChatSettings
from workspace.service import ConversationService, session_prefs
from workspace.store import InMemoryStore


@pytest.mark.parametrize("family,call,put", [
    ("vanilla", (False, False), (False, False)),
    ("1x1_spread", (False, False), (False, False)),
    ("1x1.5_spread", (False, True), (True, False)),
    ("1x2_spread", (False, True), (True, False)),
    ("1x2x1_spread", (False, False), (False, False)),
    ("seagull", (True, False), (False, True)),
    ("european_rko", (False, False), (False, False)),
    ("european_digital", (False, False), (False, False)),
    ("european_digital_rko", (False, False), (False, False)),
])
def test_all_catalog_constructions_have_directional_facts(family, call, put):
    for construction in _load_variants()[family]:
        assert construction_tails(family, construction, True) == call
        assert construction_tails(family, construction, False) == put


@pytest.mark.parametrize("direction,against,with_view", [
    ("base_higher", "lower_spot", "higher_spot"),
    ("base_lower", "higher_spot", "lower_spot"),
])
def test_relative_vs_absolute_constraints(direction, against, with_view):
    assert resolved_tail_constraint("against_view", direction) == against
    assert resolved_tail_constraint("with_view", direction) == with_view
    for absolute in ("none", "both", "lower_spot", "higher_spot"):
        assert resolved_tail_constraint(absolute, direction) == absolute


def test_unknown_custom_and_missing_config_fail_closed(monkeypatch):
    assert construction_tails("1x2_spread", {"long_delta": 0.34, "short_delta": 0.18}, True) == (None, None)
    variant = SimpleNamespace(lower_spot_tail=None, higher_spot_tail=False)
    assert "unknown" in tail_exclusion_reason(variant, "lower_spot", "base_higher")
    assert tail_exclusion_reason(variant, "higher_spot", "base_higher") is None
    assert tail_exclusion_reason(variant, "none", "base_higher") is None
    monkeypatch.setattr("knowledge_engine.tail_policy.load_directional_tail_risk", lambda: {"families": {}})
    assert construction_tails("vanilla", _load_variants()["vanilla"][0], True) == (None, None)


@pytest.mark.parametrize("is_call,expected", [(True, (True, False)), (False, (False, True))])
def test_linear_modelled_stop_does_not_protect_tail(is_call, expected):
    variant = SimpleNamespace(variant_label="Delta 1 (max-loss capped)")
    assign_variant_tails("linear", variant, is_call)
    assert (variant.lower_spot_tail, variant.higher_spot_tail) == expected


@pytest.fixture(scope="module")
def cnh_context():
    snapshot, cfg = load_snapshot(), load_config()
    session = AgentSession(snapshot=snapshot, cfg=cfg, linear_notional=100_000_000)
    _, error = dispatch(session, "run_standard_pack", {"pair": "USDCNH", "horizon_days": 365, "target_level": 7.20})
    assert not error and session.pack.variants_ranked
    return snapshot, cfg, session.view, session.pack


def session_for(context):
    snapshot, cfg, view, pack = context
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack, linear_notional=100_000_000)
    session.store(view, pack)
    return session


def test_cnh_lower_spot_excludes_seagulls_keeps_ratio_calls(cnh_context):
    session = session_for(cnh_context)
    original = session.pack
    content, error = dispatch(session, "set_tail_constraint", {"tail_constraint": "lower_spot"})
    assert not error and session.tail_constraint == "lower_spot"
    assert all(rec.variant.lower_spot_tail is False for rec in session.pack.recommended)
    assert session.pack.recommended
    assert not any(rec.structure_id in ("seagull", "linear") for rec in session.pack.recommended)
    assert any(item["structure_id"] == "seagull" for item in session.pack.tail_exclusions)
    assert "TAIL EXCLUSION" in content
    originals = {(rec.structure_id, rec.variant.variant_label): rec for rec in original.recommended}
    for rank, rec in enumerate(session.pack.recommended, 1):
        before = originals[(rec.structure_id, rec.variant.variant_label)]
        assert rec.rank == rank
        assert rec.score_ccy == before.score_ccy
        assert rec.variant.structure_notional == before.variant.structure_notional
        assert rec.variant.net_premium_pct == before.variant.net_premium_pct
    assert shortlist_reference(original, session.view) != shortlist_reference(session.pack, session.view)
    text, error = dispatch(session, "inspect_recommendations", {
        "shortlist_ref": shortlist_reference(session.pack, session.view), "family": "seagull", "display": "none",
    })
    assert not error and "lower-spot tail present" in text
    table = render_shortlist(session.pack, session.view)
    assert "Lower-spot tail:" not in table and "Higher-spot tail:" not in table
    details = render_shortlist(session.pack, session.view, ranks=[1])
    assert "Lower-spot tail: No" in details and "Higher-spot tail:" in details


def test_higher_spot_and_both_constraints(cnh_context):
    session = session_for(cnh_context)
    for constraint in ("higher_spot", "both"):
        _, error = dispatch(session, "set_tail_constraint", {"tail_constraint": constraint})
        assert not error
        assert all(rec.variant.higher_spot_tail is False for rec in session.pack.recommended)
        if constraint == "both":
            assert all(rec.variant.lower_spot_tail is False for rec in session.pack.recommended)
        else:
            assert any(rec.structure_id == "seagull" for rec in session.pack.recommended)


def test_constraint_cache_clear_and_omission(cnh_context):
    session = session_for(cnh_context)
    original = session.pack
    dispatch(session, "set_tail_constraint", {"tail_constraint": "lower_spot"})
    constrained = session.pack
    dispatch(session, "run_standard_pack", {"pair": "USDCNH", "horizon_days": 365, "target_level": 7.20})
    assert session.pack is constrained
    dispatch(session, "set_tail_constraint", {"tail_constraint": "none"})
    assert session.pack is original
    text, error = dispatch(session, "set_tail_constraint", {"tail_constraint": "invented"})
    assert error and session.pack is original and session.tail_constraint == "none"


def test_relative_preference_re_resolves_on_reversal(cnh_context):
    session = session_for(cnh_context)
    dispatch(session, "set_tail_constraint", {"tail_constraint": "against_view"})
    assert session.pack.resolved_tail_constraint == "lower_spot"
    dispatch(session, "run_standard_pack", {"pair": "USDCNH", "horizon_days": 365, "target_level": 6.0})
    assert session.tail_constraint == "against_view" and session.pack.resolved_tail_constraint == "higher_spot"
    assert all(rec.variant.higher_spot_tail is False for rec in session.pack.recommended)
    assert session.pack.recommended


def test_custom_conflict_and_unknown_are_flagged_without_changing_construction(cnh_context):
    session = session_for(cnh_context)
    dispatch(session, "set_tail_constraint", {"tail_constraint": "higher_spot"})
    for request in ("25 vs 10 1x2", "34 vs 18 1x2"):
        text, error = dispatch(session, "price_structure", {"request": request})
        assert not error and "CONFLICT WITH ACTIVE TAIL CONSTRAINT" in text
        assert "not an eligible recommendation" in text
        if request.startswith("34"):
            assert "unknown" in text


def test_no_target_fallback_obeys_filter(cnh_context):
    snapshot, cfg, _, _ = cnh_context
    view = TradeView(pair="USDCNH", direction="base_higher", horizon_days=365, direction_conviction="medium", mode="recommend")
    pack = build_pack(view, snapshot.get("USDCNH"), cfg, tail_constraint="both")
    assert not pack.variants_ranked
    assert all(rec.variant.lower_spot_tail is False and rec.variant.higher_spot_tail is False for rec in pack.recommended)


def test_all_excluded_does_not_bypass_filter_via_fallback(cnh_context, monkeypatch):
    snapshot, cfg, view, _ = cnh_context
    monkeypatch.setattr("knowledge_engine.tail_policy.load_directional_tail_risk", lambda: {"families": {}})
    pack = build_pack(view, snapshot.get("USDCNH"), cfg, tail_constraint="both")
    assert pack.variants_ranked and not pack.recommended and pack.tail_exclusions
    assert not pack.affinity_shortlist


def test_chat_tool_flow_and_saved_preferences(cnh_context):
    session = session_for(cnh_context)
    llm = FakeToolLLM(script=[
        LLMTurn("", [ToolCall("tails", "set_tail_constraint", {"tail_constraint": "lower_spot"})], "tool_use"),
        LLMTurn("[[MARKET_COMMENTARY]]Same market.\n[[TRADE_NOTES]]Lower-spot tails excluded.", [], "end_turn"),
    ])
    reply = AgentFlow(llm, session).advance("Exclude trades with tails on lower spot")
    assert "### Top structures" in reply
    assert "Lower-spot tail:" not in reply and "Higher-spot tail:" not in reply
    assert "Active tail constraint:" in reply
    svc = ConversationService(InMemoryStore(), "test@example.com")
    conv = svc.record_exchange(svc.new_conversation(), session, seq=0, prompt="exclude lower tails", reply=reply, pre_len=0)
    assert conv.settings["tail_constraint"] == "lower_spot"
    assert svc.active_version(conv).prefs["tail_constraint"] == "lower_spot"
    def make_session(saved):
        fresh = AgentSession(snapshot=session.snapshot, cfg=session.cfg, linear_notional=session.linear_notional)
        ChatSettings.from_dict(saved.settings).apply_to(fresh)
        return fresh
    resumed = svc.resume(conv.id, make_session)
    assert resumed.session.pack.resolved_tail_constraint == "lower_spot"
    assert all(rec.variant.lower_spot_tail is False for rec in resumed.session.pack.recommended)


def test_pre_view_preference_persists_and_invalid_view_is_transactional(cnh_context):
    snapshot, cfg, _, _ = cnh_context
    session = AgentSession(snapshot=snapshot, cfg=cfg)
    _, error = dispatch(session, "set_tail_constraint", {"tail_constraint": "against_view"})
    assert not error and session.tail_constraint == "against_view"
    _, error = dispatch(session, "run_standard_pack", {"pair": "FAKE", "horizon_days": 90, "tail_constraint": "none"})
    assert error and session.tail_constraint == "against_view"
    assert session_prefs(session)["tail_constraint"] == "against_view"
    assert ChatSettings.from_dict({}).tail_constraint == "none"


@pytest.mark.parametrize("is_call", [True, False])
def test_ratio_filter_allows_view_side_tail_but_not_against_view(cnh_context, is_call):
    from analytics.structure_pricer import price_variants

    market = cnh_context[3].market_state
    recommendations = []
    for family in ("1x2_spread", "1x1.5_spread", "1x2x1_spread", "seagull"):
        for variant in price_variants(market, family, target=cnh_context[3].target, is_call=is_call):
            recommendations.append(SimpleNamespace(structure_id=family, variant=variant))
    exclusions = []
    kept = _filter_directional_tails(recommendations, is_call, "against_view", exclusions)
    assert any(rec.structure_id == "1x2_spread" for rec in kept)
    assert any(rec.structure_id == "1x1.5_spread" for rec in kept)
    assert any(rec.structure_id == "1x2x1_spread" for rec in kept)
    assert not any(rec.structure_id == "seagull" for rec in kept)
    opposite = _filter_directional_tails(recommendations, is_call, "with_view", [])
    assert any(rec.structure_id == "seagull" for rec in opposite)
    assert not any(rec.structure_id in ("1x2_spread", "1x1.5_spread") for rec in opposite)
