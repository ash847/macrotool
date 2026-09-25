from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentic.agent_flow import AgentFlow, SYSTEM_PROMPT
from agentic.agent_llm import AnthropicToolLLM, FakeToolLLM, LLMTurn, ToolCall
from agentic.render import render_pack
from agentic.session import AgentSession
from agentic.shortlist import present_shortlist, render_shortlist, shortlist_reference
from agentic.standard_pack import build_pack
from agentic.tools import dispatch
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView


@pytest.fixture(scope="module")
def context():
    snapshot = load_snapshot()
    cfg = load_config()
    view = TradeView(pair="USDBRL", direction="base_higher", direction_conviction="medium",
                     horizon_days=90, magnitude_pct=6.0, mode="recommend")
    pack = build_pack(view, snapshot.get("USDBRL"), cfg, linear_notional=1_000_000)
    assert len(pack.recommended) >= 3
    return snapshot, cfg, view, pack


def test_top_five_has_agreed_columns_and_exact_sized_values(context):
    _, _, view, pack = context
    table = render_shortlist(pack, view)
    rows = [line for line in table.splitlines() if line.startswith("| ")][2:]
    assert len(rows) == min(5, len(pack.recommended))
    for line, rec in zip(rows, pack.recommended):
        assert len(line.split("|")) == 9
        assert line.startswith(f"| {rec.rank} |")
        assert f"{rec.variant.structure_notional:,.2f} USD" in line
        assert "90d · expiry" in line
    assert "Target return on premium" in table
    assert "Additional loss beyond premium?" in table
    assert "score_ccy" not in table


def test_comparison_keeps_engine_ranks_and_prices(context):
    snapshot, cfg, view, pack = context
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack)
    reference = shortlist_reference(pack, view)
    content, error = dispatch(session, "inspect_recommendations", {"shortlist_ref": reference, "ranks": [3, 1]})
    assert not error
    assert "| 1 |" in content and "| 3 |" in content and "| 2 |" not in content
    assert "SIZING AUDIT" in content
    assert "risk (engine)" in content


def test_details_do_not_reprice_and_reject_stale_or_invalid_references(context, monkeypatch):
    snapshot, cfg, view, pack = context
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack)
    def fail(*args, **kwargs):
        pytest.fail("Lookup must not price or rebuild")
    monkeypatch.setattr("agentic.tools.build_pack", fail)
    monkeypatch.setattr("agentic.tools.price_structure", fail)
    monkeypatch.setattr("analytics.structure_pricer.price_variants", fail)
    reference = shortlist_reference(pack, view)
    text, error = dispatch(session, "inspect_recommendations", {"shortlist_ref": reference, "ranks": [2]})
    assert not error and "Already-priced" in text
    for ranks in ([999], [1, 1], [True], []):
        _, error = dispatch(session, "inspect_recommendations", {"shortlist_ref": reference, "ranks": ranks})
        assert error
    text, error = dispatch(session, "inspect_recommendations", {"shortlist_ref": "old-list", "ranks": [1]})
    assert error and "no longer current" in text
    assert session.pack is pack


def test_family_status_uses_retained_rank_without_inventing_reason(context):
    snapshot, cfg, view, pack = context
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack)
    for rec in pack.recommended:
        text, error = dispatch(session, "inspect_recommendations", {
            "shortlist_ref": shortlist_reference(pack, view), "family": rec.structure_id.replace("_", " "),
        })
        assert not error and f"Engine rank {rec.rank}" in text
    empty = replace(pack, recommended=[], selector_result=SimpleNamespace(shortlist=[]))
    session.pack = empty
    text, error = dispatch(session, "inspect_recommendations", {
        "shortlist_ref": shortlist_reference(empty, view), "family": "vanilla",
    })
    assert not error and "not recorded" in text


def test_missing_values_and_credit_are_not_fabricated(context):
    _, _, view, pack = context
    original = pack.recommended[0]
    economics = replace(original.variant.economics, target_return_on_premium=None,
                        ratio_status="not_applicable", ratio_reason="Not applicable — no premium outlay")
    variant = replace(original.variant, net_premium_pct=-0.01, net_premium_ccy=-100,
                      structure_notional=None, economics=economics, can_lose_beyond_premium=None)
    altered = replace(pack, recommended=[replace(original, variant=variant)])
    table = render_shortlist(altered, view)
    assert "Receive 100.00 USD" in table
    assert "N/A — no premium outlay" in table
    assert "Unknown" in table
    assert "Unavailable" in table


def test_first_reply_and_model_history_have_one_python_table(context):
    snapshot, cfg, view, pack = context
    session = AgentSession(snapshot=snapshot, cfg=cfg)
    session.store(view, pack)
    llm = FakeToolLLM(script=[
        LLMTurn(text="", tool_calls=[ToolCall("pack", "run_standard_pack", {
            "pair": view.pair, "direction": view.direction, "horizon_days": 90, "magnitude_pct": 6.0,
        })], stop_reason="tool_use"),
        LLMTurn(text="[[SHORTLIST]]\nThe top pick fits the stated view.", tool_calls=[], stop_reason="end_turn"),
        LLMTurn(text="A general answer.", tool_calls=[], stop_reason="end_turn"),
    ])
    flow = AgentFlow(llm, session)
    reply = flow.advance("What should I trade?")
    assert reply.count("| Rank |") == 1
    assert "[[SHORTLIST]]" not in reply
    assert session.messages[-1]["content"] == reply
    assert flow.advance("What is carry?") == "A general answer."
    assert any(message.get("content") == reply for message in llm.seen[-1]["messages"])


def test_model_authored_tables_are_not_displayed_as_engine_numbers(context):
    _, _, view, pack = context
    text = "| Trade | Notional |\n| --- | --- |\n| invented | 999999 |\n\nShort explanation."
    result = present_shortlist(text, pack, view, automatic=True)
    assert "invented" not in result
    assert result.count("| Rank |") == 1
    assert result.endswith("Short explanation.")


def test_reference_and_detail_routing_are_in_model_context(context):
    _, _, view, pack = context
    text = render_pack(pack, view)
    assert f"SHORTLIST REFERENCE: {shortlist_reference(pack, view)}" in text
    assert "one paragraph, at most 120 words" in SYSTEM_PROMPT
    assert "inspect_recommendations" in SYSTEM_PROMPT
    assert "do not retype" in text


def test_anthropic_final_text_message_preserves_displayed_response():
    adapter = AnthropicToolLLM.__new__(AnthropicToolLLM)
    result = adapter.format_text_reply("table and explanation")
    assert result == {"role": "assistant", "content": [{"type": "text", "text": "table and explanation"}]}
