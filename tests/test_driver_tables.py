from dataclasses import replace

import pytest

from agentic.agent_flow import AgentFlow
from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
from agentic.session import AgentSession
from agentic.shortlist import present_shortlist, render_driver_table, shortlist_reference
from agentic.tools import TOOL_SCHEMAS, dispatch
from knowledge_engine.scenario_scorer import CellBreakdown
from tests.test_shortlist_presentation import context


def run_inspection(context, display=None, ranks=None, text="Engine-backed explanation.", extra=None):
    snapshot, cfg, view, pack = context
    args = {"shortlist_ref": shortlist_reference(pack, view)}
    if display is not None:
        args["display"] = display
    if ranks is not None:
        args["ranks"] = ranks
    args.update(extra or {})
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack)
    llm = FakeToolLLM(script=[
        LLMTurn("", [ToolCall("inspect", "inspect_recommendations", args)], "tool_use"),
        LLMTurn(text, [], "end_turn"),
    ])
    return AgentFlow(llm, session).advance("Show the requested table"), session


@pytest.mark.parametrize("display,details,positive,negative", [
    ("trade_details", True, False, False), ("contributors", False, True, False),
    ("detractors", False, False, True), ("drivers", False, True, True),
    ("both", True, True, True), ("none", False, False, False),
])
def test_explicit_table_selection_and_persisted_reply(context, display, details, positive, negative, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Inspection must not rebuild or reprice")
    monkeypatch.setattr("agentic.tools.build_pack", fail)
    monkeypatch.setattr("agentic.tools.price_structure", fail)
    reply, session = run_inspection(context, display, [3, 1])
    assert ("Trade comparison" in reply) is details
    assert ("### Top contributors" in reply) is positive
    assert ("### Top detractors" in reply) is negative
    assert "Market state" not in reply
    assert "Rank 2 ·" not in reply
    assert session.messages[-1]["content"] == reply
    assert session.pack is context[3]


def test_reading_multiple_ranks_no_longer_forces_comparison(context):
    reply, _ = run_inspection(context, ranks=[1, 2])
    assert reply == "Engine-backed explanation."


def test_single_rank_drivers_and_default_top_five(context):
    single, _ = run_inspection(context, "drivers", [2])
    assert "Rank 2 ·" in single and "Rank 1 ·" not in single
    default, _ = run_inspection(context, "contributors")
    for rec in context[3].recommended[:5]:
        assert f"Rank {rec.rank} ·" in default
    assert "Rank 6 ·" not in default


def test_family_selects_exact_retained_variants(context):
    family = context[3].recommended[0].structure_id
    reply, _ = run_inspection(context, "contributors", extra={"family": family})
    for rec in context[3].recommended:
        assert (f"Rank {rec.rank} ·" in reply) is (rec.structure_id == family)


@pytest.mark.parametrize("extra", [
    {"shortlist_ref": "stale"}, {"ranks": [999]}, {"ranks": [True]},
    {"ranks": [1, 1]}, {"ranks": []}, {"family": "vanilla", "ranks": [1]},
    {"display": "invented"}, {"display": ["drivers"]},
])
def test_invalid_requests_never_render_current_tables(context, extra):
    reply, session = run_inspection(context, "drivers", text="[[SHORTLIST]]\nPlease clarify.", extra=extra)
    assert reply == "Please clarify."
    results = session.messages[-2]["content"]
    assert results[0]["is_error"]


def test_numerical_tables_from_model_are_replaced_not_trusted(context):
    text = ("| Scenario | Share |\n| --- | --- |\n| Fabricated | 999% |\n\n"
            "<table><tr><td>Also fabricated 888%</td></tr></table>\n\nExplanation.")
    reply, _ = run_inspection(context, "drivers", [1], text=text)
    assert "999" not in reply and "888" not in reply
    assert "Trade comparison" not in reply
    assert "### Top contributors" in reply and "### Top detractors" in reply
    assert reply.endswith("Explanation.")


def test_exact_signed_shares_and_na_use_retained_full_denominator(context):
    _, _, view, pack = context
    positive = CellBreakdown("positive", "Expiry", "K", 0.006, None, 1, 1, 0.006, None)
    negative = replace(positive, scenario_id="negative", col="−1σ", contrib_pct=-0.005)
    rec = replace(pack.recommended[0], cell_drivers=([positive], [negative]),
                  absolute_contribution_total_pct=0.011)
    altered = replace(pack, recommended=[rec])
    assert "| Target hit · Expiry | +54.5% |" in render_driver_table(altered, view, [rec.rank], "contributors")
    assert "| Full reversal · Expiry | -45.5% |" in render_driver_table(altered, view, [rec.rank], "detractors")
    for total in (None, 0, 0.0001, float("nan")):
        unavailable = replace(altered, recommended=[replace(rec, absolute_contribution_total_pct=total)])
        assert "| Target hit · Expiry | N/A |" in render_driver_table(unavailable, view, [rec.rank], "contributors")


def test_missing_and_empty_driver_data_are_distinguished(context):
    _, _, view, pack = context
    for drivers, expected in [(None, "data unavailable"), (([], []), "No positive")]:
        rec = replace(pack.recommended[0], cell_drivers=drivers)
        table = render_driver_table(replace(pack, recommended=[rec]), view, [rec.rank], "contributors")
        assert expected in table and "| Scenario |" not in table


def test_repeated_calls_merge_without_duplicate_tables_and_next_turn_resets(context):
    snapshot, cfg, view, pack = context
    reference = shortlist_reference(pack, view)
    calls = [ToolCall(str(index), "inspect_recommendations", {
        "shortlist_ref": reference, "ranks": ranks, "display": display,
    }) for index, (ranks, display) in enumerate([([1], "contributors"), ([1, 2], "contributors"), ([3], "trade_details")])]
    llm = FakeToolLLM(script=[LLMTurn("", calls, "tool_use"), LLMTurn("Commentary.", [], "end_turn"),
                            LLMTurn("General answer.", [], "end_turn")])
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack)
    flow = AgentFlow(llm, session)
    reply = flow.advance("Contributors 1 and 2; details 3")
    assert reply.count("### Top contributors") == 1
    assert reply.count("**Rank 1 ·") == 1 and reply.count("**Rank 2 ·") == 1
    assert "| 3 |" in reply and "**Rank 3 ·" not in reply
    assert flow.advance("What is carry?") == "General answer."


def test_round_limit_still_renders_and_saves_engine_tables(context):
    snapshot, cfg, view, pack = context
    llm = FakeToolLLM(script=[LLMTurn("Untrusted partial table", [ToolCall("inspect", "inspect_recommendations", {
        "shortlist_ref": shortlist_reference(pack, view), "ranks": [1], "display": "drivers",
    })], "tool_use")])
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack)
    reply = AgentFlow(llm, session, max_rounds=1).advance("Show drivers")
    assert "Untrusted" not in reply and "### Top contributors" in reply
    assert session.messages[-1]["content"] == reply


def test_schema_exposes_only_supported_display_choices():
    schema = next(tool for tool in TOOL_SCHEMAS if tool["name"] == "inspect_recommendations")
    assert "display" in schema["input_schema"]["required"]
    assert set(schema["input_schema"]["properties"]["display"]["enum"]) == {
        "none", "trade_details", "contributors", "detractors", "drivers", "both", "dashboard",
    }


def test_new_view_clears_previous_driver_selection(context, monkeypatch):
    from agentic.tools import dispatch as real_dispatch

    snapshot, cfg, view, pack = context
    changed = replace(pack, market_state=replace(pack.market_state, spot=pack.market_state.spot * 1.01))

    def change_view(session, name, args):
        if name == "run_standard_pack":
            session.pack = changed
            return "New view established.", False
        return real_dispatch(session, name, args)

    monkeypatch.setattr("agentic.agent_flow.dispatch", change_view)
    llm = FakeToolLLM(script=[
        LLMTurn("", [ToolCall("old", "inspect_recommendations", {
            "shortlist_ref": shortlist_reference(pack, view), "display": "drivers", "ranks": [1],
        }), ToolCall("new", "run_standard_pack", {})], "tool_use"),
        LLMTurn("[[MARKET_COMMENTARY]]New market.\n[[TRADE_NOTES]]New risks.", [], "end_turn"),
    ])
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack)
    reply = AgentFlow(llm, session).advance("Switch view")
    assert "### Top structures" in reply and "### Top contributors" not in reply
    assert f"{changed.market_state.spot:.4f}" in reply


def test_absent_family_has_no_fabricated_driver_rows(context):
    from types import SimpleNamespace

    snapshot, cfg, view, pack = context
    empty = replace(pack, recommended=[], selector_result=SimpleNamespace(shortlist=[]))
    reply, _ = run_inspection((snapshot, cfg, view, empty), "drivers", extra={"family": "vanilla"},
                              text="No retained priced variants for this family.")
    assert reply == "No retained priced variants for this family."


def test_tool_returns_the_same_driver_table_as_public_renderer(context):
    snapshot, cfg, view, pack = context
    session = AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack)
    content, error = dispatch(session, "inspect_recommendations", {
        "shortlist_ref": shortlist_reference(pack, view), "ranks": [1], "display": "drivers",
    })
    assert not error
    for kind in ("contributors", "detractors"):
        assert render_driver_table(pack, view, [1], kind) in content
    assert "Trade comparison" not in content
