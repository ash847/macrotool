from dataclasses import replace

import pytest

from agentic.agent_flow import AgentFlow, SYSTEM_PROMPT
from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
from agentic.dashboard import DEFAULT_FIELDS, FIELD_LABELS, dashboard_cells, render_dashboard
from agentic.session import AgentSession
from agentic.shortlist import inspection_tables, present_shortlist, shortlist_reference
from agentic.tools import dispatch
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.scenario_scorer import CellBreakdown


@pytest.fixture(scope="module")
def context():
    snapshot, cfg = load_snapshot(), load_config()
    session = AgentSession(snapshot=snapshot, cfg=cfg, linear_notional=100_000_000)
    _, error = dispatch(session, "run_standard_pack", {"pair": "GBPUSD", "horizon_days": 365, "target_level": 1.40})
    assert not error and session.pack.variants_ranked
    return snapshot, cfg, session.view, session.pack


def _session(context):
    snapshot, cfg, view, pack = context
    return AgentSession(snapshot=snapshot, cfg=cfg, view=view, pack=pack, linear_notional=100_000_000)


def _args(context, **changes):
    args = {"shortlist_ref": shortlist_reference(context[3], context[2]), "display": "dashboard", "ranks": [1, 2, 3, 4, 5]}
    return {**args, **changes}


def test_original_gbpusd_request_renders_one_transposed_table_without_repricing(context, monkeypatch):
    session = _session(context)
    def fail(*args, **kwargs):
        pytest.fail("A dashboard must not rebuild or reprice")
    monkeypatch.setattr("agentic.tools.build_pack", fail)
    monkeypatch.setattr("agentic.tools.price_structure", fail)
    fields = ["legs", "notional", "premium", "target_pnl", "target_return_on_premium", "loss_budget",
              "sizing_loss_proxy", "additional_loss_beyond_premium", "directional_tails", "top_contributor", "top_detractor"]
    args = _args(context, layout="trades_as_columns", fields=fields, driver_count=1)
    llm = FakeToolLLM(script=[
        LLMTurn("", [ToolCall("summary", "inspect_recommendations", args)], "tool_use"),
        LLMTurn("The sizing proxy is not a contractual loss limit.", [], "end_turn"),
    ])
    reply = AgentFlow(llm, session).advance("One table with trades in columns, all requested details and top one driver each side")
    lines = [line for line in reply.splitlines() if line.startswith("|")]
    assert len(lines) == len(fields) + 2
    assert all(line.count("|") == 7 for line in lines)
    assert "#1 ·" in lines[0] and "#5 ·" in lines[0]
    for field, line in zip(fields, lines[2:]):
        assert line.startswith("| " + FIELD_LABELS[field] + " |")
    assert "### Top contributors" not in reply and "Trade comparison" not in reply
    assert session.pack is context[3] and session.messages[-1]["content"] == reply
    assert "| Loss budget (reference) |" in reply and "| Sizing loss proxy |" in reply
    assert reply.count("| Metric |") == 1


def test_orientation_transposes_identical_engine_cells_and_respects_field_order(context):
    _, _, view, pack = context
    fields = ["top_contributor", "premium", "loss_budget", "notional"]
    outputs = [render_dashboard(pack, view, [3, 1], layout, fields, 1) for layout in ("trades_as_columns", "trades_as_rows")]
    column_lines = [line for line in outputs[0].splitlines() if line.startswith("|")]
    row_lines = [line for line in outputs[1].splitlines() if line.startswith("|")]
    assert column_lines[0].index("#1 ·") < column_lines[0].index("#3 ·")
    columns = [[cell.strip() for cell in line.split("|")[2:-1]] for line in column_lines[2:]]
    rows = [[cell.strip() for cell in line.split("|")[2:-1]] for line in row_lines[2:]]
    assert rows == [list(cells) for cells in zip(*columns)]


@pytest.mark.parametrize("display", ["dashboard", "trade_details", "both"])
def test_table_cells_do_not_require_html_line_breaks(context, display):
    _, _, view, pack = context
    tables = inspection_tables(display, [rec.rank for rec in pack.recommended])
    text = present_shortlist("", pack, view, tables=tables)
    assert "<br>" not in text
    assert "&lt;br&gt;" not in text
    assert "; " in text


def test_default_summary_needs_no_layout_or_field_confirmation(context):
    _, _, view, pack = context
    tables = inspection_tables("dashboard", [1, 2])
    assert tables["dashboard"]["layout"] == "trades_as_columns"
    assert tables["dashboard"]["fields"] == list(DEFAULT_FIELDS)
    assert tables["dashboard"]["driver_count"] == 1
    text = present_shortlist("Brief interpretation.", pack, view, tables=tables)
    assert "| Metric |" in text and "| Top contributor |" in text
    assert "| Top detractor |" not in text


@pytest.mark.parametrize("changes", [
    {"layout": "invented"}, {"fields": []}, {"fields": "premium"}, {"fields": ["score_ccy"]},
    {"fields": ["notional", "notional"]}, {"fields": [{"value": 100}]},
    {"driver_count": True}, {"driver_count": 0}, {"driver_count": 4}, {"driver_count": 1.5},
    {"display": "both", "layout": "trades_as_columns"}, {"values": {"premium": 1}},
    {"ranks": None}, {"ranks": [999]}, {"shortlist_ref": "stale"},
])
def test_invalid_or_numerical_requests_are_rejected(context, changes):
    session = _session(context)
    _, error = dispatch(session, "inspect_recommendations", _args(context, **changes))
    assert error
    assert session.pack is context[3]


def test_signed_driver_count_and_na_keep_full_denominator(context):
    _, _, view, pack = context
    cell = CellBreakdown("test", "Expiry", "K", 0.006, None, 1, 1, 0.006, None)
    other = replace(cell, scenario_id="other", col="K+½σ", contrib_pct=0.002)
    detractor = replace(cell, scenario_id="bad", col="−1σ", contrib_pct=-0.003)
    rec = replace(pack.recommended[0], cell_drivers=([cell, other], [detractor]), absolute_contribution_total_pct=0.012)
    one = dashboard_cells(rec, pack, view, 1)
    two = dashboard_cells(rec, pack, view, 2)
    assert one["top_contributor"] == "Target hit · Expiry: +50.0%"
    assert two["top_contributor"] == "Target hit · Expiry: +50.0%; Overshoot · Expiry: +16.7%"
    assert one["top_detractor"] == "Full reversal · Expiry: -25.0%"
    unavailable = dashboard_cells(replace(rec, absolute_contribution_total_pct=0), pack, view, 1)
    assert unavailable["top_contributor"].endswith("N/A")
    assert "Unavailable" in dashboard_cells(replace(rec, cell_drivers=None), pack, view, 1)["top_contributor"]
    assert "No positive" in dashboard_cells(replace(rec, cell_drivers=([], [])), pack, view, 1)["top_contributor"]


def test_financial_cells_match_canonical_engine_facts(context):
    _, _, view, pack = context
    for rec in pack.recommended[:5]:
        cells = dashboard_cells(rec, pack, view, 1)
        variant = rec.variant
        assert cells["notional"] == f"{variant.structure_notional:,.2f} GBP"
        assert cells["loss_budget"] == f"{pack.loss_budget:,.2f} GBP"
        if variant.economics is not None:
            economics = variant.economics
            assert cells["target_pnl"].startswith(f"{economics.target_net_pnl_pct * variant.structure_notional:,.2f} GBP")
            assert cells["sizing_loss_proxy"] == f"{economics.sizing_loss_pct * variant.structure_notional:,.2f} GBP"
            if economics.ratio_status == "not_applicable":
                assert cells["target_return_on_premium"] == "N/A — no premium outlay"
        else:
            assert cells["target_pnl"] == "Unavailable" and cells["target_return_on_premium"] == "Unavailable"


def test_missing_size_and_zero_allocations_are_not_fabricated(context):
    _, _, view, pack = context
    rec = next(rec for rec in pack.recommended if rec.variant.economics is not None)
    missing = replace(rec, variant=replace(rec.variant, structure_notional=None, net_premium_ccy=None))
    cells = dashboard_cells(missing, pack, view, 1)
    assert cells["notional"] == "Unavailable" and cells["target_pnl"].startswith("Unavailable")
    zero = replace(rec, variant=replace(rec.variant, structure_notional=0, net_premium_ccy=0))
    assert dashboard_cells(zero, pack, view, 1)["notional"] == "0.00 GBP"


def test_model_table_removed_and_dashboard_replaces_separate_tables(context):
    session = _session(context)
    calls = [ToolCall("old", "inspect_recommendations", _args(context, display="both")),
             ToolCall("new", "inspect_recommendations", _args(context, fields=["premium", "top_contributor"]))]
    llm = FakeToolLLM(script=[LLMTurn("", calls, "tool_use"),
        LLMTurn("| Made up | Amount |\n| --- | --- |\n| WRONG | 999999 |\n\nBrief note.", [], "end_turn")])
    reply = AgentFlow(llm, session).advance("Actually one dashboard only")
    assert "WRONG" not in reply and "999999" not in reply
    assert reply.count("| Metric |") == 1 and "### Top contributors" not in reply
    assert "Trade comparison" not in reply and reply.endswith("Brief note.")


def test_layout_followup_preserves_history_and_uses_one_call(context):
    session = _session(context)
    fields = ["notional", "premium", "top_contributor"]
    llm = FakeToolLLM(script=[
        LLMTurn("", [ToolCall("rows", "inspect_recommendations", _args(context, layout="trades_as_rows", fields=fields))], "tool_use"),
        LLMTurn("", [], "end_turn"),
        LLMTurn("", [ToolCall("columns", "inspect_recommendations", _args(context, layout="trades_as_columns", fields=fields))], "tool_use"),
        LLMTurn("", [], "end_turn"),
    ])
    flow = AgentFlow(llm, session)
    first = flow.advance("Those three fields, trades as rows")
    second = flow.advance("Transpose that")
    assert "| Trade |" in first and "| Metric |" in second
    assert session.messages[-1]["content"] == second
    assert any(message.get("content") == first for message in session.messages)
    assert "never supply numerical cell values" not in second


def test_prompt_exposes_supported_layout_without_overpromising():
    assert "display=dashboard" in SYSTEM_PROMPT
    assert "Do not claim an action was" in SYSTEM_PROMPT
    assert "do not demand a full list of rows" in SYSTEM_PROMPT
