"""Phase 3 tests — the agent loop, driven by a scripted fake LLM (no API)."""

from agentic.agent_flow import AgentFlow
from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
from agentic.session import AgentSession
from config.loader import load_config
from data.snapshot_loader import load_snapshot

_VIEW = {"pair": "USDBRL", "direction": "base_higher", "horizon_days": 60, "magnitude_pct": 6.0}


def _tool(name, args, tid="t1"):
    return LLMTurn(text="", tool_calls=[ToolCall(tid, name, args)], stop_reason="tool_use")


def _text(t):
    return LLMTurn(text=t, tool_calls=[], stop_reason="end_turn")


def _flow(script, max_rounds=6):
    session = AgentSession(snapshot=load_snapshot(), cfg=load_config())
    return AgentFlow(FakeToolLLM(script=script), session, max_rounds=max_rounds)


def test_pack_then_narrate():
    flow = _flow([_tool("run_standard_pack", dict(_VIEW)), _text("Here's the recommendation.")])
    out = flow.advance("I'm long USDBRL, 60d, target +6%")
    assert out == "Here's the recommendation."
    assert flow.session.pack is not None


def test_price_within_pack():
    flow = _flow([
        _tool("run_standard_pack", dict(_VIEW)),
        _tool("price_structure", {"request": "34 vs 25 1x1.5"}, tid="t2"),
        _text("That 1x1.5 cuts your premium."),
    ])
    out = flow.advance("long USDBRL 60d +6%, what about a 34 vs 25 1x1.5?")
    assert out == "That 1x1.5 cuts your premium."
    assert len(flow.session.priced) == 1


def test_needs_pack_guard_surfaces_to_model():
    # Model tries to price before establishing a view → tool returns an error string
    # that goes back to the model, which then narrates.
    flow = _flow([
        _tool("price_structure", {"request": "digital 10%"}),
        _text("Let me set up the view first — which pair?"),
    ])
    out = flow.advance("price a 10% digital")
    assert "view first" in out
    # The guard message reached the message history as a tool result.
    assert any("No standard pack yet" in str(m.get("content")) for m in flow.session.messages)


def test_cache_reused_across_turns():
    flow = _flow([
        _tool("run_standard_pack", dict(_VIEW)),
        _text("first"),
        _tool("run_standard_pack", dict(_VIEW)),   # identical view → cache hit
        _text("second"),
    ])
    flow.advance("long USDBRL 60d +6%")
    pack_after_first = flow.session.pack
    flow.advance("remind me of the setup")
    assert flow.session.pack is pack_after_first   # same object, no recompute
    assert len(flow.session._cache) == 1


def test_clarification_relayed():
    flow = _flow([
        _tool("run_standard_pack", dict(_VIEW)),
        _tool("price_structure", {"request": "34 vs 25"}, tid="t2"),  # ambiguous
        _text("Did you mean a 1x1, 1x1.5 or 1x2?"),
    ])
    out = flow.advance("long USDBRL 60d +6%, price 34 vs 25")
    assert "1x1.5" in out
    assert flow.session.priced == []   # nothing priced on a clarification


def test_iteration_bound():
    # Every turn asks for a tool, never finishes → loop must stop gracefully.
    script = [_tool("run_standard_pack", dict(_VIEW), tid=f"t{i}") for i in range(10)]
    flow = _flow(script, max_rounds=3)
    out = flow.advance("loop forever")
    assert "narrow the request" in out
