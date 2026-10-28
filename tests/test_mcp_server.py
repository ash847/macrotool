"""MacroTool MCP server: tool contract, conversation handling, calm failures, and a real
stdio round-trip (the way Claude Desktop talks to it). No Anthropic API calls — the inner
agent runs on a scripted fake model (MACROTOOL_MCP_FAKE=1); the engine itself is real.

Skipped automatically when the optional `mcp` package isn't installed.
"""

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp_server import server as S  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch):
    monkeypatch.setenv("MACROTOOL_MCP_FAKE", "1")
    monkeypatch.setattr(S, "_flow", None)
    yield


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    """Capture what would go to Supabase (never writes for real in tests)."""
    from interface import supabase_logger as L
    rows = {"chat_turns": [], "app_errors": []}
    monkeypatch.setattr(L, "log_chat_turn", lambda **kw: rows["chat_turns"].append(kw))
    monkeypatch.setattr(L, "log_app_error", lambda **kw: rows["app_errors"].append(kw))
    monkeypatch.setattr(S, "_secret", lambda name: "pm@fund.com" if name == "MACROTOOL_USER_EMAIL" else None)
    return rows


def _ask(question, **kw):
    return asyncio.run(S.ask_macrotool(question, **kw))


def test_exactly_one_tool_with_the_routing_contract():
    (tool,) = asyncio.run(S.server.list_tools())
    assert tool.name == "ask_macrotool"
    assert tool.input_schema["required"] == ["question"]
    assert set(tool.input_schema["properties"]) == {"question", "new_conversation"}
    d = tool.description
    # the three things the outer Claude must be told
    assert "ALWAYS call it instead of answering from your own knowledge" in d      # when to route
    assert "Do not restate a view in your own wording" in d                        # don't reinterpret
    assert "exactly as" in d and "returned" in d                                   # relay verbatim
    assert "new_conversation=true" in d                                            # follow-ups continue


def test_server_level_instructions_route_everything_to_the_tool():
    assert "ask_macrotool" in S.SERVER_INSTRUCTIONS
    assert "exactly as returned" in S.SERVER_INSTRUCTIONS


def test_first_reply_is_the_real_engine_output_not_the_model():
    reply = _ask("I think USDBRL falls to 4.50 in three months")
    assert "### Market state — USDBRL" in reply
    assert "| Rank | Structure | Variant |" in reply        # priced by the real engine


def test_follow_ups_continue_the_same_conversation_and_new_conversation_resets():
    _ask("I think USDBRL falls to 4.50 in three months")
    first = S._flow
    assert _ask("why is the top structure ranked first?") == "FAKE REPLY 2"
    assert S._flow is first                                 # same session carried on
    _ask("a different trade", new_conversation=True)
    assert S._flow is not first                             # fresh session


def test_missing_api_key_is_explained_not_raised(monkeypatch):
    monkeypatch.delenv("MACROTOOL_MCP_FAKE")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MACROTOOL_SECRETS", str(REPO / "does-not-exist.toml"))
    monkeypatch.setattr(S, "_secret", lambda name: None)
    reply = _ask("USDBRL view")
    assert "isn't set up yet" in reply and "ANTHROPIC_API_KEY" in reply


def test_unexpected_failure_gives_a_calm_message_without_a_traceback(monkeypatch):
    def boom():
        raise ValueError("secret internal detail")
    monkeypatch.setattr(S, "_new_flow", boom)
    reply = _ask("anything")
    assert "couldn't complete that request" in reply
    assert "secret internal detail" not in reply and "Traceback" not in reply


def test_print_config_points_at_this_checkout(capsys):
    S._print_config()
    out = capsys.readouterr().out
    assert '"macrotool"' in out and str(REPO / "mcp_server" / "server.py") in out


def test_real_stdio_round_trip():
    """Spawn the server as Claude Desktop would, list the tool, call it twice."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(REPO / "mcp_server" / "server.py")],
            env={"MACROTOOL_MCP_FAKE": "1", "PATH": "/usr/bin:/bin"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                first = await session.call_tool("ask_macrotool", {"question": "USDBRL to 4.50 in 3 months"})
                second = await session.call_tool("ask_macrotool", {"question": "why is #1 first?"})
                return tools, first, second

    tools, first, second = asyncio.run(asyncio.wait_for(run(), timeout=90))
    assert [t.name for t in tools] == ["ask_macrotool"]
    assert not first.is_error
    assert "### Market state — USDBRL" in first.content[0].text
    assert second.content[0].text == "FAKE REPLY 2"          # server kept the conversation


# --------------------------------------------------------------------------------------
# chat_turns logging (surface "mcp")
# --------------------------------------------------------------------------------------

def test_each_exchange_is_logged_like_the_app_with_its_tool_trace(logged):
    q1, q2 = "I think USDBRL falls to 4.50 in three months", "why is the top structure ranked first?"
    r1, r2 = _ask(q1), _ask(q2)
    rows = logged["chat_turns"]
    assert [(r["seq"], r["role"]) for r in rows] == [
        (0, "user"), (1, "assistant"), (2, "user"), (3, "assistant")]
    assert [r["text"] for r in rows] == [q1, r1, q2, r2]       # the query MacroTool received
    assert {r["surface"] for r in rows} == {"mcp"}
    assert {r["user_email"] for r in rows} == {"pm@fund.com"}
    assert {r["session_id"] for r in rows} == {S.SESSION_ID}
    assert len({r["chat_id"] for r in rows}) == 1
    # The fake model keeps its own message format, so only the provider-neutral per-call
    # metrics show here; tool-call extraction from Anthropic messages is tested below.
    assert [t["event"] for t in rows[1]["tool_trace"]] == ["llm_call", "llm_call"]
    assert [t["event"] for t in rows[3]["tool_trace"]] == ["llm_call"]
    assert rows[1]["pair"] == "USDBRL" and rows[1]["view_json"]["pair"] == "USDBRL"
    assert rows[0].get("tool_trace") is None and rows[2].get("tool_trace") is None


def test_tool_trace_pairs_anthropic_tool_calls_with_their_results():
    from agentic.telemetry import tool_trace_from_messages
    messages = [
        {"role": "user", "content": "reprice the 1x1"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "pricing"},
            {"type": "tool_use", "id": "t1", "name": "price_structure", "input": {"request": "1x1 40/20"}},
            {"type": "tool_use", "id": "t2", "name": "price_structure", "input": {"request": "1x1 5.0/4.8"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": "bad strike", "is_error": True},
            {"type": "tool_result", "tool_use_id": "t1", "content": "priced"},
        ]},
    ]
    assert tool_trace_from_messages(messages) == [
        {"name": "price_structure", "args": {"request": "1x1 5.0/4.8"}, "result": "bad strike", "is_error": True},
        {"name": "price_structure", "args": {"request": "1x1 40/20"}, "result": "priced", "is_error": False},
    ]


def test_new_conversation_gets_a_new_chat_id_and_restarts_seq(logged):
    _ask("I think USDBRL falls to 4.50 in three months")
    _ask("a different trade", new_conversation=True)
    rows = logged["chat_turns"]
    assert rows[0]["chat_id"] != rows[2]["chat_id"]
    assert [r["seq"] for r in rows] == [0, 1, 0, 1]


def test_a_failed_exchange_logs_the_error_and_the_reply_shown(logged, monkeypatch):
    _ask("I think USDBRL falls to 4.50 in three months")

    def boom(_question):
        raise ValueError("delta 0.048 outside supported range [0.10, 0.50]")
    monkeypatch.setattr(S._flow, "advance", boom)
    reply = _ask("reprice the 1x1 with 5.00 and 4.80 strikes")
    (err,) = logged["app_errors"]
    assert err["context"] == "mcp_ask_macrotool" and err["error_type"] == "ValueError"
    assert "0.048" in err["message"] and "Traceback" in err["traceback"]
    assert err["user_email"] == "pm@fund.com"
    user, answer = logged["chat_turns"][2:]
    assert user["text"] == "reprice the 1x1 with 5.00 and 4.80 strikes"
    assert answer["text"] == reply and "couldn't complete" in reply


def test_logging_failure_never_breaks_the_reply(logged, monkeypatch):
    from interface import supabase_logger as L

    def down(**_):
        raise ConnectionError("supabase down")
    monkeypatch.setattr(L, "log_chat_turn", down)
    assert "### Market state — USDBRL" in _ask("I think USDBRL falls to 4.50 in three months")
