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
