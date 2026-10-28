"""MacroTool as a local MCP server — one tool that hands the PM's message to MacroTool's
own agent.

Why the whole agent and not just the calculator: the agent carries everything that makes
the answers trustworthy — direction conventions, carry framing, the plain-English
vocabulary, the comparison rules, and Python-built tables. The outer Claude's only job is
to pass the question in and show the reply as returned.

Run (Claude Desktop starts this for you):  python mcp_server/server.py
Print the Claude Desktop config for this machine:  python mcp_server/server.py --print-config
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib
import traceback
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mcp.server.mcpserver import Context, MCPServer  # noqa: E402

from agentic.agent_flow import AgentFlow  # noqa: E402
from agentic.agent_llm import DEFAULT_MODEL, AnthropicToolLLM, FakeToolLLM, LLMTurn, ToolCall  # noqa: E402
from agentic.session import AgentSession  # noqa: E402
from config.loader import load_config  # noqa: E402
from data.snapshot_loader import load_snapshot  # noqa: E402
from workspace.settings import ChatSettings  # noqa: E402

DEFAULT_CAPITAL = 100_000_000.0     # same default W as the app's sidebar
PROGRESS_EVERY_S = 5                # keep the client's request alive on long answers

TOOL_DESCRIPTION = """\
Ask MacroTool, an EM FX options structuring engine with its own analyst agent.

Use it for ANY question about FX option trade structuring, pricing, sizing or comparing
structures: a new view ("I think USDBRL goes to 4.50 in three months"), a change to a view
(new target, tenor or pair), and follow-ups about structures it has shown ("why is the
1x1.5 ranked above the vanilla?", "what is the main risk?", "how was this sized?"), plus
the market terms it uses. ALWAYS call it instead of answering from your own knowledge:
every number, strike, payoff, direction and ranking must come from MacroTool. Never
estimate, recompute or reinterpret them.

Pass the user's message in their own words, including the pair, tenor and target or move
if they gave them. Do not restate a view in your own wording or work out the direction
yourself. Call it again for every follow-up on the same trade. Set new_conversation=true
only when the user starts a different trade from scratch.

The reply is final and already formatted, tables included. Show it to the user exactly as
returned: do not summarise, reword, reorder, or add numbers, and do not translate its
directions or terms into your own. Add a brief note afterwards only if the user asked
something outside MacroTool's scope.
"""

SERVER_INSTRUCTIONS = (
    "Route every FX options structuring, pricing or sizing question to ask_macrotool, and "
    "relay its reply exactly as returned. Never answer such questions from your own knowledge."
)

server = MCPServer("macrotool", instructions=SERVER_INSTRUCTIONS)

_flow: AgentFlow | None = None          # one running conversation per server process
_lock = asyncio.Lock()

# chat_turns telemetry, same table as the app (surface "mcp"). One chat_id per
# conversation; seq counts rows within it (two per exchange).
SURFACE = "mcp"
SESSION_ID = str(uuid.uuid4())          # one per server process
_chat_id: str | None = None
_seq = 0


# --------------------------------------------------------------------------------------
# Building the agent
# --------------------------------------------------------------------------------------

def _secret(name: str) -> str | None:
    """A setting from the environment, else from a Streamlit-style secrets.toml
    (MACROTOOL_SECRETS if set, else the repo's own .streamlit/secrets.toml)."""
    if os.environ.get(name):
        return os.environ[name]
    for candidate in (os.environ.get("MACROTOOL_SECRETS"), REPO / ".streamlit" / "secrets.toml"):
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            value = tomllib.loads(Path(candidate).read_text()).get(name)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if value:
            return str(value)
    return None


def _fake_llm() -> FakeToolLLM:
    """Scripted stand-in for the model (protocol testing only; no API call)."""
    first = LLMTurn(text="", stop_reason="tool_use", tool_calls=[ToolCall(
        "t1", "run_standard_pack",
        {"pair": "USDBRL", "direction": "base_lower", "horizon_days": 90, "magnitude_pct": 6.0})])
    replies = [LLMTurn(text=f"FAKE REPLY {i}", tool_calls=[], stop_reason="end_turn") for i in range(1, 9)]
    return FakeToolLLM(script=[first, *replies])


def _make_llm():
    if os.environ.get("MACROTOOL_MCP_FAKE") == "1":
        return _fake_llm()
    key = _secret("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "MacroTool has no Anthropic API key. Set ANTHROPIC_API_KEY in the server's "
            "environment, or point MACROTOOL_SECRETS at a secrets.toml that contains it.")
    return AnthropicToolLLM(api_key=key, model=_secret("ANTHROPIC_MODEL") or DEFAULT_MODEL)


def _new_flow() -> AgentFlow:
    global _chat_id, _seq
    session = AgentSession(snapshot=load_snapshot(), cfg=load_config(), linear_notional=DEFAULT_CAPITAL)
    ChatSettings().apply_to(session)
    flow = AgentFlow(_make_llm(), session)
    _chat_id, _seq = str(uuid.uuid4()), 0
    return flow


# --------------------------------------------------------------------------------------
# Logging (Supabase, via the app's own logger; a no-op when it isn't configured)
# --------------------------------------------------------------------------------------

def _configure_logging() -> None:
    """Give the app's Supabase logger its credentials from the secrets file. This also
    lets the engine read the same remote config as the app. Off in fake (test) mode."""
    if os.environ.get("MACROTOOL_MCP_FAKE") == "1":
        return
    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
        value = _secret(name)
        if value and not os.environ.get(name):
            os.environ[name] = value
    from interface.supabase_logger import init_status, reinit
    reinit()
    ok, err = init_status()
    if not ok:
        print(f"[macrotool-mcp] Supabase logging off: {err}", file=sys.stderr)


def _log_exchange(flow: AgentFlow, question: str, reply: str, pre_len: int) -> None:
    """The question and reply as two chat_turns rows. Never raises."""
    global _seq
    seq, _seq = _seq, _seq + 2
    try:
        from interface.supabase_logger import log_chat_exchange
        log_chat_exchange(surface=SURFACE, chat_id=_chat_id, seq=seq, session=flow.session,
                          prompt=question, reply=reply, pre_len=pre_len,
                          session_id=SESSION_ID, user_email=_secret("MACROTOOL_USER_EMAIL"))
    except Exception as e:
        print(f"[macrotool-mcp] chat logging failed: {type(e).__name__}: {e}", file=sys.stderr)


def _log_failure(error: Exception) -> None:
    """The exception (with traceback) as an app_errors row. Never raises."""
    try:
        from interface.supabase_logger import log_app_error
        log_app_error(context="mcp_ask_macrotool", error_type=type(error).__name__,
                      message=str(error),
                      traceback="".join(traceback.format_exception(error)),
                      user_email=_secret("MACROTOOL_USER_EMAIL"), session_id=SESSION_ID)
    except Exception as e:
        print(f"[macrotool-mcp] error logging failed: {type(e).__name__}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------------------

@server.tool(name="ask_macrotool", description=TOOL_DESCRIPTION)
async def ask_macrotool(question: str, new_conversation: bool = False, ctx: Context | None = None) -> str:
    """question: the user's message, in their own words. new_conversation: start a fresh
    trade (default false: continue the current one)."""
    global _flow
    async with _lock:
        pre_len = None
        try:
            if _flow is None or new_conversation:
                _flow = _new_flow()
            pre_len = len(_flow.session.messages)
            task = asyncio.create_task(asyncio.to_thread(_flow.advance, question))
            waited = 0
            while True:
                done, _ = await asyncio.wait({task}, timeout=PROGRESS_EVERY_S)
                if done:
                    break
                waited += PROGRESS_EVERY_S
                if ctx is not None:
                    await ctx.report_progress(waited, None, f"MacroTool is working ({waited}s)…")
            reply = task.result()
        except RuntimeError as e:                       # configuration problems: say so plainly
            print(f"[macrotool-mcp] {e}", file=sys.stderr)
            reply = f"MacroTool isn't set up yet: {e}"
            await asyncio.to_thread(_log_failure, e)
        except Exception as e:                          # never leak a stack trace into the chat
            print(f"[macrotool-mcp] {type(e).__name__}: {e}", file=sys.stderr)
            reply = ("MacroTool couldn't complete that request. Please try again in a moment. "
                     "(Details are in the server log.)")
            await asyncio.to_thread(_log_failure, e)
        if pre_len is not None:                         # a conversation exists to log against
            await asyncio.to_thread(_log_exchange, _flow, question, reply, pre_len)
        return reply


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def _print_config() -> None:
    """The block to merge into Claude Desktop's claude_desktop_config.json for THIS checkout."""
    python = REPO / ".venv" / "bin" / "python"
    env = {}
    if not (REPO / ".streamlit" / "secrets.toml").is_file():
        env["MACROTOOL_SECRETS"] = "/ABSOLUTE/PATH/TO/secrets.toml  (must contain ANTHROPIC_API_KEY)"
    entry = {"command": str(python), "args": [str(Path(__file__).resolve())]}
    if env:
        entry["env"] = env
    print(json.dumps({"mcpServers": {"macrotool": entry}}, indent=2))


if __name__ == "__main__":
    if "--print-config" in sys.argv:
        _print_config()
    else:
        _configure_logging()
        server.run("stdio")
