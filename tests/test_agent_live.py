"""Phase 3 — live smoke test against the real Anthropic API.

Skipped unless ANTHROPIC_API_KEY is set. Verifies the loop wires to a real model:
a plain-English view triggers run_standard_pack and a coherent pack comes back.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY — live smoke test skipped",
)


def test_live_view_triggers_standard_pack():
    from agentic.agent_flow import AgentFlow
    from agentic.agent_llm import AnthropicToolLLM
    from agentic.session import AgentSession
    from config.loader import load_config
    from data.snapshot_loader import load_snapshot

    session = AgentSession(snapshot=load_snapshot(), cfg=load_config())
    flow = AgentFlow(AnthropicToolLLM(), session)

    out = flow.advance("I'm long USDBRL over the next 3 months, looking for about a 6% move higher.")

    assert session.pack is not None          # run_standard_pack fired
    assert session.view.pair == "USDBRL"
    assert isinstance(out, str) and out.strip()
