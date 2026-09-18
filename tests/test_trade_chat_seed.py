"""Seeding an AgentSession from a Trade View trade (agentic/seed.py, task 1).

The seed injects a synthetic run_standard_pack turn so the agent starts already knowing
the trade — no extra API call — and the history ends on an assistant turn so the next
real user message alternates correctly.
"""

from __future__ import annotations

import pytest

from agentic.agent_flow import AgentFlow
from agentic.agent_llm import FakeToolLLM, LLMTurn
from agentic.seed import (
    DEFAULT_OPENING,
    SEED_TOOL_ID,
    seed_session_from_pack,
    view_to_pack_args,
)
from agentic.session import AgentSession
from agentic.standard_pack import build_pack
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView


@pytest.fixture(scope="module")
def seeded():
    snap = load_snapshot()
    cfg = load_config()
    view = TradeView(
        pair="USDBRL", direction="base_higher", direction_conviction="medium",
        horizon_days=90, magnitude_pct=6.0, mode="recommend",
    )
    pack = build_pack(view, snap.get("USDBRL"), cfg)
    session = AgentSession(snapshot=snap, cfg=cfg)
    seed_session_from_pack(session, view, pack)
    return session, view, pack


def test_view_to_pack_args_roundtrips_the_view():
    view = TradeView(
        pair="USDTRY", direction="base_lower", direction_conviction="medium",
        horizon_days=60, magnitude_pct=8.0, mode="recommend",
    )
    args = view_to_pack_args(view)
    assert args == {
        "pair": "USDTRY", "horizon_days": 60, "direction": "base_lower",
        "mode": "recommend", "magnitude_pct": 8.0,
    }


def test_pure_directional_view_omits_magnitude():
    view = TradeView(
        pair="USDBRL", direction="base_higher", direction_conviction="medium",
        horizon_days=30, mode="recommend",
    )
    assert "magnitude_pct" not in view_to_pack_args(view)


def test_seed_sets_pack_and_prewarms_cache(seeded):
    session, view, pack = seeded
    assert session.pack is pack
    assert session.view is view
    assert session.get_cached(view) is pack  # a re-run of the same view reuses it


def test_seed_message_shape(seeded):
    session, _, _ = seeded
    roles = [m["role"] for m in session.messages]
    assert roles == ["assistant", "user", "assistant"]

    tool_use = session.messages[0]["content"][1]
    tool_result = session.messages[1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "run_standard_pack"
    assert tool_use["id"] == SEED_TOOL_ID
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == SEED_TOOL_ID
    assert tool_result["is_error"] is False
    assert tool_result["content"]  # the rendered pack text

    # History ends on an assistant opening line so the next user turn alternates.
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["content"] == DEFAULT_OPENING


def test_advance_over_seeded_session_narrates_without_reprice(seeded):
    """A 'why' question on a seeded session narrates over the pack (no tool call)."""
    session, _, _ = seeded
    # Fresh session copy so we don't mutate the module fixture's history for other tests.
    s2 = AgentSession(snapshot=session.snapshot, cfg=session.cfg)
    s2.messages = list(session.messages)
    s2.view, s2.pack = session.view, session.pack

    llm = FakeToolLLM(script=[LLMTurn(text="It leads on carry roll-down.", tool_calls=[], stop_reason="end_turn")])
    flow = AgentFlow(llm, s2)
    reply = flow.advance("why the top pick?")
    assert reply == "It leads on carry roll-down."
    # The model saw the seeded pack in its message history.
    seen_roles = [m["role"] for m in llm.seen[0]["messages"]]
    assert seen_roles[:3] == ["assistant", "user", "assistant"]
    assert seen_roles[-1] == "user"  # the appended question
