"""Background agent turns (interface/agent_job.py): the turn completes and is saved
independently of the Streamlit page that started it (so navigating away mid-turn no
longer loses the reply)."""

from __future__ import annotations

import threading

from agentic.agent_flow import AgentFlow
from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
from agentic.session import AgentSession
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from interface.agent_job import start_agent_job
from workspace.service import ConversationService, display_turns
from workspace.store import InMemoryStore

ME = "pm@fund.com"


class _GatedLLM(FakeToolLLM):
    """FakeToolLLM whose first call blocks until ``gate`` is set — a slow API call."""

    def __init__(self, script):
        super().__init__(script=script)
        self.gate = threading.Event()

    def create(self, messages, system, tools):
        self.gate.wait(10)
        return super().create(messages, system, tools)


def _setup(script):
    llm = _GatedLLM(script)
    session = AgentSession(snapshot=load_snapshot(), cfg=load_config())
    svc = ConversationService(InMemoryStore(), ME)
    return llm, AgentFlow(llm, session), svc, svc.new_conversation()


def test_turn_completes_and_saves_without_the_page():
    llm, flow, svc, conv = _setup([
        LLMTurn(text="", tool_calls=[ToolCall(id="t1", name="run_standard_pack",
                args={"pair": "USDBRL", "horizon_days": 90, "target_level": 4.9})],
                stop_reason="tool_use"),
        LLMTurn(text="the read", tool_calls=[], stop_reason="end_turn"),
    ])
    seen = []
    job = start_agent_job(flow, "BRL lower", conversation=conv, seq=0, service=svc,
                          after=lambda j: seen.append(j.reply))
    assert not job.done.is_set()          # still "calling the API"
    llm.gate.set()
    assert job.done.wait(30)
    assert job.reply == "the read" and not job.failed and job.store_error is None
    assert job.conversation.title.startswith("USDBRL ↓")
    assert seen == ["the read"]           # telemetry hook ran in the thread
    saved = svc.store.list_turns(ME, conv.id)
    assert display_turns(saved) == [("user", "BRL lower"), ("assistant", "the read")]
    assert saved[0].llm_messages          # replayable history saved


def test_llm_failure_is_captured_and_saved_as_a_failed_turn():
    llm, flow, svc, conv = _setup([])     # exhausted script → create() raises
    llm.gate.set()
    job = start_agent_job(flow, "hello", conversation=conv, seq=0, service=svc)
    assert job.done.wait(30)
    assert job.failed and job.error is not None and job.reply.startswith("⚠️")
    (turn,) = svc.store.list_turns(ME, conv.id)
    assert turn.llm_messages == [] and turn.reply_text.startswith("⚠️")
