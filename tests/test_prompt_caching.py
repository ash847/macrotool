from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic.agent_llm import AnthropicToolLLM, FakeToolLLM, LLMTurn, ToolCall
from agentic.telemetry import exchange_trace, tracked_call


def adapter(enabled=True, usage=None):
    llm = AnthropicToolLLM.__new__(AnthropicToolLLM)
    llm.model = "claude-sonnet-5-5"
    llm.cache_enabled = enabled
    content = [SimpleNamespace(type="thinking", thinking="", signature="opaque"),
               SimpleNamespace(type="tool_use", id="call", name="probe", input={}),
               SimpleNamespace(type="text", text="Checking.")]
    response = SimpleNamespace(content=content, model=llm.model, stop_reason="tool_use",
                               usage=usage, _request_id="request-test")
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=response)))
    return llm, response


def test_cache_breakpoints_usage_and_raw_blocks_are_preserved():
    usage = SimpleNamespace(input_tokens=25, cache_creation_input_tokens=800,
                            cache_read_input_tokens=2000, output_tokens=50)
    llm, response = adapter(usage=usage)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"name": "probe", "input_schema": {"type": "object"}}]
    original = deepcopy((messages, tools))
    turn = llm.create(messages, "Stable instructions", tools)
    request = llm._client.messages.create.call_args.kwargs
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["system"] == [{"type": "text", "text": "Stable instructions", "cache_control": {"type": "ephemeral"}}]
    assert (messages, tools) == original
    assert request["messages"] is messages and request["tools"] is tools
    assert turn.text == "Checking." and turn.tool_calls == [ToolCall("call", "probe", {})]
    assert llm.format_assistant(turn)["content"] is response.content
    assert turn.metrics["input_tokens"] == 25
    assert turn.metrics["cache_creation_input_tokens"] == 800
    assert turn.metrics["cache_read_input_tokens"] == 2000
    assert turn.metrics["output_tokens"] == 50
    assert turn.metrics["latency_ms"] >= 0
    assert turn.metrics["request_id"] == "request-test"


def test_caching_can_be_disabled_and_missing_usage_is_not_zero():
    llm, _ = adapter(enabled=False)
    turn = llm.create([], "Instructions", [])
    request = llm._client.messages.create.call_args.kwargs
    assert "cache_control" not in request and request["system"] == "Instructions"
    assert turn.metrics["cache_read_input_tokens"] is None
    assert turn.metrics["cache_enabled"] is False


def test_empty_system_does_not_get_an_invalid_empty_cache_block():
    llm, _ = adapter()
    llm.create([{"role": "user", "content": "hello"}], "", [])
    assert llm._client.messages.create.call_args.kwargs["system"] == ""


def test_calls_are_scoped_to_exchange_and_kept_out_of_model_history():
    session = SimpleNamespace(messages=[{"role": "user", "content": "hello"}], llm_calls=[])
    llm = FakeToolLLM([LLMTurn("one", [], "end_turn", metrics={"cache_read_input_tokens": 500}),
                       LLMTurn("two", [], "end_turn", metrics={"cache_read_input_tokens": 750})])
    tracked_call(llm, session, "system", [], 0)
    tracked_call(llm, session, "system", [], 2)
    tools = [{"name": "engine", "result": "facts", "is_error": False}]
    trace = exchange_trace(tools, session, 2)
    assert len(trace) == 2 and trace[1]["cache_read_input_tokens"] == 750
    assert trace[1]["event"] == "llm_call"
    assert session.messages == [{"role": "user", "content": "hello"}]
    assert len(tools) == 1


def test_failure_logs_type_and_duration_but_not_secret_exception_text():
    session = SimpleNamespace(messages=[], llm_calls=[])
    llm = SimpleNamespace(model="test", create=Mock(side_effect=RuntimeError("private content")))
    with pytest.raises(RuntimeError):
        tracked_call(llm, session, "system", [], 0)
    record = session.llm_calls[0]
    assert record["is_error"] and record["error_type"] == "RuntimeError"
    assert record["latency_ms"] >= 0 and record["input_tokens"] is None
    assert "private content" not in str(record)


def test_flow_records_each_round_including_final_response(monkeypatch):
    from agentic.agent_flow import AgentFlow
    from agentic.session import AgentSession
    session = AgentSession(snapshot=SimpleNamespace(currencies={}), cfg=None)
    llm = FakeToolLLM([
        LLMTurn("", [ToolCall("probe", "probe", {})], "tool_use", metrics={"cache_creation_input_tokens": 1000}),
        LLMTurn("Done", [], "end_turn", metrics={"cache_read_input_tokens": 1000}),
    ])
    monkeypatch.setattr("agentic.agent_flow.dispatch", lambda *_: ("facts", False))
    assert AgentFlow(llm, session).advance("check") == "Done"
    assert len(session.llm_calls) == 2
    assert all(call["exchange_start"] == 0 for call in session.llm_calls)
    assert [call["stop_reason"] for call in session.llm_calls] == ["tool_use", "end_turn"]


def test_usage_trace_fits_existing_supabase_log(monkeypatch):
    from interface import supabase_logger
    client = Mock()
    monkeypatch.setattr(supabase_logger, "_service_client", client)
    trace = [{"event": "llm_call", "input_tokens": 20, "cache_read_input_tokens": 1000}]
    supabase_logger.log_chat_turn(session_id="session", chat_id="chat", seq=1,
                                  surface="agent_tab", role="assistant", text="reply", tool_trace=trace)
    row = client.table.return_value.insert.call_args.args[0]
    assert row["tool_trace"] == trace
