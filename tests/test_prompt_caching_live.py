"""Opt-in paid checks of system-prefix and multi-turn cache reuse."""

import os
import uuid

import pytest

from agentic.agent_llm import AnthropicToolLLM


@pytest.mark.skipif(os.getenv("RUN_LIVE_CACHE_TESTS") != "1", reason="Opt-in paid model calls")
def test_live_system_and_history_cache_hits():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key and os.getenv("LIVE_SECRETS_FILE"):
        import tomllib
        with open(os.environ["LIVE_SECRETS_FILE"], "rb") as source:
            key = tomllib.load(source).get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("No Anthropic key configured")
    llm = AnthropicToolLLM(api_key=key)
    system = f"Cache validation {uuid.uuid4()}. " + (
        "This is a synthetic integration check. Use the probe tool when asked, then acknowledge its result briefly. "
        "Do not invent financial facts or numerical results. Preserve the supplied tool facts and answer concisely. "
    ) * 25
    tools = [{"name": "cache_probe", "description": "Return a synthetic acknowledgement.",
              "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}}]
    messages = [llm.format_user("Call cache_probe once now; after its result, say only 'Acknowledged'.")]
    first = llm.create(messages, system, tools)
    print("\nCACHE FIRST", first.metrics)
    assert first.tool_calls and all(call.name == "cache_probe" for call in first.tool_calls)
    assert first.metrics["cache_creation_input_tokens"] > 0
    messages.append(llm.format_assistant(first))
    messages.append(llm.format_tool_results([(call, "Synthetic check succeeded.", False) for call in first.tool_calls]))
    second = llm.create(messages, system, tools)
    print("CACHE FOLLOW-UP", second.metrics)
    assert second.stop_reason == "end_turn" and second.text and not second.tool_calls
    assert second.metrics["cache_read_input_tokens"] > 0
    third = llm.create([llm.format_user("No tool needed. Say only 'Ready'.")], system, tools)
    print("CACHE NEW CHAT", third.metrics)
    assert third.stop_reason == "end_turn" and third.metrics["cache_read_input_tokens"] > 0
