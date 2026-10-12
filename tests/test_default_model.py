from agentic.agent_llm import DEFAULT_MODEL
from conversation.client import DEFAULT_ANTHROPIC_MODEL, resolve_model


def test_agent_and_ui_default_to_sonnet_55(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert DEFAULT_MODEL == DEFAULT_ANTHROPIC_MODEL == "claude-sonnet-5-5"
    assert resolve_model("anthropic") == DEFAULT_MODEL


def test_explicit_model_override_is_preserved():
    assert resolve_model("anthropic", "claude-sonnet-4-6") == "claude-sonnet-4-6"
