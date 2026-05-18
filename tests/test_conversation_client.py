from __future__ import annotations

import pytest

from conversation.client import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    MacroToolClient,
    get_gemini_location,
    get_gemini_mode,
    get_gemini_project,
    get_api_key_for_provider,
    has_gemini_adc_credentials,
    resolve_model,
    resolve_provider,
    use_gemini_vertexai,
)


class _FakeAnthropicStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _FakeAnthropicConnectionError(Exception):
    pass


class _FakeAnthropicStream:
    def __init__(self, chunks):
        self.text_stream = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeAnthropicMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _FakeAnthropicStream(response)


class _FakeAnthropicClient:
    def __init__(self, responses):
        self.messages = _FakeAnthropicMessages(responses)


class _FakeAnthropicModule:
    APIStatusError = _FakeAnthropicStatusError
    APIConnectionError = _FakeAnthropicConnectionError

    def __init__(self, responses):
        self._responses = responses

    def Anthropic(self, api_key=None):
        return _FakeAnthropicClient(self._responses)


class _FakeGeminiChunk:
    def __init__(self, text: str):
        self.text = text


class _FakeGenerateContentConfig:
    def __init__(self, system_instruction: str, max_output_tokens: int):
        self.system_instruction = system_instruction
        self.max_output_tokens = max_output_tokens


class _FakeGeminiModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return iter(response)


class _FakeGeminiClient:
    def __init__(self, responses, api_key=None, credentials=None):
        self.api_key = api_key
        self.credentials = credentials
        self.models = _FakeGeminiModels(responses)


class _FakeGenAIModule:
    def __init__(self, responses):
        self._responses = responses

    def Client(self, api_key=None, credentials=None, **kwargs):
        return _FakeGeminiClient(self._responses, api_key=api_key, credentials=credentials)


class _FakeGeminiTypesModule:
    GenerateContentConfig = _FakeGenerateContentConfig


class _FakeGeminiRetryableError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_resolve_provider_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider() == "anthropic"


def test_resolve_provider_rejects_unknown_value():
    with pytest.raises(ValueError):
        resolve_provider("openai")


def test_resolve_model_uses_provider_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert resolve_model("anthropic") == DEFAULT_ANTHROPIC_MODEL
    assert resolve_model("gemini") == DEFAULT_GEMINI_MODEL


def test_get_api_key_for_provider_uses_provider_specific_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")

    assert get_api_key_for_provider("anthropic") == "anth-key"
    assert get_api_key_for_provider("gemini") == "gem-key"


def test_gemini_vertex_helpers_read_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "macrotool-dev")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    assert use_gemini_vertexai() is True
    assert get_gemini_mode() == "vertexai"
    assert get_gemini_project() == "macrotool-dev"
    assert get_gemini_location() == "us-central1"


def test_has_gemini_adc_credentials_false_without_vertex(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    assert has_gemini_adc_credentials() is False


def test_has_gemini_adc_credentials_true_with_explicit_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    assert has_gemini_adc_credentials(credentials=object()) is True


def test_anthropic_client_streams_and_buffers(monkeypatch):
    responses = [["Hello", " world"]]
    fake_anthropic = _FakeAnthropicModule(responses)

    def fake_import(name):
        assert name == "anthropic"
        return fake_anthropic

    monkeypatch.setattr("conversation.client.importlib.import_module", fake_import)

    client = MacroToolClient(provider="anthropic", api_key="test-key", model="claude-test")
    chunks = list(client.stream([{"role": "user", "content": "Hi"}], "System prompt"))

    assert chunks == ["Hello", " world"]
    assert client.last_response == "Hello world"
    call = client._provider_client._client.messages.calls[0]
    assert call["model"] == "claude-test"
    assert call["system"] == "System prompt"
    assert call["messages"] == [{"role": "user", "content": "Hi"}]


def test_anthropic_client_retries_retryable_status(monkeypatch):
    responses = [_FakeAnthropicStatusError(529), ["Recovered"]]
    fake_anthropic = _FakeAnthropicModule(responses)

    def fake_import(name):
        assert name == "anthropic"
        return fake_anthropic

    monkeypatch.setattr("conversation.client.importlib.import_module", fake_import)
    monkeypatch.setattr("conversation.client.time.sleep", lambda _: None)

    client = MacroToolClient(provider="anthropic", api_key="test-key")
    chunks = list(client.stream([{"role": "user", "content": "Hi"}], "System"))

    assert chunks == ["Recovered"]
    assert client.last_response == "Recovered"
    assert len(client._provider_client._client.messages.calls) == 2


def test_gemini_client_maps_messages_and_system_instruction(monkeypatch):
    responses = [[_FakeGeminiChunk("Answer"), _FakeGeminiChunk(" done")]]
    fake_genai = _FakeGenAIModule(responses)
    fake_types = _FakeGeminiTypesModule()

    def fake_import(name):
        if name == "google.genai":
            return fake_genai
        if name == "google.genai.types":
            return fake_types
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("conversation.client.importlib.import_module", fake_import)

    client = MacroToolClient(provider="gemini", api_key="gem-key", model="gemini-test")
    chunks = list(client.stream([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Why this structure?"},
    ], "Use the explanation pack.", max_tokens=333))

    assert chunks == ["Answer", " done"]
    assert client.last_response == "Answer done"

    call = client._provider_client._client.models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi there"}]},
        {"role": "user", "parts": [{"text": "Why this structure?"}]},
    ]
    assert call["config"].system_instruction == "Use the explanation pack."
    assert call["config"].max_output_tokens == 333


def test_gemini_client_uses_vertexai_config_when_enabled(monkeypatch):
    responses = [[_FakeGeminiChunk("Vertex OK")]]
    fake_genai = _FakeGenAIModule(responses)
    fake_types = _FakeGeminiTypesModule()
    captured_kwargs = {}

    class _CapturingGenAIModule(_FakeGenAIModule):
        def Client(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeGeminiClient(self._responses, api_key=kwargs.get("api_key"))

    def fake_import(name):
        if name == "google.genai":
            return _CapturingGenAIModule(responses)
        if name == "google.genai.types":
            return fake_types
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("conversation.client.importlib.import_module", fake_import)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "macrotool-dev")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = MacroToolClient(provider="gemini", model="gemini-test")
    chunks = list(client.stream([{"role": "user", "content": "Hello"}], "System"))

    assert chunks == ["Vertex OK"]
    assert captured_kwargs == {
        "vertexai": True,
        "project": "macrotool-dev",
        "location": "us-central1",
    }


def test_gemini_client_passes_explicit_vertex_credentials(monkeypatch):
    responses = [[_FakeGeminiChunk("Vertex Secret OK")]]
    fake_types = _FakeGeminiTypesModule()
    captured_kwargs = {}
    sentinel_credentials = object()

    class _CapturingGenAIModule(_FakeGenAIModule):
        def Client(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeGeminiClient(
                self._responses,
                api_key=kwargs.get("api_key"),
                credentials=kwargs.get("credentials"),
            )

    def fake_import(name):
        if name == "google.genai":
            return _CapturingGenAIModule(responses)
        if name == "google.genai.types":
            return fake_types
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("conversation.client.importlib.import_module", fake_import)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "macrotool-dev")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = MacroToolClient(
        provider="gemini",
        model="gemini-test",
        credentials=sentinel_credentials,
    )
    chunks = list(client.stream([{"role": "user", "content": "Hello"}], "System"))

    assert chunks == ["Vertex Secret OK"]
    assert captured_kwargs == {
        "vertexai": True,
        "project": "macrotool-dev",
        "location": "us-central1",
        "credentials": sentinel_credentials,
    }


def test_gemini_client_retries_retryable_exception(monkeypatch):
    responses = [_FakeGeminiRetryableError(503), [_FakeGeminiChunk("Recovered")]]
    fake_genai = _FakeGenAIModule(responses)
    fake_types = _FakeGeminiTypesModule()

    def fake_import(name):
        if name == "google.genai":
            return fake_genai
        if name == "google.genai.types":
            return fake_types
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("conversation.client.importlib.import_module", fake_import)
    monkeypatch.setattr("conversation.client.time.sleep", lambda _: None)

    client = MacroToolClient(provider="gemini", api_key="gem-key")
    chunks = list(client.stream([{"role": "user", "content": "Hello"}], "System"))

    assert chunks == ["Recovered"]
    assert client.last_response == "Recovered"
    assert len(client._provider_client._client.models.calls) == 2


def test_gemini_client_ignores_blank_messages(monkeypatch):
    responses = [[_FakeGeminiChunk("OK")]]
    fake_genai = _FakeGenAIModule(responses)
    fake_types = _FakeGeminiTypesModule()

    def fake_import(name):
        if name == "google.genai":
            return fake_genai
        if name == "google.genai.types":
            return fake_types
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("conversation.client.importlib.import_module", fake_import)

    client = MacroToolClient(provider="gemini", api_key="gem-key")
    list(client.stream([
        {"role": "user", "content": "  "},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Live question"},
    ], "System"))

    call = client._provider_client._client.models.calls[0]
    assert call["contents"] == [{"role": "user", "parts": [{"text": "Live question"}]}]
