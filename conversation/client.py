"""
Provider-aware LLM client facade.

The rest of MacroTool uses a normalized message format:

    {"role": "user" | "assistant", "content": "..."}

Provider adapters translate that into the request shapes expected by the SDK.
They all expose the same small streaming contract and buffer the aggregated
response in ``last_response`` for tag parsing by the flow layer.
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any, Generator, Protocol

DEFAULT_PROVIDER = "anthropic"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_TOKENS = 2048  # Conversational responses; not long-form generation
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]  # seconds between attempts
_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_MODEL_ENV_VARS = {
    "anthropic": "ANTHROPIC_MODEL",
    "gemini": "GEMINI_MODEL",
}
_GEMINI_VERTEX_ENV = "GOOGLE_GENAI_USE_VERTEXAI"
_GEMINI_PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"
_GEMINI_LOCATION_ENV = "GOOGLE_CLOUD_LOCATION"
_DEFAULT_MODELS = {
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
}


class _ProviderClient(Protocol):
    provider: str
    model: str
    last_response: str

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int,
    ) -> Generator[str, None, None]: ...


def resolve_provider(provider: str | None = None) -> str:
    resolved = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if resolved not in _DEFAULT_MODELS:
        supported = ", ".join(sorted(_DEFAULT_MODELS))
        raise ValueError(f"Unsupported LLM provider {resolved!r}. Expected one of: {supported}")
    return resolved


def resolve_model(provider: str, model: str | None = None) -> str:
    resolved_provider = resolve_provider(provider)
    if model:
        return model
    env_model = os.environ.get(_MODEL_ENV_VARS[resolved_provider])
    return env_model or _DEFAULT_MODELS[resolved_provider]


def get_api_key_for_provider(provider: str) -> str | None:
    return os.environ.get(_PROVIDER_ENV_VARS[resolve_provider(provider)])


def use_gemini_vertexai() -> bool:
    value = os.environ.get(_GEMINI_VERTEX_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_gemini_project() -> str | None:
    value = os.environ.get(_GEMINI_PROJECT_ENV)
    return value or None


def get_gemini_location() -> str | None:
    value = os.environ.get(_GEMINI_LOCATION_ENV)
    return value or None


def get_gemini_mode() -> str:
    return "vertexai" if use_gemini_vertexai() else "developer_api"


def has_gemini_adc_credentials(credentials: Any | None = None) -> bool:
    if credentials is not None:
        return True
    if not use_gemini_vertexai():
        return False
    try:
        google_auth = importlib.import_module("google.auth")
        google_auth.default()
        return True
    except Exception:
        return False


class MacroToolClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider: str = DEFAULT_PROVIDER,
        credentials: Any | None = None,
    ):
        self.provider = resolve_provider(provider)
        self.model = resolve_model(self.provider, model)
        effective_key = api_key if api_key is not None else get_api_key_for_provider(self.provider)
        if self.provider == "anthropic":
            self._provider_client: _ProviderClient = AnthropicProviderClient(
                api_key=effective_key,
                model=self.model,
            )
        else:
            self._provider_client = GeminiProviderClient(
                api_key=effective_key,
                model=self.model,
                use_vertexai=use_gemini_vertexai(),
                project=get_gemini_project(),
                location=get_gemini_location(),
                credentials=credentials,
            )

    @property
    def last_response(self) -> str:
        return self._provider_client.last_response

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = MAX_TOKENS,
    ) -> Generator[str, None, None]:
        yield from self._provider_client.stream(messages, system, max_tokens)


class AnthropicProviderClient:
    provider = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_ANTHROPIC_MODEL):
        anthropic = importlib.import_module("anthropic")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.last_response = ""

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = MAX_TOKENS,
    ) -> Generator[str, None, None]:
        last_exc: Exception | None = None
        self.last_response = ""

        for attempt in range(_MAX_RETRIES):
            try:
                with self._client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        self.last_response += text
                        yield text
                return
            except self._anthropic.APIStatusError as exc:
                if _should_retry_status(getattr(exc, "status_code", None)):
                    last_exc = exc
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_RETRY_DELAYS[attempt])
                        self.last_response = ""
                        continue
                raise
            except self._anthropic.APIConnectionError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAYS[attempt])
                    self.last_response = ""
                    continue
                raise

        if last_exc is not None:
            raise last_exc


class GeminiProviderClient:
    provider = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        use_vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        credentials: Any | None = None,
    ):
        genai = importlib.import_module("google.genai")
        self._types = importlib.import_module("google.genai.types")
        self._mode = "vertexai" if use_vertexai else "developer_api"
        self._project = project
        self._location = location
        self._credentials = credentials
        if use_vertexai:
            client_kwargs = {
                "vertexai": True,
                "project": project,
                "location": location,
            }
            if credentials is not None:
                client_kwargs["credentials"] = credentials
        else:
            client_kwargs = {"api_key": api_key}
        self._client = genai.Client(**client_kwargs)
        self.model = model
        self.last_response = ""

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = MAX_TOKENS,
    ) -> Generator[str, None, None]:
        contents = _messages_to_gemini_contents(messages)
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        last_exc: Exception | None = None
        self.last_response = ""

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                for chunk in response:
                    text = getattr(chunk, "text", None) or ""
                    if not text:
                        continue
                    self.last_response += text
                    yield text
                return
            except Exception as exc:
                if _should_retry_google_exception(exc):
                    last_exc = exc
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_RETRY_DELAYS[attempt])
                        self.last_response = ""
                        continue
                raise

        if last_exc is not None:
            raise last_exc


def _messages_to_gemini_contents(messages: list[dict]) -> list[dict]:
    contents: list[dict] = []
    for message in messages:
        text = (message.get("content") or "").strip()
        if not text:
            continue
        role = "model" if message.get("role") == "assistant" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": text}],
        })
    return contents


def _should_retry_status(status_code: int | None) -> bool:
    return status_code in (429, 503, 529) or (status_code is not None and status_code >= 500)


def _should_retry_google_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if _should_retry_status(status_code):
        return True

    code = getattr(exc, "code", None)
    if _should_retry_status(code):
        return True

    message = str(exc).lower()
    retry_fragments = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "resource exhausted",
        "unavailable",
        "timed out",
        "timeout",
        "connection",
        "temporarily",
        "overloaded",
    )
    return any(fragment in message for fragment in retry_fragments)
