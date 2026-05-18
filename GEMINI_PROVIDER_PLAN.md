# Gemini Provider Plan

## Status: Complete (as of 2026-05-18)

All phases implemented and live on branch `claude/charming-leavitt-47b984`. Streamlit app runs on Gemini 2.5 Flash via Vertex AI with service account credentials. Key issues resolved during deployment:

- `uv.lock` was stale — `google-genai` was in `pyproject.toml` and `requirements.txt` but not in the lockfile, causing silent install failure. Fixed by running `uv lock` and committing.
- Service account `Credentials.from_service_account_info()` required explicit `scopes=["https://www.googleapis.com/auth/cloud-platform"]` — without it the OAuth token request fails with `invalid_scope`.
- Secrets must be top-level TOML keys, not nested under `[auth.google]`.

The plan below is retained as implementation history.

---

## Latest Deployment Debugging Update

Update as of May 18, 2026:

- Provider selection and Vertex config are now implemented in code and pushed on branch `claude/charming-leavitt-47b984`.
- Streamlit deployment debugging exposed two practical issues that were not obvious from the implementation plan alone:
  - Streamlit secrets placement matters:
    - `LLM_PROVIDER`, `GEMINI_MODEL`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` must be top-level TOML keys.
    - If they are placed under `[auth.google]`, the app falls back to Anthropic because it cannot find top-level `LLM_PROVIDER`.
  - Streamlit Cloud is currently honoring repo-root `requirements.txt` for package installation:
    - `google-genai` was already added to `pyproject.toml`
    - the deployed app still crashed with `ModuleNotFoundError: No module named 'google.genai'`
    - root cause was missing `google-genai` in repo-root `requirements.txt`
    - fixed in pushed commit `f75d8a9`
- Current deployment status after those fixes:
  - provider selection should now switch correctly to Gemini when secrets are top-level
  - the next likely blocker, if any, is service-account secret parsing rather than provider wiring

## Goal

Add Google Gemini as a second LLM provider alongside Anthropic, without changing the deterministic engine contract:

- Python still computes all trade numbers before any LLM call.
- `ConversationFlow` and `context_builder` remain provider-agnostic.
- We can switch providers in the app and in deployment config.
- Follow-up Q&A can be tested on Gemini with the same explanation-pack context already built in this branch.

## Update: Vertex AI / ADC Path

Plan update as of May 16, 2026:

- The first pass assumed Gemini Developer API keys.
- The target runtime for this project should now prefer Gemini on Vertex AI with Application Default Credentials when org policy blocks API keys.
- Developer API key mode can remain supported as an optional fallback, but Vertex AI + ADC is now the primary Gemini path to implement and document.
- For Streamlit Community Cloud specifically, the practical deployment path is now:
  - Vertex AI Gemini
  - service account JSON stored in Streamlit secrets
  - credentials constructed in-app and passed to the SDK
  - native ambient ADC remains a secondary path for environments that provide it

## Current State

As of May 16, 2026, the conversation stack is effectively Anthropic-only:

- `conversation/client.py` is a thin wrapper around `anthropic.Anthropic`.
- `ConversationFlow` depends only on `MacroToolClient.stream(messages, system, max_tokens=...)`.
- `interface/app.py` injects `ANTHROPIC_API_KEY` from Streamlit secrets and then manually replaces `new_flow._client._client`.
- The deterministic intake page is still in structured silent mode, but the follow-up conversation path is active in `ConversationFlow`.
- This worktree already builds and injects `explanation_pack_context` into follow-up prompts.

That is a good starting point because the provider-specific code is still concentrated in one small area.

## External API Notes

This plan now supports two Gemini auth modes:

- Use the official `google-genai` Python SDK.
- Prefer Vertex AI with ADC:
  - `vertexai=True`
  - `project=GOOGLE_CLOUD_PROJECT`
  - `location=GOOGLE_CLOUD_LOCATION`
  - credentials discovered from ADC
- On Streamlit Community Cloud, allow an equivalent credentials path by loading a GCP service account JSON from secrets and passing it as explicit `credentials=...`.
- Optionally support Gemini Developer API key mode:
  - `GEMINI_API_KEY`
- Send stateless text requests with `generateContent` / SDK equivalents.
- Pass prior conversation turns explicitly in `contents`.
- Pass the system prompt as Gemini `system_instruction`.
- Prefer streaming support if the SDK stream surface is stable in our environment; otherwise start with non-streaming and wrap it in the same client contract.

## Recommended Architecture

Introduce a provider abstraction at the client boundary, not in the engine or prompt layers.

### 1. Replace the Anthropic-specific wrapper with a provider-neutral client

Refactor `conversation/client.py` so `MacroToolClient` becomes a facade over provider implementations:

```python
class MacroToolClient:
    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
    ):
        ...

    def stream(self, messages: list[dict], system: str, max_tokens: int = 2048):
        ...
```

Under the hood:

- `AnthropicProviderClient`
- `GeminiProviderClient`

Both should expose the same small interface:

```python
class BaseProviderClient(Protocol):
    model: str
    last_response: str

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int,
    ) -> Generator[str, None, None]: ...
```

This keeps `ConversationFlow` unchanged except for how it constructs the client.

### 2. Keep the message schema normalized inside MacroTool

Today `ConversationFlow.messages` uses a simple internal format:

```python
{"role": "user" | "assistant", "content": "..."}
```

Keep that internal representation unchanged.

Each provider adapter should translate from this normalized format into its own request shape:

- Anthropic:
  - `messages=[{"role": ..., "content": ...}]`
  - `system=...`
- Gemini:
  - `contents=[{"role": "user"|"model", "parts": [{"text": ...}]}]`
  - `system_instruction={"parts": [{"text": system}]}`

This avoids provider-specific branching elsewhere in the codebase.

### 3. Add a small provider config layer

Add a single source of truth for LLM provider selection and defaults.

Suggested environment/secrets:

- `LLM_PROVIDER`
  - allowed: `anthropic`, `gemini`
  - default: `anthropic`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL`
- `GEMINI_MODEL`
- `GOOGLE_GENAI_USE_VERTEXAI`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- optional Streamlit secret section: `[gcp_service_account]`
- optional fallback: `GEMINI_API_KEY`

Recommended defaults:

- Anthropic default model:
  - keep current `claude-sonnet-4-6` unless the branch already changes it elsewhere
- Gemini default model:
  - choose one explicit default and pin it in config rather than relying on a moving default
  - prefer a fast, general-purpose text model suitable for follow-up Q&A

Put this logic in one place, e.g. a new `conversation/provider_config.py` or directly in `conversation/client.py` if kept small.

## Implementation Plan

## Phase 1: Provider Config and Secrets

Files:

- `interface/app.py`
- `conversation/client.py`
- `pyproject.toml`
- optional: `README.md` or deployment notes later

Changes:

- Add `google-genai` to dependencies in `pyproject.toml`.
- Extend `_inject_secrets()` to include:
  - `LLM_PROVIDER`
  - `GEMINI_MODEL`
  - `GOOGLE_GENAI_USE_VERTEXAI`
  - `GOOGLE_CLOUD_PROJECT`
  - `GOOGLE_CLOUD_LOCATION`
  - optional fallback: `GEMINI_API_KEY`
  - optionally `ANTHROPIC_MODEL`
- In Streamlit, also support a TOML section like:

```toml
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```
- Replace `_get_api_key()` with a provider-aware config helper:
  - `_get_llm_provider()`
  - `_get_provider_api_key(provider)`
  - `_get_provider_model(provider)`
- Update sidebar status messaging:
  - show which provider is active
  - report the correct readiness state for that provider
  - for Gemini Vertex mode, surface whether project/location are configured and whether ADC is detectable

Acceptance criteria:

- App can read provider selection from secrets/env.
- No direct Anthropic-only wording remains in the configuration path.

## Phase 2: Provider Abstraction in `conversation/client.py`

Files:

- `conversation/client.py`

Refactor:

- Extract Anthropic logic into `AnthropicProviderClient`.
- Add `GeminiProviderClient`.
- Make `MacroToolClient` a thin delegating facade.

Gemini responsibilities:

- Initialize the Google client in one of two ways:
  - Vertex AI mode: `genai.Client(vertexai=True, project=..., location=...)`
  - Developer API mode: `genai.Client(api_key=...)`
- In Vertex AI mode, also allow explicit credentials:
  - `genai.Client(vertexai=True, project=..., location=..., credentials=...)`
- Translate MacroTool messages to Gemini `contents`.
- Map `assistant` history turns to Gemini `model` turns.
- Pass `system` as `system_instruction`.
- Respect `max_tokens` if supported by the SDK field used; otherwise document the nearest equivalent.
- Buffer full response text into `last_response`.

Streaming decision:

- Preferred: implement true streaming if the `google-genai` SDK stream iterator is straightforward and stable.
- Fallback: implement a non-streaming Gemini call first, then yield the full text once.

Why this fallback is acceptable:

- The current primary goal is to test answer quality and feel.
- MacroTool already tolerates best-effort enrichment elsewhere.
- We can add token streaming later without changing `ConversationFlow`.

Acceptance criteria:

- `ConversationFlow` can call `.stream(...)` without knowing the provider.
- `last_response` remains available for override/tag parsing.

## Phase 3: Update `ConversationFlow` Construction

Files:

- `conversation/flow.py`
- `interface/app.py`

Changes:

- Update `ConversationFlow.__init__` to accept:

```python
def __init__(
    self,
    api_key: str | None = None,
    snapshot: MarketSnapshot | None = None,
    provider: str = "anthropic",
    model: str | None = None,
):
```

- Construct `MacroToolClient(provider=provider, api_key=api_key, model=model)`.
- In `_make_flow()`, stop mutating `new_flow._client._client` directly.
- Instead, create `ConversationFlow` with the resolved provider, key, and model.

This removes a fragile implementation leak from `interface/app.py`.

Acceptance criteria:

- No code outside `conversation/client.py` needs to import Anthropic or Google SDKs directly.

## Phase 4: Gemini Request Mapping Details

Files:

- `conversation/client.py`

Define a deterministic mapping:

### Internal MacroTool message

```python
{"role": "user", "content": "text"}
{"role": "assistant", "content": "text"}
```

### Gemini request content

```python
{
  "role": "user" | "model",
  "parts": [{"text": "..."}],
}
```

Rules:

- `user` -> `user`
- `assistant` -> `model`
- Ignore empty messages rather than sending blank turns
- Keep message order identical
- Send `system` separately, not as a synthetic first user turn

Potential gotcha:

- MacroTool relies on “messages must end in a user turn” before API calls.
- Preserve that invariant exactly for Gemini as well.
- Do not let provider-specific history transforms reorder or merge turns.

Acceptance criteria:

- The three-call intake flow and DONE follow-up flow work with the same internal `messages` list.

## Phase 4B: Vertex AI / ADC Runtime Support

Files:

- `conversation/client.py`
- `interface/app.py`
- tests

Changes:

- Add helpers for:
  - `GOOGLE_GENAI_USE_VERTEXAI`
  - `GOOGLE_CLOUD_PROJECT`
  - `GOOGLE_CLOUD_LOCATION`
- Add a Streamlit-facing service account secret loader for Community Cloud deployments.
- When Gemini Vertex mode is enabled:
  - do not require `GEMINI_API_KEY`
  - initialize the SDK for Vertex AI
  - allow ADC discovery from the runtime
  - allow explicit service account credentials constructed from secrets
- Update the app status text so a user can distinguish:
  - Vertex config missing
  - Vertex configured but ADC missing
  - Vertex ADC ready
  - Vertex service account secret invalid
  - Vertex service account ready

Acceptance criteria:

- Gemini can run in environments where API keys are disallowed.
- The app no longer implies that a Gemini API key is mandatory.
- Streamlit Community Cloud has a supported path even without native Google-hosted ADC.

## Phase 5: Error Handling and Retries

Files:

- `conversation/client.py`

Anthropic already retries transient failures. Mirror that behavior for Gemini:

- Retry transient rate-limit / overloaded / 5xx cases
- Retry connection errors
- Reset `last_response` before retry
- Raise the last exception on final failure

Keep the retry contract aligned across providers so the app experience is consistent.

Acceptance criteria:

- Provider switching does not silently weaken resilience.

## Phase 6: Observability and Logging

Files:

- `conversation/flow.py`
- `conversation/tracing.py`
- `interface/debug_log.py`
- `interface/app.py`

Changes:

- Add provider and model metadata to any trace/generation metadata we control.
- Surface active provider/model in admin debug areas if present.
- Keep prompt logging provider-agnostic.

Recommended metadata additions:

- `llm_provider`
- `llm_model`

Acceptance criteria:

- When comparing Claude vs Gemini behavior, traces make it obvious which provider produced each answer.

## Phase 7: Tests

Files:

- new: `tests/test_conversation_client.py`
- update existing prompt/flow tests if needed

Test layers:

### Unit tests for provider mapping

- Anthropic client still receives the same payload shape as before
- Gemini adapter maps:
  - `assistant` -> `model`
  - `system` -> `system_instruction`
  - ordered text turns into `parts`

### Behavioral tests for client facade

- `MacroToolClient(provider="anthropic")` delegates correctly
- `MacroToolClient(provider="gemini")` delegates correctly
- `last_response` is populated in both cases

### Error-path tests

- transient Gemini error retries
- final Gemini error bubbles up
- invalid provider raises a clear configuration error

### Regression tests

- existing follow-up prompt tests should remain unchanged
- no engine tests should need provider awareness

Acceptance criteria:

- We can refactor provider internals without risking prompt/history regressions.

## Phase 8: Manual Validation

Suggested manual checks once implemented:

1. Start app with `LLM_PROVIDER=anthropic`
2. Confirm no behavior regression
3. Start app with `LLM_PROVIDER=gemini`
4. Run recommend-mode intake through `ConversationFlow`
5. Ask follow-up questions:
   - `Why this over vanilla?`
   - `Why not digital?`
   - `Why this variant?`
   - `What would change the recommendation?`
6. Compare:
   - tag extraction reliability
   - formatting cleanliness
   - willingness to stay qualitative
   - tendency to leak or invent internals

## Open Design Decisions

These should be resolved before coding or during Phase 2.

### 1. Streaming now or later?

Recommendation:

- Start with the simplest reliable Gemini path.
- If SDK streaming is straightforward, implement it now.
- If not, ship non-streaming first and keep the same facade.

### 2. One model per provider or user-selectable models?

Recommendation:

- Add provider-level model env vars now.
- Do not add a UI model picker yet.
- Keep app UX focused on provider switching only.

### 3. Should intake parsing be provider-specific?

Recommendation:

- No.
- Keep prompts identical first.
- Only add provider-specific prompt tuning if Gemini struggles with `[VIEW: {...}]` extraction or follow-up discipline.

### 4. Do we need a provider-neutral file split now?

Recommendation:

- Yes, but keep it light.
- Either:
  - one `conversation/client.py` with two small internal classes, or
  - `conversation/providers/anthropic_client.py` and `conversation/providers/gemini_client.py`

For the current codebase size, a single file with two internal adapters is fine.

### 5. Which Gemini auth mode is primary?

Recommendation:

- Treat Vertex AI + ADC as the primary path for this project.
- Keep Gemini Developer API key mode as an optional fallback for non-enterprise environments.

## Risks

### 1. Tag extraction drift

Risk:

- Gemini may be less consistent than Claude at emitting the exact `[VIEW: {...}]` tag format.

Mitigation:

- Keep the existing parser.
- Add focused intake tests with mocked Gemini responses.
- If needed, tighten the intake prompt rather than branching logic first.

### 2. Output feel differs even with the same context

Risk:

- Gemini may answer follow-up questions more expansively, less cautiously, or with different formatting.

Mitigation:

- Use the existing explanation pack and “keep it qualitative” instructions as the anchor.
- Compare outputs manually before changing prompts.

### 3. SDK response shape changes

Risk:

- The Google SDK is newer in this codebase and may have different streaming and response object behavior than Anthropic.

Mitigation:

- Isolate all SDK-specific parsing in `GeminiProviderClient`.
- Keep the rest of the app working against `last_response` and streamed text only.

### 4. Deployment dependency reinstall

Risk:

- Streamlit Cloud requires a `pyproject.toml` version bump for Python dependency changes.

Mitigation:

- Bump package version when the dependency is added.
- Treat provider addition as a Python dependency change, not just a config change.

## Concrete File Checklist

- `pyproject.toml`
  - add `google-genai`
  - bump version
- `conversation/client.py`
  - provider facade
  - Anthropic adapter
  - Gemini adapter
- `conversation/flow.py`
  - accept provider/model config
- `interface/app.py`
  - provider-aware secrets/env/config
  - provider-aware sidebar status
  - remove direct Anthropic client mutation
- `tests/test_conversation_client.py`
  - new provider mapping, retry, and Vertex ADC config tests
- optional docs:
  - `README.md`
  - deployment notes

## Recommended Order of Work

1. Add config/env plumbing and dependency declaration
2. Refactor `MacroToolClient` into a provider facade
3. Add Vertex AI / ADC Gemini mode alongside optional Developer API key mode
4. Wire `ConversationFlow` and `_make_flow()` to pass provider/model/key explicitly
5. Implement Gemini streaming path
6. Add unit tests for provider translation, retries, and Vertex config handling
7. Run focused manual follow-up tests on Gemini
8. Tune prompts only if the quality gap is real

## Definition of Done

We are done when:

- MacroTool can run with either `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=gemini`
- Gemini can run with either Vertex AI ADC config or Developer API key config
- The same `ConversationFlow` code works unchanged above the client layer
- Follow-up prompt wiring still includes the explanation pack
- Provider/model are visible in logs or traces
- Focused tests pass
- We have manually compared a handful of “why this / why not that / why this variant” follow-up answers across providers

## Sources

- Google Gemini API libraries: [ai.google.dev/gemini-api/docs/libraries](https://ai.google.dev/gemini-api/docs/libraries)
- Google Gemini API quickstart: [ai.google.dev/gemini-api/docs/quickstart](https://ai.google.dev/gemini-api/docs/quickstart)
- Google Gemini text generation and system instructions: [ai.google.dev/gemini-api/docs/text-generation](https://ai.google.dev/gemini-api/docs/text-generation)
