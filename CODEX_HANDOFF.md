Current branch

- Worktree: `/Users/ash/Documents/Coding work/MacroTool/.claude/worktrees/charming-leavitt-47b984`
- Branch: `claude/charming-leavitt-47b984`
- Latest pushed commit: `1fc1466` — `Add advisor chat panel below structure evaluation`

Current goal

Evaluate and improve the quality of the advisor chat responses. The Gemini/Vertex pipeline is live and functional. The next work is qualitative: assess whether the explanation pack gives the LLM enough signal to answer "why this structure?" and "why not X?" questions with real depth, and improve the system prompt and/or pack content accordingly.

Important: the app currently makes NO intake LLM calls

- `_submit_structured_view()` in `interface/app.py` calls `flow._run_engines()` (quant engine only) and returns structured output directly.
- `flow.advance()` is never called anywhere in the app.
- The LLM is only used in the advisor chat panel (below the structure evaluation output).
- The "Follow-up prompt wiring" noted in earlier docs refers to code in `conversation/` that was built but is not wired to the intake path.

What is done and live

**Gemini/Vertex provider**
- `conversation/client.py` — provider-aware facade with `AnthropicProviderClient` and `GeminiProviderClient`
- `conversation/flow.py` — accepts `provider`, `model`, `credentials` config
- `interface/app.py` — reads `LLM_PROVIDER`, model, and Vertex config from Streamlit secrets/env; shows provider-specific sidebar status
- `pyproject.toml` and `requirements.txt` — both include `google-genai>=1.40.0`
- `uv.lock` — updated to include `google-genai` (was missing, caused silent install failure on Streamlit Cloud)
- Service account credentials fix — `Credentials.from_service_account_info()` now passes `scopes=["https://www.googleapis.com/auth/cloud-platform"]` (missing scope caused OAuth error)

**Advisor chat panel**
- Added to `interface/app.py` below the structure evaluation output
- Only shown in recommend mode when `flow.explanation_pack_context` is available
- Multi-turn: full conversation history sent on every API call
- System prompt embeds the full rendered explanation pack (~1,300 tokens; well within Gemini 2.5 Flash 1M context window)
- Clears on "New view"

**Sidebar test button**
- "Test LLM connection" button appears in the sidebar when Gemini is configured and ready
- Makes a minimal one-shot call and shows the raw response — useful for deployment smoke tests

**Deployment status**
- App loads cleanly on branch `claude/charming-leavitt-47b984`
- Sidebar shows "Vertex service account ready"
- Advisor chat is functional end-to-end

Deployment gotchas (do not forget)

- `uv.lock` must be kept in sync with `pyproject.toml`. A stale lockfile silently overrides `requirements.txt` — run `uv lock` and commit after any dependency change.
- `LLM_PROVIDER` and all Gemini/Vertex secrets must be top-level TOML keys, not under `[auth.google]`.
- Service account credentials require `scopes=["https://www.googleapis.com/auth/cloud-platform"]` — without this the OAuth token request fails with `invalid_scope`.
- Python source changes require a `pyproject.toml` version bump to trigger Streamlit Cloud package reinstall.

Known-good Streamlit secrets shape

```toml
LLM_PROVIDER = "gemini"
GEMINI_MODEL = "gemini-2.5-flash"
GOOGLE_GENAI_USE_VERTEXAI = "true"
GOOGLE_CLOUD_PROJECT = "macrotool"
GOOGLE_CLOUD_LOCATION = "us-central1"

[auth]
redirect_uri = "https://macrotool-claude.streamlit.app/~/+/oauth2callback"
cookie_secret = "..."

[auth.google]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[gcp_service_account]
type = "service_account"
project_id = "macrotool"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\nREAL_KEY_LINE_1\nREAL_KEY_LINE_2\n-----END PRIVATE KEY-----\n"
client_email = "macrotool-streamlit@macrotool.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```

Focused verification run

```
'/Users/ash/Documents/Coding work/MacroTool/.venv/bin/python' -m pytest tests/test_conversation_client.py tests/test_comparator.py tests/test_explanation_context.py tests/test_followup_prompt.py
```

Files most relevant to the current task

- `interface/app.py` — chat panel is near the bottom of the Trade View page render block; search for "Advisor chat"
- `conversation/explanation_context.py` — renders the explanation pack to text
- `knowledge_engine/comparator.py` — builds the explanation pack from engine outputs
- `conversation/flow.py` — `_build_explanation_pack_context()` wires engine → pack → rendered string
- `conversation/client.py` — `GeminiProviderClient`
- `GEMINI_PROVIDER_PLAN.md` — full provider implementation history (now complete)
- `COMPARATOR_PACK_PLAN.md` — comparator design decisions and phase log

Recommended next step

- Evaluate response quality: submit a trade view, ask "why this structure?", "why not vanilla?", "why this variant?" and review the depth and accuracy of the answers.
- Identify gaps: is the explanation pack missing context the LLM needs? Is the system prompt too loose or too tight?
- Improve accordingly: options include enriching the pack content, tightening the system prompt, or adding scenario-level data to the context.
