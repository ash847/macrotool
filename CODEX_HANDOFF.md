Current branch

- Worktree: `/Users/ash/Documents/Coding work/MacroTool/.claude/worktrees/charming-leavitt-47b984`
- Branch: `claude/charming-leavitt-47b984`
- Latest pushed commit: TBD (see git log)

Current goal

- Wire real LLM calls into the app flow (view extraction → validation → structure rec narration).
- Gemini/Vertex provider is configured and connectivity is confirmed — the missing piece is calling `flow.advance()` from the intake submission path.

Important: the app currently makes NO LLM calls

- `_submit_structured_view()` in `interface/app.py` calls `flow._run_engines()` (quant engine only) and returns structured output directly.
- `flow.advance()` is never called anywhere in the app.
- The LLM client is instantiated and the provider/credentials are wired up, but no text generation happens at runtime.
- The "Follow-up prompt wiring" noted below refers to code that exists in `conversation/` but is not yet connected to the UI submission path.

What is already done

- Comparator/explanation-pack work is on this branch.
- `conversation/client.py` has a provider-aware facade:
  - `AnthropicProviderClient`
  - `GeminiProviderClient`
- `conversation/flow.py` accepts provider/model/credentials config and has `advance()` implemented.
- `interface/app.py`:
  - reads `LLM_PROVIDER`, model, and Vertex config from Streamlit secrets/env
  - supports a `[gcp_service_account]` secrets block
  - shows provider-specific sidebar readiness messages
  - has a "Test LLM connection" button (sidebar, shown when Gemini ready) that makes a minimal one-shot call and shows the raw response
- `pyproject.toml` and `requirements.txt` both include `google-genai>=1.40.0`
- `uv.lock` updated to include `google-genai` (was missing, caused ModuleNotFoundError on deploy)
- Focused provider tests exist in `tests/test_conversation_client.py`

Deployment status

- App loads cleanly. Sidebar shows "Vertex service account ready".
- `uv.lock` stale-lockfile bug resolved — `google-genai` now installs correctly on Streamlit Cloud.
- "Test LLM connection" button added to sidebar for smoke testing without a full intake submission.

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

Important caveats:

- `LLM_PROVIDER` and all Gemini/Vertex settings must be top-level, not under `[auth.google]`.
- After any `pyproject.toml` dependency change, run `uv lock` and commit `uv.lock` — a stale lockfile silently overrides `requirements.txt` on Community Cloud.

Focused verification run

- `'/Users/ash/Documents/Coding work/MacroTool/.venv/bin/python' -m pytest tests/test_conversation_client.py tests/test_comparator.py tests/test_explanation_context.py tests/test_followup_prompt.py`

Files most relevant to the current task

- `conversation/client.py`
- `conversation/flow.py`
- `interface/app.py` (see `_submit_structured_view` around line 444)
- `pyproject.toml`
- `requirements.txt`
- `uv.lock`
- `tests/test_conversation_client.py`
- `GEMINI_PROVIDER_PLAN.md`

Recommended next step

- Use the "Test LLM connection" sidebar button on the live app to confirm Gemini/Vertex generates text end-to-end.
- Then wire `flow.advance()` into the intake submission path so trade views trigger real LLM narration.
  - Entry point: `_submit_structured_view()` in `interface/app.py` — after `flow._run_engines()` succeeds, call `flow.advance()` and stream the response into the conversation UI.
