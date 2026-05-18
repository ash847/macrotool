Current state

- Branch: `main`
- Latest pushed commit: `d496373` — `Refactor app.py into separate interface modules`
- App is live on Streamlit Cloud, Gemini/Vertex provider active

Current goal

Evaluate and improve the quality of the advisor chat responses. The Gemini/Vertex pipeline is live and functional. The next work is qualitative: assess whether the explanation pack gives the LLM enough signal to answer "why this structure?" and "why not X?" questions with real depth, and improve the system prompt and/or pack content accordingly.

Important: the app currently makes NO intake LLM calls

- `_submit_structured_view()` in `interface/app.py` calls `flow._run_engines()` (quant engine only) and returns structured output directly.
- `flow.advance()` is never called anywhere in the app.
- The LLM is only used in the advisor chat panel (below the structure evaluation output).

What is done and live

**Interface module split**
`interface/app.py` (1,029 lines) is now a thin orchestration shell. Three modules extracted to avoid merge conflicts during parallel branch work:
- `interface/llm_config.py` — LLM provider config functions (`get_llm_provider`, `get_provider_model`, `gemini_status`, `get_gemini_vertex_credentials`, etc.)
- `interface/structure_eval.py` — shared helpers (`fmt_ccy`, `variant_label_with_strikes`, `LINEAR_NOTIONAL`, `target_price`) + `render_structure_variants()` + `render_structure_evaluation()`
- `interface/advisor_chat.py` — `render_advisor_chat()`

**Gemini/Vertex provider**
- `conversation/client.py` — provider-aware facade with `AnthropicProviderClient` and `GeminiProviderClient`
- `conversation/flow.py` — accepts `provider`, `model`, `credentials` config
- `interface/llm_config.py` — reads `LLM_PROVIDER`, model, and Vertex config from Streamlit secrets/env; exposes sidebar status helpers
- `pyproject.toml` and `requirements.txt` — both include `google-genai>=1.40.0`
- `uv.lock` — updated to include `google-genai`
- Service account credentials — `Credentials.from_service_account_info()` passes `scopes=["https://www.googleapis.com/auth/cloud-platform"]`

**Advisor chat panel**
- In `interface/advisor_chat.py`, rendered below the structure evaluation output
- Only shown when `flow.explanation_pack_context` is available
- Multi-turn: full conversation history sent on every API call
- System prompt embeds the full rendered explanation pack (~1,300 tokens)
- Clears on "New view"
- Restricted to pack-only answers for structure/comparison/payoff questions; general knowledge fallback requires explicit "This is outside my expertise. However..." prefix

**Explanation pack / wing risk**
- `conversation/explanation_context.py` — renders explanation pack; includes "Wing risk (chosen structure)" section with per-leg tail/cap/knockout annotations
- `knowledge_engine/comparator.py` — `RecommendationExplanationPack` has `is_call: bool`; set from `market_state.put_call == "Call"`

**Structure evaluation**
- Flat ranked list of ALL variants across ALL structures, sorted by descending PM overlay weighted P&L (currency amount)
- Expander titles start with structure type name

**Sidebar test button**
- "Test LLM connection" shown in sidebar when Gemini is configured and ready

Deployment gotchas (do not forget)

- `uv.lock` must be kept in sync with `pyproject.toml`. A stale lockfile silently overrides `requirements.txt` — run `uv lock` and commit after any dependency change.
- `LLM_PROVIDER = "gemini"` must be set in Streamlit secrets — default falls back to `"anthropic"`.
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

Key files

- `interface/app.py` — thin shell; Trade View page orchestration
- `interface/advisor_chat.py` — chat panel; search for `render_advisor_chat`
- `interface/structure_eval.py` — structure evaluation rendering
- `interface/llm_config.py` — LLM provider config
- `conversation/explanation_context.py` — renders explanation pack to text
- `knowledge_engine/comparator.py` — builds explanation pack from engine outputs
- `conversation/flow.py` — `_build_explanation_pack_context()` wires engine → pack → rendered string
- `conversation/client.py` — `GeminiProviderClient`

Verification

```
.venv/bin/python -m pytest   # 338 tests
```

Recommended next step

Evaluate response quality: submit a trade view, ask "why this structure?", "why not vanilla?", "why this variant?" and assess depth and accuracy. Identify gaps in the explanation pack and improve pack content or system prompt accordingly.
