Current state

- Branch: `main`
- Latest pushed commit: `a694d9c` — `Enrich explanation pack: trim variants, add context rationale and scenario definitions`
- App is live on Streamlit Cloud, Gemini/Vertex provider active

Current goal

Evaluate advisor chat response quality with the enriched explanation pack. The pack now contains active scenario weighting rationale and scenario column definitions, and the variant ranking is trimmed to a focused set. Assess whether the LLM can now answer "why this weighting?", "what does the flat scenario mean?", and "why not vanilla?" with real depth.

Important: the app currently makes NO intake LLM calls

- `_submit_structured_view()` in `interface/app.py` calls `flow._run_engines()` (quant engine only) and returns structured output directly.
- `flow.advance()` is never called anywhere in the app.
- The LLM is only used in the advisor chat panel (below the structure evaluation output).

What is done and live

**Explanation pack — current shape (~1,800 tokens for typical trade)**
- `conversation/explanation_context.py` — renderer; `knowledge_engine/comparator.py` — data model
- Sections in order:
  1. Chosen structure + recommendation basis
  2. Active scenario weighting — fired context id + full rationale comment (base + any overlays)
  3. Variant ranking — top 5 globally + best variant per structure type not in top 5, tagged `[best of type]`
  4. Wing risk (chosen structure) — per-leg tail/cap/knockout annotations
  5. Summary / Construction / Risks reasons
  6. Scenario column definitions — plain-English description of each grid column (F, t%→K, K, K+½σ, −½σ, −1σ, Δvol), loaded from `knowledge/defaults/scenario_definitions.json`
  7. Variant comparisons (top vs next 5)
  8. Structure comparisons (chosen vs key challengers)
  9. Unavailable comparisons
  10. Disclosure constraints
- `RecommendationExplanationPack` fields: `active_context_ids`, `active_context_comments` — populated by re-running `compute_family_weights` inside `build_recommendation_pack`
- Scenario column descriptions live in `scenario_definitions.json` under `"scenario_column_descriptions"` — tunable without code changes

**Interface module split**
`interface/app.py` is a thin orchestration shell. Extracted modules:
- `interface/llm_config.py` — LLM provider config (`get_llm_provider`, `gemini_status`, `get_gemini_vertex_credentials`, etc.)
- `interface/structure_eval.py` — shared helpers (`fmt_ccy`, `variant_label_with_strikes`, `LINEAR_NOTIONAL`, `target_price`) + `render_structure_variants()` + `render_structure_evaluation()`
- `interface/advisor_chat.py` — `render_advisor_chat()`

**Advisor chat panel**
- In `interface/advisor_chat.py`, shown below structure evaluation when explanation pack is available
- Multi-turn with full conversation history; clears on Trade View nav (which now also resets the flow)
- System prompt embeds the full rendered explanation pack
- Restricted to pack-only for structure/comparison/payoff questions; out-of-scope questions require "This is outside my expertise. However..." prefix

**Sidebar UX**
- Trade View nav button resets flow to landing page (replaces old "New view" button)
- "New view" button and "Pair reference" expander removed
- Risk/Reward section above LLM/Supabase status; slider in bordered container
- "Test LLM connection" button when Gemini is ready

**Gemini/Vertex provider**
- `conversation/client.py` — provider-aware facade; `conversation/flow.py` — accepts provider/model/credentials
- Service account: `Credentials.from_service_account_info()` with `scopes=["https://www.googleapis.com/auth/cloud-platform"]`
- `LLM_PROVIDER = "gemini"` must be set in Streamlit secrets — default falls back to `"anthropic"`

Deployment gotchas (do not forget)

- `uv.lock` must be kept in sync with `pyproject.toml` — run `uv lock` and commit after any dependency change.
- `LLM_PROVIDER = "gemini"` must be in Streamlit secrets.
- Service account credentials require the cloud-platform OAuth scope.
- Python source changes require a `pyproject.toml` version bump for Streamlit Cloud package reinstall.
- JSON file changes (`scenario_definitions.json`, `affinity_scores.json`, etc.) deploy immediately without a version bump.

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
- `interface/advisor_chat.py` — chat panel
- `interface/structure_eval.py` — structure evaluation rendering
- `interface/llm_config.py` — LLM provider config
- `conversation/explanation_context.py` — renders explanation pack; `_select_display_variants()` trims ranking
- `knowledge_engine/comparator.py` — `build_recommendation_pack()` builds and populates the pack
- `knowledge/defaults/scenario_definitions.json` — scenario weighting contexts + `scenario_column_descriptions`
- `conversation/flow.py` — `_build_explanation_pack_context()` wires engine → pack → rendered string

Verification

```
.venv/bin/python -m pytest   # 338 tests
```

Recommended next step

Evaluate response quality with the enriched pack: ask "why was this scenario weighting applied?", "what does the flat scenario mean?", "why not 1x1 spread?". Identify whether the pack content is now sufficient or whether structure-level scoring rationale (why each structure was gated/ranked) should be added next.
