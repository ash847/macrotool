# MacroTool

EM FX trade structuring and sizing tool for macro PMs.

The app takes a PM view in plain English, computes the market state and structure selection deterministically, then uses the LLM only to narrate the pre-computed numbers.

## Quick Start

```bash
uv sync --extra dev
.venv/bin/streamlit run interface/app.py
```

Open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

The app now expects Google OIDC auth and server-side secrets. For local or deployed use, configure Streamlit secrets before opening the app.

Minimum runtime secrets:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
SUPABASE_URL = "..."
SUPABASE_ANON_KEY = "..."
SUPABASE_SERVICE_KEY = "..."
admin_emails = ["ash@fund.com"]

[auth]
redirect_uri = "https://<your-app>.streamlit.app/~/+/oauth2callback"
cookie_secret = "<openssl rand -hex 32>"

[auth.google]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

## Useful Commands

```bash
.venv/bin/streamlit run interface/app.py
.venv/bin/python demo.py
.venv/bin/python demo.py --pair USDTRY --direction base_higher --horizon-days 60
.venv/bin/python -m pytest
```

`demo.py` runs the full deterministic pipeline without calling the LLM, so it is the fastest smoke test for the quant and rule-engine layers.

## Project Layout

```text
data/              Pydantic market snapshot models and JSON snapshot loader
analytics/         Pure quant computation: MarketState, distributions, scenario logic
pricing/           Option pricing, forwards interpolation, scenario matrices
knowledge/         JSON rulebase and tunable defaults
knowledge_engine/  Structure scoring, sizing, critique, conventions
config/            Layered config and session override support
conversation/      LLM flow, prompt assembly, tracing
interface/         Streamlit app, charts, debug log, Supabase logger
tests/             Unit test suite
```

## Packaging

This repo uses top-level Python packages rather than a single `macrotool/` package directory. Hatch is configured explicitly in `pyproject.toml` so `uv sync` can build the editable package and include runtime JSON assets from `knowledge/`.

Commit `uv.lock` after dependency changes. For a clean install:

```bash
uv sync --extra dev
```

## Deployment

GitHub repository: `ash847/macrotool`

Streamlit Community Cloud redeploys from `main`. For Python source or dependency changes, bump the `pyproject.toml` version so Streamlit Cloud performs a fresh package reinstall. JSON-only changes in `knowledge/` deploy without a version bump.

Runtime secrets:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
SUPABASE_URL = "..."
SUPABASE_ANON_KEY = "..."
SUPABASE_SERVICE_KEY = "..."
admin_emails = ["ash@fund.com"]

[auth]
redirect_uri = "https://<your-app>.streamlit.app/~/+/oauth2callback"
cookie_secret = "<openssl rand -hex 32>"

[auth.google]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Auth is fail-closed: without the `[auth]` block, the app stops at startup and shows an authentication configuration error.

`SUPABASE_SERVICE_KEY` is used server-side for app writes, engine config reads, and admin-only reads/writes.
`SUPABASE_ANON_KEY` is retained only for direct REST smoke tests and Security Advisor checks.

Only emails listed in `admin_emails` can access admin pages (`Market Data`, `Structure Selection`, `Context Rules`, `Query log`). All other authenticated users see `Trade View` only.
