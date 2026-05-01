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

The app needs an Anthropic API key for LLM-backed recommendations. Provide it either in the Streamlit sidebar or via Streamlit secrets:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
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

Runtime secrets are optional except for `ANTHROPIC_API_KEY` when using LLM-backed flows:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
```

Langfuse and Supabase are no-op safe when their secrets are missing.
