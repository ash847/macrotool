# MacroTool

EM FX trade structuring and sizing tool for macro PMs.

The Agent page supports conversational structuring: Python computes prices, rankings,
risk and sizing; the LLM interprets requests and narrates engine results. The separate
Trade View screen retains a structured, silent engine path for admin testing.

## Current Status - 28 October 2026

The previous source release is **v0.2.58, commit `746a75d`, pushed to
`origin/agentic-workflow`**. This is a source-release checkpoint, not confirmation
that a cloud app has deployed it. User acceptance testing is pending.

- Shared entry pricing and fixed-contract valuation cover vanillas, 1x1/1x1.5/1x2
  spreads, butterflies, seagulls and explicitly requested flexible ratio spreads.
- Construction resolves strikes and quantities once. Scenario/Kelly calculations
  retain those quantities, including solved seagull wings. Exact delta inversion
  and strike-specific smile pricing are preserved; ATMF is not 50 delta.
- Ask the Agent for `1x1.3 spread ATMF/25d` for a flexible ratio with recommended
  sizing under the session's existing fixed-loss or Kelly regime. Ratios R >= 1
  are supported. This does not expand the default recommendation menu or Kelly dropdown.
- Bare leg numbers mean **strikes**; deltas require `d`, `delta` or `Δ` (public
  request range 10-50 delta). `1x1.3 5.00/5.20` retains literal strikes. Direction
  comes from the session; implausible strikes and reversed spreads are refused,
  not silently reinterpreted. The ratio token is separate from the leg numbers.
- The local [MCP demo](mcp_server/README.md) and shared exchange telemetry are included.
  MCP is not a production external API or an app authentication replacement.

Verification: latest full suite **1,482 passed, seven skipped**, including the
Greeks tests. The optional MCP suite previously passed 13 tests separately at
the integration checkpoint; it was not rerun for T23. No live cloud/UI check.

Next product work: user testing of vanilla-leg inception Greeks (T23), then
selected-trade expiry payoff/net P&L charts (T21) and broader custom packages (T22).

Release v0.2.59 adds custom-trade comparison before charting.
Price a trade, then ask: "Compare my 1x1.3 ATMF/25d with recommendations
1 and 2 and explain the contributors and detractors." The engine retains its
full scenario evidence using the recommendation run's exact grid and weights.
Comparison separates per-unit performance from actual-sized performance and
identifies scenarios driving the difference. Raw scores/weights stay private;
signed attribution shares use all absolute contributions, not net profit.
Stale contexts and failed scenario prices are unavailable, not fabricated zeros.
The existing shortlist and eligibility rules are not changed. Custom seagulls
now use the same stop input as the pack for existing sizing rules. Accept this
comparison slice and T23 before moving to T21; no live API/cloud test yet.
Local v0.2.59 full suite: **1,499 passed, seven skipped** (187.06 seconds),
including 17 dedicated custom-comparison tests. `git diff --check` passed.

Version 0.2.58 adds T23 vanilla-leg inception Greeks, ahead of charts at the user's
request. Ask the agent for delta/gamma/vega on an exact recommendation or custom
trade. No automatic UI display is added. Delta is in base units; gamma is base
delta change per 1% spot rise; vega is base currency per one vol point, converted
at fixed inception spot. Strikes, ratios, notional, rates and entry leg vols stay
fixed. Digital/ERKO/linear Greeks are unavailable. Cloud acceptance testing
remains pending.
Final local suite: **1,482 passed, seven skipped** (165.76 seconds), including
58 new Greeks cases. Agent routing is tested with a fake model, not a live API.
Specialized product migration remains incomplete (T19). Launch controls remain
separate priorities, not implied complete by engine progress.

Documentation map:
- [Active implementation checklist](BETA_IMPLEMENTATION_TODO.md): scope, tags, urgency and evidence.
- [Beta roadmap](BETA_READINESS_PLAN.md): launch/security/legal/data gates.
- [Current engine handoff](PRODUCT_ENGINE_HANDOFF_CODEX.md): checkpoint and resume instructions.
- [Flexible-package specification](FLEXIBLE_PACKAGE_SPEC_CODEX.md): implemented ratios versus proposed arbitrary packages.

## Quick Start

```bash
uv sync --extra dev
.venv/bin/streamlit run interface/app.py
```

Open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

The app expects Auth0 email magic-link login and server-side secrets. Configure
Streamlit secrets before opening it; Google login is not offered. The authenticated
email must also be in `allowed_emails`; admin status alone does not grant admission.

Minimum runtime secrets:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
SUPABASE_URL = "..."
SUPABASE_ANON_KEY = "..."
SUPABASE_SERVICE_KEY = "..."
admin_emails = ["ash@fund.com"]
allowed_emails = ["ash@fund.com", "tester@fund.com"]

[auth]
redirect_uri = "https://<your-app>.streamlit.app/~/+/oauth2callback"
cookie_secret = "<openssl rand -hex 32>"

[auth.auth0]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://<your-auth0-domain>/.well-known/openid-configuration"
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
agentic/           Tool-driven Agent loop, parsing, rendering and telemetry
workspace/         Saved user/chat workspace persistence
mcp_server/        Optional local MCP demo and evaluation tools
interface/         Streamlit app, charts, debug log, Supabase logger
tests/             Unit test suite
```

## Packaging

This repo uses top-level Python packages rather than a single `macrotool/` package directory. Hatch is configured explicitly in `pyproject.toml` so `uv sync` can build the editable package and include runtime JSON assets from `knowledge/`.

Commit `uv.lock` after dependency changes. For a clean install:

```bash
uv sync --extra dev
```

## PM Preferences

The intake form supports three PM preference inputs:

- `Primary objective`: `Balanced`, `Keep cost low`, `Hold up if the path is slow/noisy`, `Keep risk clean`
- `Structure constraint`: `No restriction`, `Avoid capped structures`, `Avoid complex structures`, `Avoid tail-risky structures`
- `Trade management style`: `Standard hold`, `May monetise early`, `Need defendable mark-to-market`

`Structure constraint` now works in two layers:

- Soft preference scores still live in `knowledge/defaults/affinity_scores.json` as the `structure_constraint` dimension.
- Hard exclusions are also applied in `knowledge_engine/structure_scorer.py` for structures that should never survive ranking under some constraints.

Current hard exclusions:

- `Avoid complex structures`: excludes `seagull`, `rko`, `european_rko`, `european_digital_rko`
- `Avoid capped structures`: currently a vanilla-only hard preference.
- `Avoid tail-risky structures`: final eligibility requires a verified finite
  contractual loss bound; unknown/unbounded bounds are excluded. Do not infer
  safety from the sizing proxy or a family name. Construction/tail classification
  uses actual legs and cross-checks catalog declarations; disagreements stay unknown.

## Kelly Screen

Admins can access the separate `Kelly Sizing` page. Testers currently see Agent and About.

Current modes:

- `Standalone`: pair-and-horizon driven. The user picks a supported currency pair and tenor, the page derives the current forward and ATM vol from the live snapshot, and the PM elicits a subjective distribution to price a single vanilla option.
- `From Trade Rec`: session-linked. If a live `Trade View` recommendation exists, the Kelly page reads the top ranked concrete variants from that session, shows up to 10 of them in a dropdown, and lets the PM size one of those recommended trades against their subjective distribution.

Current implementation notes:

- `Standalone` no longer uses a free-form fixture or manual `F/σ/T` baseline in normal operation; it is anchored to the current snapshot and tenor interpolation.
- `From Trade Rec` operates on concrete variants, not abstract structure families, because Kelly needs a fully specified payoff.
- The payoff bridge currently supports the recommended structure variants already produced by `analytics.structure_pricer.price_variants()`, including vanilla, spreads, ratio spreads, seagulls, digitals, and RKOs.

Streamlit state lessons from the recent Kelly build:

- Anchor and bucket widgets must be read back from `st.session_state` after rendering. Relying on transient widget return values made the chart and edge section lag or ignore `+/-` edits.
- Baseline reseeding must happen only when the actual market context changes, such as switching Kelly mode, changing pair/tenor, or selecting a different Trade Rec candidate. Reseeding on every rerun causes the anchor grid to snap back toward baseline and makes the `+/-` buttons feel unstable.

## Deployment

GitHub repository: `ash847/macrotool`

Verify each Streamlit app's configured branch and deployed version in its hosting
settings. This work was pushed to `agentic-workflow`, not to `main`; a push is not
proof of a completed deployment. For Python source or dependency changes, bump
`pyproject.toml` to trigger package reinstall. JSON-only changes do not require a bump.

Current tested source version: `0.2.59`, custom-trade comparison release on
`agentic-workflow`. User acceptance testing is on cloud, not a local app;
verify the deployed version before testing. A successful push alone does not
confirm Streamlit deployment.

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
allowed_emails = ["ash@fund.com", "tester@fund.com"]

[auth]
redirect_uri = "https://<your-app>.streamlit.app/~/+/oauth2callback"
cookie_secret = "<openssl rand -hex 32>"

[auth.auth0]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://<your-auth0-domain>/.well-known/openid-configuration"
```

Auth is fail-closed without complete `[auth]` and `[auth.auth0]` settings. Admission
checks the configured issuer, verified email, email magic-link subject and
`allowed_emails`. Keep the local `MACROTOOL_DEV` bypass disabled in deployments.

`SUPABASE_SERVICE_KEY` is used server-side for app writes, engine config reads, and admin-only reads/writes.
`SUPABASE_ANON_KEY` is retained only for direct REST smoke tests and Security Advisor checks.

Admitted users in `admin_emails` can access admin/test pages, including Trade View,
Kelly Sizing, Market Data, Structure Selection, Scenario Weightings and Query log.
Other admitted users see **Agent and About**. Live provider/database permissions
and server-side ownership checks still require the roadmap's separate audit.
