# Codex: beta readiness code review to-do list

> **Historical source review; superseded as an implementation checklist.**
> Use [BETA_IMPLEMENTATION_TODO.md](BETA_IMPLEMENTATION_TODO.md), which combines
> every Claude and Codex point with tags, urgency and staged completion gates.
> The original review below is retained for provenance, not separate task tracking.

Author: **Codex (OpenAI coding agent)**  
Review date: 13 October 2026  
Reviewed branch/commit: agentic-workflow at bf049fc (v0.2.49)  
Companion roadmap: [BETA_READINESS_PLAN.md](BETA_READINESS_PLAN.md)

## Scope and status

This is Codex's recommendation list, not a record of approved changes or completed
work. No application code was changed during the review. Each item below needs an
owner and evidence before it is marked complete.

Keep the current architecture: Python calculates numbers, the agent calls defined
tools, and Python renders the tables. Do not rewrite the UI or split the application
into many services merely to prepare for beta.

Verification: the non-live tests/ run produced **1,077 passes and 6 failures**.
Failures covered old login expectations, snapshot assumptions, scoring expectations
and smile-pricing expectations. Live database permissions, deployment settings,
provider controls and legal/data contracts were not audited. Test results alone
are not a production-readiness sign-off.

## Prioritised checklist

- [ ] **1. Before wider beta: strengthen database protection.**
  Current choice: logging migrations disable row-level security; workspace reads
  check ownership, but writes largely trust supplied objects.
  Change: check ownership on every write and deny direct public access to private
  tables. Audit actual deployed permissions, not just the SQL files.
  Example/risk: a mistaken conversation ID must never let one user overwrite
  another user's chat. The review does not establish that live data is exposed.
  Trade-off: more database work now, fewer security assumptions in future features.
  Done when: two-user read/write/delete denial tests and live permissions checks pass.
  Evidence: db/migrations_tester_logging.sql:23; workspace/store.py:161.

- [ ] **2. Before wider beta: centralise access, spending and job limits.**
  Current choice: approved-email checks exist, but paid calls have no durable
  per-user allowance; each submitted job starts a background thread.
  Change: check access and reserve allowance before each paid call; add bounded
  concurrency, timeouts and a global stop switch. Include retries in accounting.
  Example/risk: five browser tabs must not create five independent allowances.
  Trade-off: some requests wait or are refused; costs and load become controllable.
  Keep a separate worker service optional until measurements justify it.
  Done when: concurrent requests, revocation, retries and quota-store failures are
  tested; new paid work stops safely when access or allowance is unavailable.
  Evidence: agentic/telemetry.py:20; interface/agent_job.py:81.

- [ ] **3. Before wider beta: make uncertain risk produce caution, not maximum size.**
  Current choice: a missing or near-zero sizing-loss estimate sends notional to
  10 times W; net-credit trades also receive 10 times W.
  Change: separate premium, stress loss and contractual maximum loss throughout
  the engine, extending TradeEconomics. Require an approved policy for cases
  where the trade cannot be reliably sized.
  Example/risk: an unavailable risk estimate should not automatically mean a
  large position. A notional cap is not a guaranteed loss cap.
  Trade-off: fewer automatically sized recommendations, but clearer risk control.
  Done when: quant approves the policy, edge-case tests pass, and affected
  comparisons are rerun. Do not silently change financial behaviour in a refactor.
  Evidence: analytics/structure_pricer.py:363; analytics/trade_economics.py:10.

- [ ] **4. Before wider beta: make releases clean and repeatable.**
  Current choice: the wheel configuration omits agentic and workspace, although
  the application imports them. Dependency definitions disagree; no CI workflow
  was found in this checkout.
  Change: include all runtime packages, reconcile dependency files, add automated
  checks, and test installation outside the source checkout. Establish deliberate
  staging-to-production promotion and rollback.
  Example/risk: running from the checkout can hide missing packages until a clean
  installation or hosting move.
  Trade-off: more release preparation, fewer deployment surprises.
  Done when: a clean installation starts successfully, release checks pass, and
  all six observed test failures are resolved or explicitly dispositioned with
  an owner and expiry. Do not simply accept a permanently failing suite.
  Evidence: pyproject.toml:27; requirements.txt; uv.lock; tests/.

- [ ] **5. Before wider beta: freeze the settings used for each calculation.**
  Current choice: the engine reaches into UI/database code to fetch mutable
  settings; cached packs do not identify their configuration version.
  Change: resolve a versioned settings bundle outside the engine and pass it in.
  Save code, config, data and relevant prompt/model identities with each result;
  include the relevant identities in cache keys.
  Example/risk: yesterday's recommendation must remain explainable after today's
  scenario-weight change.
  Trade-off: updates need deliberate activation rather than changing behaviour
  everywhere immediately.
  Done when: an old result can be reproduced from its recorded inputs and a
  config update cannot silently mix old cached results with new calculations.
  Evidence: knowledge_engine/scenario_weighter.py:115; agentic/session.py:46;
  workspace/service.py:194.

- [ ] **6. Before wider beta: save a completed turn reliably.**
  Current choice: idea, version, conversation, link and reply are separate writes.
  Version numbers are calculated from the last saved version.
  Change: use a database transaction, a unique request ID and conflict checks.
  Keep the session-only fallback explicitly labelled as not saved.
  Example/risk: an outage halfway through saving can leave a partial record;
  two tabs can compete for the next version number.
  Trade-off: a more demanding save implementation, but safe retries and more
  dependable history.
  Done when: injected failures and simultaneous writes leave either a complete
  saved turn or a clear retryable failure, without duplicates or partial history.
  Evidence: workspace/service.py:170; workspace/service.py:210.

- [ ] **7. Before wider beta: distinguish optional missing output from failed pricing.**
  Current choice: broad exception handling can replace a failed smile with flat
  volatility or failed ranking with fallback output. Some labels exist, but the
  original cause is lost.
  Change: return explicit statuses and reasons, such as flat-vol fallback or
  ranking failed. Decide which failures permit a recommendation.
  Example/risk: an optional chart failing is different from the model pricing a
  trade with a different volatility assumption.
  Trade-off: more visible warnings and occasional blocked results, rather than
  apparently normal results calculated by a different method.
  Done when: failure-injection tests confirm the status reaches the UI, narration
  and logs; a valid target is not blamed for an internal ranking failure.
  Evidence: agentic/standard_pack.py:341; agentic/standard_pack.py:407;
  agentic/shortlist.py:174.

- [ ] **8. Before broader use: preserve the user's original view separately.**
  Current choice: spot-percentage requests are converted correctly, then direction
  and magnitude are replaced with forward-based values.
  Change: retain the requested reference, original direction/magnitude and absolute
  target alongside the engine's forward-relative interpretation.
  Example/risk: spot 100, forward 105 and target 102 means up from spot but below
  forward. One direction field should not have to explain both meanings.
  Trade-off: more fields and migration work; clearer explanations, tail preferences
  and saved-trade behaviour. Pricing conventions need not change.
  Done when: targets between spot and forward round-trip through saving, settings
  changes and refresh, with both interpretations correctly explained.
  Evidence: agentic/tools.py:314; knowledge_engine/models.py:34;
  workspace/models.py:46.

- [ ] **9. Before wider beta, then extend for new assets: make data quality explicit.**
  Current choice: the snapshot has a headline date, but its note says some
  discount curves are older than the FX quotes.
  Change: carry component dates, sources, approved tenor ranges and validation
  status in the loaded data, not only in a separate document.
  Example/risk: fresh spot plus an old curve should not look like a fully refreshed
  market. Define when stale or mixed inputs warn and when they block new output.
  Trade-off: more ingestion work and possible restrictions, but easier feed
  integration, reproducibility and diagnosis.
  Done when: every result identifies its actual input versions; invalid imports
  cannot replace the last approved dataset.
  Evidence: data/schema.py:57; data/LATEST_FX_SNAPSHOT.md.

- [ ] **10. Before XAU/BTC/ETH: define assets and contracts explicitly.**
  Current choice: models assume FX conventions, and the rate helper defaults
  unrecognised base currencies to the USD discount curve.
  Change: define units, settlement currency, contract type, forward source,
  pricing conventions and permitted products explicitly.
  Example/risk: BTC-settled and USD-settled options must not differ only by symbol;
  gold must not inherit an unsuitable FX convention accidentally.
  Trade-off: slower first-asset onboarding, less special-case code for later assets.
  Done when: each approved asset/product passes independent unit, settlement,
  price, payoff and sizing checks; unsupported combinations are rejected.
  Evidence: pricing/forwards.py:179; data/schema.py:30.

- [ ] **11. Before many more packages: finish the product-model migration gradually.**
  Current choice: Structure and Leg objects exist, but custom pricing calculates
  a legacy variant and separately reprices a product object for the leg breakdown.
  Change: move consumers onto one authoritative priced package, following the
  existing PRODUCT_MODEL_PLAN.md and protecting each step with parity tests.
  Example/risk: fixing a package in one path should not leave its displayed legs
  or another pricing path using different logic.
  Trade-off: controlled refactoring now versus maintaining two implementations.
  Done when: the same priced object feeds totals, legs, payoff, scenarios and
  sizing for migrated products. Keep financial corrections separate from migration.
  Evidence: agentic/price_structure.py:110; analytics/product_model.py:56.

- [ ] **12. During beta preparation: simplify UI orchestration and operational records.**
  Current choice: the main Streamlit file coordinates jobs, settings, persistence
  and logging. Conversation content is duplicated across workspace and telemetry
  records; contact delivery and reactions do not form a complete support workflow.
  Change: extract those operations into small application services and keep
  Streamlit as the screen. Link feedback, errors and usage to one run ID. Define
  retention/deletion across duplicate transcripts and a durable feedback workflow.
  Example/risk: a report that a recommendation was wrong should identify the exact
  calculation without searching several logs. Deleting a chat should not leave
  an undocumented duplicate indefinitely.
  Trade-off: modest restructuring and less unrestricted trace storage; easier
  support, testing and future interfaces.
  Done when: a turn can be exercised without Streamlit, feedback links to its
  calculation version, and retention/deletion tests cover all intended stores.
  Evidence: interface/app.py:1305; interface/app.py:1055;
  interface/supabase_logger.py:118; workspace/service.py:163.

## Execution notes

Use the existing beta plan as the delivery roadmap, with this checklist as its
code-review companion. Assign owners, dependencies and completion evidence before
starting implementation. Do not treat all structural improvements as reasons to
delay the core FX pilot: asset and broad product expansion have separate gates.

Correct the roadmap's outdated statement that admission checks are absent:
interface/security.py now verifies magic-link identity and allowed_emails.
This progress does not establish live deployment compliance, durable revocation
at every paid call, or complete database isolation.

Suggested grouping: establish protection and release checks first (1, 2, 4);
resolve sizing policy in parallel (3); then improve reproducibility, saving and
failure handling (5-9, 12). Complete the relevant asset/product foundations
(10-11) before expanding those surfaces.
