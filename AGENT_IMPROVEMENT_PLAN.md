# Agent improvement plan

Status: approved Steps 1, 4, 5, 6 and the configuration-based no-tails rule
implemented locally. Step 6 automated regression validation is complete; live-chat
acceptance remains pending. Explicitly deferred work stays deferred.
Recorded: 2026-09-24.
Implementation target: the `agentic-workflow` checkout at
`/Users/ash/Documents/Coding work/agentic-workflow`.
This document is stored in the MacroTool workspace; recording it does not change
the application or approve implementation of the proposed changes.

## Objective and boundaries

Make the deterministic engine the source of financial meaning, trade attributes,
ranking, and sizing explanations. The agent narrates those results, not inferred
economics. Preserve current pricing, ranking, and sizing policy unless a separate
change is explicitly approved. Do not change historical backtest data or outputs.

Use Python for typed financial contracts and computations; JSON for tunable
policy and approved vocabulary; serializable result packs for computed values and
provenance. Do not store calculated trade facts in static knowledge JSON.

## Current-code qualifications

- The inspected agent path already forwards Kelly inputs, user identity, and
  preferences. Missing-distribution fallback is disclosed. Verify these end to
  end rather than assuming the earlier plumbing defects remain.
- Engine-authored payoff profiles exist. Extend and correct them rather than
  introduce a competing description system.
- `max_loss` currently also represents sizing proxies such as premium paid or
  loss at a stop. It must not universally be presented as contractual maximum loss.
- Linked Trade View chat rebuilds a pack. The existing seed/handoff can instead
  receive the actual shared result.
- The current ranked shortlist selects representatives per shortlisted family;
  it is not necessarily the global top five individual variants.

## Guided workflow

For each step: explain the distinction in plain language, agree the user-visible
meaning and policy, implement only after approval, then demonstrate acceptance
examples and tests before moving on. Decisions below are proposals, not approvals.

### Step 1 — Financial definitions and typed output contract

Status: implemented locally in agentic-workflow; 178 selected regression tests passed.

Files in the implementation target:
- `analytics/product_model.py`: typed financial output records.
- `analytics/product_pricer.py`, `analytics/structure_pricer.py`: populate them.
- `agentic/standard_pack.py`: carry them without reinterpretation.
- `agentic/render.py`, `agentic/agent_flow.py`: consume the explicit meanings.

Data:
- Premium: signed amount, debit/credit direction, currency, notional basis.
- Contractual maximum loss: bounded/unbounded/unknown, amount when defined,
  settlement currency, valuation basis, horizon and assumptions.
- Sizing loss measure: method, amount per unit, stop/stress assumptions.
- Loss budget: requested budget and derivation, separate from the loss measure.
- R:R metrics: name, numerator, denominator, value, status and unavailable reason.

Agreed decisions: keep premium, contractual maximum loss, sizing loss measure and
loss budget distinct. Preserve existing sizing calculations initially.
For the premium-spend method discussed here, the stop is only an intermediate
input to calculate premium spend, not an assumed exit or a separate risk denominator.
The relevant ratio is **net payout at specified target / premium paid**, with the
numerator net of entry premium. Do not call this maximum payout: it is evaluated
at the specified target, not the best possible outcome. Do not introduce target
P&L / sizing risk as a metric for this method.
Evaluate target net P&L at the engine's specified evaluation horizon, using
expiry payoff only when that horizon equals expiry. Before expiry, use the
engine's mark-to-market valuation, retaining its scenario assumptions. Label the
horizon explicitly, for example “Net P&L at target, at 3-month horizon”. Keep
numerator and denominator on a consistent currency/valuation basis and expose
the engine's conventions rather than introducing new ones in narration.
For zero-cost and net-credit trades, display this ratio as
“Not applicable — no premium outlay”. Continue to display net payout at the
specified target and separate risk information.

Acceptance: debit vanilla, debit ratio spread, credit/zero-cost structure and
stop-sized seagull receive truthful, distinct descriptions. Missing or undefined
ratios produce a supplied reason, never an invented explanation.

Implementation notes:
- Added `analytics/trade_economics.py` and an additive `economics` record on
  `PricedVariant`; the standard pack carries it with each priced variant.
- Agent renderers use the canonical net-target-return and risk labels rather than
  the ambiguous legacy gross-return/max-loss fields. Prompt constraints match.
- Target net P&L uses existing full-package scenario valuation, with explicit
  horizon and premium treatment. Legacy ratio-spread target fields may omit short
  legs; those legacy fields remain unchanged, but are not the new metric's source.
- Complex-package contractual maximum loss remains unknown pending Step 2; only
  supported non-negative long-option payoffs receive the premium loss bound.
- Fixed-loss and Kelly pricing/sizing/ranking invariance tested in both directions.
- Added `AGENT_FINANCIAL_DEFINITIONS.md` in the implementation checkout; package
  version bumped to 0.2.15. No commit, push or deployment performed.

### Step 2 — Construction-config no-tails rule; further attributes deferred

Status: approved config-based eligibility implemented; full tail geometry and
additional Greeks remain deferred, not implemented by this step.

Agreed scope revision:
- “No tails” excludes constructions that can lose more than premium paid on
  either side of spot, not merely mathematically unlimited losses.
- User approved domain-authored per-construction configuration instead of new
  risk mathematics for this eligibility decision.
- `knowledge/defaults/structure_variants.json` is the single classification
  source: `can_lose_beyond_premium` is true, false, or unknown if absent/invalid.
- Only explicit false passes. Custom requests match exact configured terms
  (not labels); unclassified requests remain unknown and cannot pass no-tails.

Implemented scope:
- Replaced the hardcoded no-tail family gate with catalog eligibility and
  individual construction filtering before ranking.
- Applied the same filter to the agent comparator, fallback, custom-price tool
  and Trade View variant/evaluation paths; expose the flag to agent narration.
- Ratio 1x1.5, ratio 1x2 and seagull constructions are flagged true; configured
  long vanillas, debit spreads, symmetric butterflies and long digital/KO
  constructions are false. Existing disabled-product gates stay unchanged.
- Unknown maximum-loss amounts from Step 1 remain separate; this flag neither
  quantifies losses nor implies unlimited loss. Pricing/sizing formulas unchanged.
- Added `CONSTRUCTION_RISK_POLICY.md` and policy tests in agentic-workflow;
  package version is 0.2.16. No commit, push or deployment performed.
- Validation: all 16 new policy tests pass. Broader regression run: 233 passed,
  one known pre-existing test deselected. The initial run reproduced a failure in
  `TestRanking.test_vanilla_wins_far_target_high_carry`; executing the original
  HEAD scorer against the same inputs/config returned the identical ranking
  (1x1 spread ahead of vanilla). The unrelated expectation/config was not changed.

The following broader geometry/attribute work is retained for future discussion,
not included in the approved configuration-only implementation:

Files:
- `knowledge_engine/payoff_profile.py`: extend existing geometry output.
- `knowledge_engine/structure_attributes.py`: attributes backed by actual trades.
- New `analytics/trade_risk.py` if additional pure risk computation is needed.
- New `knowledge/defaults/risk_vocabulary.json`: approved labels and explanations.

Data: lower/upper region behaviour, boundedness on the stated payout basis,
deterioration thresholds, net-P&L breakevens, terminal/path-dependent conditions;
supported Greeks with units and conventions, otherwise explicit unavailable status.
Scenario sensitivity tags are not substitutes for mathematical Greeks.

Decisions: define exactly what the no-tail preference excludes. Distinguish
favourable spot direction from favourable payoff: an overshoot can hurt a ratio
spread. Distinguish loss onset from the strike where payoff starts deteriorating.

Acceptance: calculations respect positive FX spot and currency conversion;
premium-inclusive breakevens are not called structural intrinsic crossings;
family labels do not determine delta or loss geometry. Avoid “Tail: None” being
misread as “no risk”. Any revised eligibility policy needs explicit approval.

### Step 3 — Canonical pack and applied-preference parity

Status: deferred by user. Focused audit completed; no Step 3 application changes
made. Defer the shared-pack refactor given the planned retirement of Trade View.

Audit results:
- 71 existing workspace, personal-weight, linked-chat seed, sizing-regime and
  agent-tool tests passed.
- Three temporary diagnostic probes reproduced: chat Refresh rebuilds from the
  session's existing snapshot without requesting the current effective snapshot;
  same-view cached requests retain old weights until explicit Refresh (which
  correctly rebuilds); linked Trade View chat signature ignores bin-only changes
  to a Kelly distribution.
- The refresh snapshot probe executed the actual UI helper with a changed-data
  provider and the real conversation service; passing the updated snapshot into
  the session made the subsequent refresh use the new spot.
- Live-data TODO: reload the current effective market snapshot before chat
  Refresh, invalidate stale packs, and rerun the freshness checks when live data
  is introduced. User explicitly deferred this while data remains static/stale.
  Do not implement now. Defer the linked-chat-only issue as well.

Files: `agentic/standard_pack.py`, `agentic/session.py`, `agentic/seed.py`,
`interface/app.py`, `interface/agent_settings_ui.py` and existing settings storage.

Data: pack ID, schema version, input fingerprint, market-data identity/as-of,
resolved configuration/weights revision, applied preferences, requested/effective
sizing method, stable trade IDs, ranking basis and candidate-universe description.
Keep computed provenance in a serializable pack, not static defaults JSON.

Decisions: linked chat uses the same result as Trade View; independently configured
chats may differ. Draft settings stay distinct from applied settings. Configuration
and data changes invalidate caches even if the user identity remains unchanged.

Acceptance: identical applied inputs produce identical trades, rankings and sizing;
unapplied edits are visibly labelled; personal-weight edits invalidate old results.

### Step 4 — Sizing trace and monetary units

Status: user approved; implemented locally in agentic-workflow; 142 selected tests
passed. Existing sizing formulas and disclosed missing-distribution
fallback are unchanged.

Implementation:
- Added `analytics/sizing_trace.py` and a `SizingTrace` on priced variants,
  populated by the actual fixed-loss/Kelly sizing branches rather than a second
  calculator.
- Captures requested/effective methods, reasons, budget and denominator, pre-cap
  and final notionals, cap and determining rule. Kelly also records bankroll,
  full-Kelly fraction, lambda and a fingerprint of the supplied distribution.
- Ranked packs attach the actual budget derivation; custom-trade agent results
  inherit the active pack's budget context and disclosed fallback reason.
- Agent render paths supply a deterministic SIZING AUDIT in full currency units,
  including for unsized/error results. Zero allocation is distinct from missing
  inputs or a failed calculation.
- Added `tests/test_sizing_trace.py` and `SIZING_EXPLANATION.md`; package version
  bumped to 0.2.17. No commit, push or deployment performed.
- Validation includes direct comparison against the original HEAD sizing kernels
  for fixed-loss/Kelly and both directions: notionals, premiums, other numerical
  variant fields, ranking and scenario scores are identical. The new trace is
  additive. Display precision avoids exposing insignificant saved-target rounding
  differences as changed conversation results; stored values retain full precision.

Files: `analytics/structure_pricer.py`, `agentic/standard_pack.py`,
`agentic/render.py`, `interface/agent_settings_ui.py`; use a shared display formatter.

Data: requested/effective method, fallback reason, capital/reference notional and
currency, budget derivation, stop, per-unit sizing measure, pre-cap notional, cap,
final notional, binding constraint; Kelly distribution identity, optimal fraction
and lambda. Distinguish valid zero size, missing inputs and calculation failure.

Decisions: whether missing Kelly inputs block sizing or allow an explicitly
disclosed fallback; confirm capital currency versus trade-notional currency and
conversion rules. No silent change to the existing fallback policy.

Acceptance: “why this notional?” is answered entirely from the trace; cap-bound and
credit trades are not said to have maximum loss equal to budget; full currency
units are stored and display scaling occurs exactly once. Test 1,000, 1m and 1bn.

### Step 5 — Compact shortlist and detail on demand

Status: user approved initial columns; implemented locally in agentic-workflow.

Implemented scope:
- Python renders rank, structure/key terms, sized notional, premium, net P&L at
  target, target return on premium, and additional-loss-beyond-premium flag.
- The table states the valuation horizon and best-retained-variant-per-family
  scope. Internal ranking scores remain hidden; ranking and sizing are unchanged.
- Agent prompt requests one short top-pick paragraph (at most 120 words).
  This prose constraint requires live-model testing; numbers come directly from
  the deterministic renderer, not model-authored table rows.
- `agentic/shortlist.py` owns presentation and a lightweight shortlist reference.
  `inspect_recommendations` in `agentic/tools.py` retrieves cached ranks/families
  without repricing and rejects stale references. This is not the deferred
  shared-pack/live-data refactor.
- Displayed replies are retained in model history and the saved transcript.
  Ordinary follow-ups and custom pricing do not automatically repeat the table.
- Exact historical exclusion/pricing-failure reasons are not retained in the
  existing pack. Lookup explicitly reports that limitation rather than inventing
  reasons. Adding detailed candidate diagnostics remains future work.
- Added `SHORTLIST_PRESENTATION.md` and focused presentation tests; version
  bumped to 0.2.18. No commit, push or deployment performed.
- Validation: 129 selected tests passed across shortlist presentation, agent
  flow/tools, saved chats, linked-chat seeds, sizing traces/regimes, trade
  economics and construction-risk policy. `git diff --check` passed. Live LLM
  and browser presentation have not yet been tested.

Files: `agentic/render.py`, `agentic/tools.py`, `agentic/agent_flow.py` and the
existing UI rendering surface.

Data: the approved columns above; detail and comparison requests use a shortlist
fingerprint plus retained engine ranks. Candidate/exclusion diagnostics and the
broader stable pack/trade-ID contract remain deferred.

Decisions: choose visible columns; decide whether numeric scores are disclosable
(the current prompt forbids them); label best-per-family versus global-variant
ranking accurately. Do not change selection policy as a presentation side effect.

Acceptance: one compact shortlist and top-pick explanation; compare 1 and 3,
explain 2, risk on all five, and why vanilla is absent work without unnecessary
repricing or invented candidates.

### Step 6 — Language polish and cross-cutting regression checks

Status: implementation and automated checks complete locally; live-model
acceptance remains pending because no API key is available.

Implementation:
- `knowledge/defaults/agent_vocabulary.json` holds tunable shortlist caveats and
  narration rules. The existing cached loader supplies both prompt and table;
  restart after editing. Numerical facts and risk classifications stay in their
  existing sources, not in vocabulary JSON.
- European RKO display ranges now run low to high, preserving the strike and
  knock-out endpoint labels in both directions. Pricing and leg order unchanged.
- Currency formatting retains full units and normalizes negative zero. Tests
  distinguish missing values from computed zero and debit from credit.
- Added `tests/test_agent_language.py` for both directions, money scales/currencies,
  wording, range roles, comparison/sizing follow-ups without repricing and stale
  rank references after market/weight changes.
- Added `AGENT_LANGUAGE_REGRESSION.md`, including a live-chat acceptance script.
  No ANTHROPIC_API_KEY is available in this execution environment; scripted
  provider-free tests do not constitute real-model acceptance.
- Focused validation: 60 tests passed. Package version is 0.2.19.
  No commit, push or deployment performed.
- Full-suite first pass: 814 passed, five failed, one live test skipped.
  Fixed the relevant construction-parity assertion to compare construction terms
  without the Step 2 catalog risk metadata; separate risk-policy tests still
  verify that metadata. The 44 construction/parser and risk-policy tests pass.
- Four other failures were reproduced independently in an untouched archive of
  HEAD `aa31e8e`; no unrelated code or expectations were changed:
  - Snapshot overrides test requests GBPUSD 1W, absent from the current snapshot.
  - Structure-scorer test expects vanilla first; configured ranking gives 1x1.
  - Two vol-surface tests assume equal flat/smile candidate counts for USDTRY
    European RKO/digital; the current snapshot produces different counts.
  These are baseline issues, not a clean full-suite pass. Final regression rerun:
  815 passed, one skipped, four deselected (precisely those baseline failures).
  The live test skips without an API key. `git diff --check` also passed.

Files: approved vocabulary JSON, agent renderer/prompt, and focused tests under
the implementation target's existing `tests/` directory.

Keep numeric range ordering and monetary formatting deterministic. Keep tunable
wording in JSON, not financial formulas. Test both trade directions, debit/credit/
zero-cost structures, capped and open-loss exposures, absent Kelly distributions,
personal weights, draft/applied settings, changed data/configuration, and stable
follow-up trade references. LLM phrasing may vary; supplied facts may not.

## Decision log

- 2026-09-24: user requested this plan be recorded and reviewed step by step.
- No implementation decisions or application changes approved by that request.
- User agreed to separate the four financial concepts while retaining existing
  sizing calculations.
- User clarified that the stop only calculates premium spend for this method;
  target P&L / sizing risk is not used.
- User confirmed the ratio numerator is net payout at the specified target,
  net of premium, not maximum payout. Application implementation remains pending.
- User approved “Not applicable — no premium outlay” for the ratio on zero-cost
  and net-credit trades, retaining target net payout and separate risk information.
- User approved evaluating target net P&L at the engine's specified evaluation
  horizon, using expiry payoff only at expiry and explicitly labelling the horizon.
  Step 1 financial definitions are agreed; application implementation is not yet
  approved.
- User subsequently approved Step 1 implementation. Implemented locally and
  validated with 178 selected tests; remaining plan steps are not implemented.
- User approved replacing Step 2 mathematical eligibility work with per-construction
  configuration. Implemented the no-tails gate and shared consumers; additional
  geometry and Greeks remain deferred.
- User deferred market-data refresh changes until live data is introduced and
  requested proceeding to the next improvement; no refresh fix implemented.
- User approved adding only missing sizing explanation fields and retaining the
  existing disclosed Kelly-to-fixed-loss fallback. Step 4 implemented locally.
- User approved the initial shortlist columns and proceeding with Step 5.
  Implemented presentation and already-priced detail lookup locally, without
  changing financial computations or reviving the deferred live-data work.
- User approved Step 6. Implemented local language configuration, display-only
  fixes and expanded regressions; live-model acceptance remains explicit.
