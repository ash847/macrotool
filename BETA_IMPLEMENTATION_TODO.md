# Beta implementation: combined Claude and Codex checklist

Consolidated by **Codex (OpenAI coding agent)** on 13 October 2026 from the
independent Claude and Codex reviews of agentic-workflow at bf049fc / v0.2.49.
The recommendations come from both AI assistants, not from a team sign-off.

**This is the single active implementation checklist for the two reviews.**
The source reviews are retained as historical evidence, not parallel task lists:
[Claude review](BETA_READINESS_TODO_CLAUDE.md) and
[Codex review](CODEX_BETA_REVIEW_TODO.md).
The broader [beta roadmap](BETA_READINESS_PLAN.md) still governs legal, licensing,
audience and launch decisions. Completing this technical list alone does not
authorise launch.

## How to use this file

Items are ordered by implementation stage. Each has one or two functional tags
from the requested set, an urgency, dependencies and a completion check.
Assign an owner before starting. Tick an item only after recording its commit,
test output or live-check evidence in the evidence column below.

Tags (the words below are the full tag names used on each task):

- core engine function improvement
- testing improvement
- system consistency
- system controls
- engineering improvements

Urgency:

- **U0 - before new invitations:** containment and a trustworthy release gate.
- **U1 - before widening the pilot:** reliability, explainability and reproducibility.
- **U2 - before the relevant expansion:** asset/product foundations and cleanup.

The day labels are suggested work sessions over several days, not a delivery
promise. Database migrations, quant decisions and external approvals may take
longer. Finish a stage's gate before promoting its changes. Independent work can
run in parallel, but keep behavioural changes separate from code-only refactors.

Baseline: the reviewed non-live tests/ run had 1,077 passes and six failures.
The tests used the shared main virtual environment, not a newly built release
environment. Live database permissions, deployed branches, provider limits and
contracts were not audited. Recheck the current checkout before implementing.

## Differences between the reviews: decisions to make explicit

| Topic | Difference or correction | Combined recommendation |
| --- | --- | --- |
| Database protection | Claude says database rules protect against application bugs; the app currently uses a service key that bypasses row rules. | Enable rules to deny direct client access AND check ownership on server writes. Rules alone do not protect a leaked service key. Assess narrower credentials separately. |
| Spending | Claude proposes a check at turn start and a secrets kill switch; Codex calls for checks before every paid call, including retries. | Use a turn-start check for friendly feedback plus atomic reservation at each paid-call boundary. Recheck the stop switch there; define how quickly changes take effect. |
| Sizing | Claude offers scenario-worst-case sizing OR clearer labels; Codex asks for an approved policy when risk is missing. | Keep clearer labels, but do not assume they make maximum-size fallback safe. Quant must choose a policy; scenario worst-case is a candidate, not an approved replacement or a contractual loss bound. |
| Net-credit example | Claude says max loss is just premium and gives a EURPLN example. The code has legacy sizing fields and a separate TradeEconomics loss assessment. | Treat the example as a case to reproduce, not a newly verified result. Separate premium, sizing estimate and contractual loss before changing conclusions or rankings. |
| Timing | Claude's sections say before roughly 100 users, but its closing sequence says before growing past roughly 20; Codex uses wider-beta gates. | Use U0/U1/U2 gates below, not headcount as evidence of safety. |
| Hosting | Claude states main deploys directly and proposes agentic-workflow staging. Neither review checked live hosting. | Verify actual deployments first; then approve and document branch mapping. agentic-workflow staging / main production is a proposal, not a verified fact. |
| Tests | Claude says fix or delete failing tests; Codex allows explicit, time-limited disposition. | Fix real defects; replace obsolete expectations with tests of intended behaviour. Delete only obsolete or redundant coverage with a reason. Any quarantine needs an owner and expiry and cannot hide a material defect. |

The architecture recommendations agree: keep Python-owned calculations, defined
agent tools and deterministic tables. Keep Streamlit initially; do not build a
distributed system or rewrite the app based on guessed user counts.

## Stage 1 - Day 1: establish a safe baseline and release path

- [ ] **T01. Reproduce and resolve the six test failures.**

  Tags: testing improvement; engineering improvements. **Urgency: U0.**
  Depends on: none. Sources: Claude 1; Codex 4.
  Now: two login tests expect removed Google login, one override test expects an
  absent tenor, one scorer expectation differs from tuning, and two smile tests
  fail. Some tests mix current market data with fixed historical assumptions.
  Do: reproduce in the agreed environment; classify each as a defect, obsolete
  expectation or fixture problem. Use frozen coherent market fixtures for pricing
  tests. Record a reason for any deletion, replacement or temporary quarantine.
  Example/trade-off: normalising six failures can hide a seventh real bug.
  Maintaining fixtures costs effort but makes failures meaningful.
  Done when: intended behaviour is tested, material defects are fixed, and the
  baseline is green or has an explicitly approved, owned, expiring exception.
  Evidence: tests/test_roles_visibility.py, test_snapshot_overrides.py,
  test_structure_scorer.py, test_vol_surface_refactor.py.

- [ ] **T02. Fix packaging and add automatic release checks.**

  Tags: engineering improvements; testing improvement. **Urgency: U0.**
  Depends on: T01 for a passing release gate. Sources: Claude 1; Codex 4.
  Now: the wheel omits agentic and workspace; requirements.txt, pyproject.toml
  and uv.lock disagree; no checked-in CI workflow was found.
  Do: choose one dependency source of truth, regenerate the others as needed,
  include every runtime package, and run checks on pushes and pull requests.
  Test a built installation outside the source checkout; keep paid live tests
  explicit opt-in. Make required checks block promotion.
  Example/trade-off: an app can work in its checkout but fail in a clean container.
  Releases take more preparation but become repeatable.
  Done when: a clean installation imports and starts the app, tests run in CI,
  and a failing check prevents promotion.

- [ ] **T03. Verify deployments and establish staging, promotion and rollback.**

  Tags: engineering improvements; system controls. **Urgency: U0.**
  Depends on: T02 for promotion. Sources: Claude 5; Codex 4.
  Do: record the actual branch, commit and version of each live app. Agree a
  staging/production mapping, separate credentials and data, and a deliberate
  promotion process. Rehearse app/config/data rollback and database restore.
  Example/trade-off: an ordinary development push should not unexpectedly change
  users' recommendations. More release discipline reduces accidental outages.
  Keep Streamlit for the initial cohort; measure simultaneous use before changing
  hosts. Moving hosts and rewriting the UI are separate decisions.
  Done when: a staged release and rollback are demonstrated and documented.

Stage gate: a known baseline and a safe place to test migrations. U0 invitations
remain blocked until Stage 2 is complete as well.

## Stage 2 - Days 2-3: protect users, data and spending

- [ ] **T04. Lock down private tables and enforce ownership on writes.**

  Tags: system controls; testing improvement. **Urgency: U0.**
  Depends on: T03 staging. Sources: Claude 3; Codex 1.
  Now: logging SQL disables row rules on chat_turns, app_errors and reactions;
  service-key workspace writes trust supplied objects.
  Do: audit live grants and policies, ship reviewed migrations denying direct
  public access, and validate owner identity on every save/upsert/link operation.
  Add two-user read, overwrite and delete-denial tests.
  Example/trade-off: user A must not overwrite B's chat by passing B's ID.
  Extra checks reduce reliance on every caller behaving correctly. Row rules
  do not remove the need for these checks while service-key access is retained.
  Done when: live access evidence and negative ownership tests pass.
  Evidence: db/migrations_tester_logging.sql; workspace/store.py.

- [ ] **T05. Tidy admission and define a durable identity/status path.**

  Tags: system controls; system consistency. **Urgency: U0.**
  Depends on: T04 for the durable table; coordinate with T06. Sources: Claude 4; Codex 2.
  Now: verified magic-link admission exists. Personal-weight email comparisons
  are not consistently normalised; access-request mail has a hard-coded fallback
  recipient; approval is a secrets list without an approval history.
  Do: normalise identity once, normalise personal-weight membership, configure
  the access-request recipient explicitly, and fail clearly if it is missing.
  Add user status, approval/revocation history and accepted-terms version before
  new invitations; use trusted stable identity rather than arbitrary inputs.
  Recheck permission before subsequent paid calls, not just page load. Prevent
  development-auth bypass in production.
  Example/trade-off: revoking access should stop new calls in an existing chat;
  an already dispatched provider call may still finish. Administration takes
  more work but becomes auditable.
  Done when: mixed-case identity, multiple tabs, revocation and production
  bypass-denial tests pass; an owner can approve/suspend/revoke with a record.

- [ ] **T06. Add durable allowances and a global AI stop switch.**

  Tags: system controls; testing improvement. **Urgency: U0.**
  Depends on: T04; shares identity boundary with T05. Sources: Claude 2; Codex 2.
  Do: check allowance before starting a turn; atomically reserve and reconcile
  allowance for each paid call, including retries and follow-ups. Add per-user
  daily and global limits, a stop switch checked at the call boundary, and
  dedicated provider workspace limits/alerts. Fail closed when allowance storage
  is unavailable; leave authorised saved results and support readable.
  Example/trade-off: five tabs must not spend the same remaining allowance five
  times. Some requests will be refused. In-flight calls can still create spend,
  so do not promise zero overshoot or rely on best-effort telemetry as a ledger.
  Done when: concurrent requests, retries, uncertain timeouts, switch activation
  and storage outages have explicit tested outcomes.
  Evidence: agentic/telemetry.py; agentic/agent_llm.py.

- [ ] **T07. Bound message size and growing chat history.**

  Tags: system controls; system consistency. **Urgency: U0.**
  Depends on: T06 for cost measurement. Sources: Claude 2; Codex 2.
  Now: calls resend the whole history; up to six rounds can each request 9,000
  output tokens. A long pasted note makes subsequent calls expensive.
  Do: enforce message, total-context and tool-result limits. Define when to ask
  for a new chat or summarise old discussion; retain the active deterministic
  trade/settings and preserve the saved original transcript separately.
  Trade-off: limiting or summarising history can lose early detail. Do not let
  a summary invent or replace the authoritative current numbers.
  Done when: oversized input is refused politely and long chats retain the right
  active trade and settings without unbounded provider requests.

- [ ] **T08. Bound background work and make interruption behaviour explicit.**

  Tags: system controls; engineering improvements. **Urgency: U0.**
  Depends on: T05-T06 identity and allowances. Sources: Claude 9; Codex 2.
  Now: a thread per turn fixes navigation-related lost replies, but does not
  survive server restarts and has no shared concurrency bound.
  Do: cap simultaneous work per user and process, add deadlines and safe
  cancellation boundaries, and clearly report interrupted/unsaved work.
  A timeout must not silently start a duplicate billable request.
  Trade-off: some turns wait; limiting threads is cheaper than a full queue.
  Add a durable job queue only if load/recovery requirements justify it.
  Done when: navigation, multiple tabs, slow providers and restarts have tested
  outcomes and do not bypass limits.
  Evidence: interface/agent_job.py.

Stage gate: no new invitations until access, database and spending checks pass
and the controls are verified in staging and the actual deployment.

## Stage 3 - Day 3-4: make answers traceable and chats dependable

- [ ] **T09. Resolve versioned config outside the UI and freeze it per run.**

  Tags: testing improvement; system consistency. **Urgency: U1.**
  Depends on: T02. Sources: Claude 6 and 8; Codex 5.
  Now: engine loaders import interface database/security helpers; remote weights,
  affinity and commentary are mutable and cached independently.
  Do: create a UI-independent config resolver and pass its immutable result to
  the engine. Record code/config/data/model/prompt identities, the resolved user
  profile and relevant distribution identity per run. Include versions in cache
  keys. Retain the underlying inputs, not just their hashes.
  Audit live remote commentary against approved rules and checked-in content.
  Example/trade-off: explain yesterday's answer after today's weight change.
  Controlled activation is less convenient than instant edits, but prevents
  mixed versions. Reproduce deterministic numbers exactly; versioned prompts
  alone do not guarantee identical LLM prose.
  Done when: a saved calculation can be replayed and a config change cannot
  silently reuse an incompatible cached result.
  Evidence: knowledge_engine/loader.py; scenario_weighter.py; agentic/session.py.

- [ ] **T10. Save each exchange atomically and make retries safe.**

  Tags: system consistency; engineering improvements. **Urgency: U1.**
  Depends on: T04. Sources: Codex 6.
  Do: save the idea/version/conversation/link/turn in one database transaction;
  add a unique request ID, safe retries and version-conflict handling.
  Keep session-only mode visibly labelled; distinguish a generated reply from
  a successfully saved reply.
  Example/trade-off: two tabs or a network failure must not leave half a turn
  saved. More database work buys reliable history.
  Done when: injected failures and simultaneous saves produce a complete record
  or a clear recoverable error without duplicates.
  Evidence: workspace/service.py:170; workspace/store.py.

- [ ] **T11. Split the Agent page from UI-independent application operations.**

  Tags: engineering improvements; system consistency. **Urgency: U1.**
  Depends on: coordinate with T05-T10; avoid a large simultaneous rewrite.
  Sources: Claude 12; Codex 12.
  Do: move the Agent page out of the large app.py and extract job, settings,
  persistence and logging operations into small services. Keep Streamlit as
  the display layer and preserve existing behaviour.
  Example/trade-off: a spend check should be reviewable without reading a
  2,094-line page script. Modest refactoring costs time but simplifies testing
  and future interfaces. Do not make this a prerequisite for every urgent fix.
  Done when: a turn can be tested without Streamlit and page/navigation tests pass.

- [ ] **T12. Link support records to runs and control duplicated transcripts.**

  Tags: system controls; system consistency. **Urgency: U1; privacy approval remains a launch gate.**
  Depends on: T09 run identity; T04 storage protection. Sources: Codex 12.
  Do: use one run ID across usage, errors, feedback and saved output; add durable
  feedback triage/delivery state. Agree and implement retention and deletion
  across workspace transcripts, telemetry and intended support stores.
  Example/trade-off: a complaint identifies the exact calculation; deleting a
  chat does not leave an undocumented copy forever. Less raw logging reduces
  debugging detail but reduces unnecessary confidential-data storage.
  Done when: feedback resolves to a run version, failed delivery is visible,
  and deletion/retention tests cover all intended stores.

Stage gate: a result can be traced and its calculations reproduced; interrupted
or concurrent chat activity cannot quietly corrupt the saved state.

## Stage 4 - Day 4-5: make financial meaning and degraded output clear

- [ ] **T13. Approve the sizing policy for net-credit and missing-risk cases.**

  Tags: core engine function improvement; system consistency. **Urgency: U1; start the quant decision on Day 1.**
  Depends on: T01 tests; T09 for comparison evidence. Sources: Claude 10; Codex 3.
  Now: net-credit and missing/near-zero loss-proxy cases can receive 10 times W.
  Do: keep premium, stress-loss estimate and contractual loss separate using
  TradeEconomics. Reproduce Claude's EURPLN example before drawing conclusions.
  Decide whether to restrict/leave unsized, use an approved stress rule, or retain
  a clearly qualified policy. Labels are necessary but are not risk control.
  Scenario worst-case sizing is an option to evaluate, not an automatic solution.
  Trade-off: safer restrictions may remove recommendations; a changed sizing
  method can change rankings and needs quant approval and comparison runs.
  Done when: the chosen policy and rationale are recorded, edge cases pass,
  affected rankings are reviewed, and uncertainty does not silently mean max size.

- [ ] **T14. Add a score and gap-to-best display on a common basis.**

  Tags: system consistency; core engine function improvement. **Urgency: U1.**
  Depends on: T13 interpretation; T09 versioned settings. Sources: Claude 10; roadmap section 9.
  Do: render scenario-weighted P&L score and absolute gap from the top package in
  Python. Keep family fit separate; state currency, sizing and evaluation basis.
  Handle ties, negative scores and zero best scores; do not divide by a near-zero
  winner to manufacture a percentage gap.
  Example/trade-off: rank 1 and rank 2 may be almost identical. More columns
  improve comparison but add clutter and expose more methodology.
  Done when: quant/product approve wording and deterministic display tests pass.

- [ ] **T15. Expose pricing/ranking failures and approve fallback rules.**

  Tags: core engine function improvement; system consistency. **Urgency: U1.**
  Depends on: T09/T12 for traceable reasons. Sources: Codex 7.
  Do: distinguish missing charts, flat-vol fallback, unavailable ranking and
  invalid pricing. Carry status/reason to the UI, agent and logs; decide which
  conditions block recommendations.
  Example/trade-off: a failed smile must not look like a normal smile-priced
  result; an internal error must not blame the user's valid target.
  More warnings or blocked results are preferable to hidden method changes.
  Done when: failure-injection tests verify the displayed result and reason.
  Evidence: agentic/standard_pack.py:341 and :407; agentic/shortlist.py:174.

- [ ] **T16. Preserve original user intent alongside engine direction.**

  Tags: system consistency; core engine function improvement. **Urgency: U1.**
  Depends on: T10 saved-state migration; T09 versioning. Sources: Codex 8.
  Do: retain original direction, move reference and magnitude, resolved target
  and forward-relative interpretation. Validate finite positive targets and
  approved horizons at the input boundary.
  Example/trade-off: spot 100 / forward 105 / target 102 is up from spot but below
  forward. More explicit fields avoid changing what a saved conversation appears
  to mean. Preserve pricing conventions unless separately approved.
  Done when: between-spot-and-forward views, explicit forward requests, settings
  changes and refresh round-trip with correct labels and tail preferences.

- [ ] **T17. Validate and display market-data freshness by component.**

  Tags: core engine function improvement; testing improvement. **Urgency: U1.**
  Depends on: T09 provenance; T03 publication/rollback. Sources: Claude 7; Codex 9.
  Do: record dates/sources for spot, forwards, vols and curves; validate units,
  supported tenors and missing data before publishing a snapshot. Reject bad
  imports rather than overwrite the last good version. Show the input as-of
  information in every answer, including historical saved answers.
  Example/trade-off: a recent headline date must not conceal older EUR/GBP
  curves. Checked-in snapshots stay useful, but freshness policy can restrict
  results until data is updated.
  Done when: invalid imports are quarantined and mixed/stale data clearly warns
  or blocks according to an approved policy.

- [ ] **T21. Add trade payoff and P&L charts to the Agent chat.**

  Tags: core engine function improvement; system consistency. **Urgency: U1.**
  Depends on: T11 Agent page separation, T13 financial meanings, T16 trade context.
  Source: explicit user addition after the combined review. ID T21 preserves
  existing task IDs; implement it in Stage 4, not after the expansion tasks.
  Do: let users chart a selected recommended package or a custom-priced trade.
  Use the actual resolved strikes, leg ratios, premium and final sized notional
  from the engine; reuse existing chart/payoff code where suitable. The LLM
  requests a chart but does not calculate or draw its numerical curve.
  Distinguish gross expiry payoff from net expiry P&L after entry premium.
  If a pre-expiry mark-to-market curve is offered, label its evaluation date,
  remaining maturity and volatility assumptions separately.
  Label the horizontal axis as terminal/scenario spot and the vertical axis
  with currency and sized/per-unit basis. Mark current spot, forward, target,
  strikes, any expiry barrier, zero P&L and valid break-even points.
  Keep the chart tied to its trade/run and sizing version; changing trade or
  settings must refresh it or clearly mark it as historical. Do not silently
  replace charts attached to earlier answers with the current trade.
  Example/risk: a zero-cost seagull can have a negative tail; zero entry premium
  must not appear to mean zero risk. A finite chart range must not imply capped
  losses. Show barrier discontinuities rather than smoothing across them.
  Trade-off: charts make risks easier to understand but add rendering cost and
  screen clutter; make them available on selection/request, not on every answer.
  Done when: chart values match the engine and tables for calls, puts, spreads,
  seagulls, digitals and ERKOs; premium, currency conversion, caps, break-evens,
  barrier boundaries and trade/settings changes have regression coverage.
  Check readable desktop/mobile rendering and show unavailable states explicitly.

Stage gate: quant signs off financial semantics and users can distinguish a
current normal result from a stale, historical, restricted or degraded result.
Do not widen the cohort solely because a day or headcount target was reached.

## Stage 5 - later sessions: prepare expansion without delaying urgent controls

- [ ] **T23. Add inception package Greeks: delta, gamma and vega.**

  Tags: core engine function improvement; system consistency. **Urgency: U2.**
  Track: Function / Product. Source: explicit user addition on 27 October 2026.
  Scope: show inception sensitivities for the selected recommended or
  custom-priced package at its actual sized notional, calculated in Python.
  Agree spot versus forward delta, gamma units, reporting currency, premium/
  conversion treatment and fixed-vol versus moving-smile assumptions before
  implementation. Display vega per one volatility point, with explicit units.
  Start with vanillas and vanilla-leg packages: sum each leg's sensitivities
  using its signed quantity. Include seagull wing quantities and custom ratios
  when supported. Keep strikes, leg ratios and sized notional fixed when
  calculating sensitivities; do not reconstruct or resize the trade on a bump.
  Validate digitals and ERKOs separately using appropriate product formulas or
  bump-and-reprice methods; do not substitute ordinary vanilla Greeks.
  Dependencies: T13 risk meanings and T09 run context; coordinate with T19's
  product model and T22's flexible construction. Detailed design remains open.
  Example/trade-off: a 1x2 spread's net delta is long-leg delta minus twice the
  short-leg delta. Consistent conventions add work but prevent misleading
  comparisons between trades, currencies and sizing levels.
  Done when: analytical and numerical checks agree within documented tolerances,
  signed-leg aggregation and notional scaling are tested, barrier/digital bump
  stability is checked, and displayed Greeks remain tied to the correct trade,
  market snapshot and sizing version. Show unavailable rather than inventing
  values for unsupported cases.

- [ ] **T22. More flexible package construction.**

  Tags: core engine function improvement; system consistency. **Urgency: U2.**
  Track: Function / Product. Source: explicit user addition on 27 October 2026.
  Scope: let users request spreads with different leg ratios rather than being
  limited to the preset constructions. For example, support a requested 1x1.3
  spread with its actual leg quantities reflected consistently in pricing,
  payoff/P&L charts, risk and sizing.
  Design/specification: deferred; agree supported constructions, ratio limits,
  validation and user interaction before implementation. Coordinate with T19's
  product-model work and T21's charts; detailed dependencies remain to be decided.
  Trade-off: greater flexibility adds validation and risk-explanation complexity.
  This is a backlog addition, not approval to change the default recommendation
  menu or implement unrestricted structures now.
  Done when: a specification is agreed and the subsequently implemented scope
  passes pricing/payoff, sizing and display-consistency tests for custom ratios.

- [ ] **T18. Define asset and contract metadata before XAU/BTC/ETH.**

  Tags: core engine function improvement; engineering improvements. **Urgency: U2.**
  Depends on: T09, T13, T17. Sources: Codex 10; roadmap asset workstreams.
  Do: explicitly define units, settlement/collateral currency, contract type,
  forward source, pricing conventions and supported packages. Remove the
  assumption that an unrecognised base asset can use the USD curve.
  Example/trade-off: BTC-settled and USD-settled contracts need different treatment;
  XAU is not just another currency code. More upfront specification makes later
  additions safer.
  Done when: each separately approved asset passes independent price/payoff/unit/
  sizing checks and unsupported contracts are rejected. This task is a foundation,
  not a claim that all three assets can ship within this week's checklist.

- [ ] **T19. Converge product pricing and review legacy paths.**

  Tags: engineering improvements; core engine function improvement. **Urgency: U2.**
  Depends on: T01 parity tests; coordinate with T18. Sources: Claude 11; Codex 11.
  Do: follow PRODUCT_MODEL_PLAN.md toward one authoritative priced package for
  totals, legs, payoff, scenarios and sizing. Audit the separate Kelly bridge
  and retain its parity coverage until consumers converge.
  Inventory users of conversation/ and config/override_detector.py; decide
  whether the old intake path is needed. Retire it only after callers are migrated.
  Example/trade-off: fixing pricing once should fix every display. A gradual
  migration costs time but avoids a high-risk rewrite and double maintenance.
  Done when: each migrated family has one source of truth and retained parity/
  correctness coverage. Keep financial corrections separate from refactoring.

- [ ] **T20. Organise docs and keep one live status record.**

  Tags: engineering improvements. **Urgency: U2; update known stale claims now.**
  Depends on: none for status updates. Sources: Claude 13; Codex execution notes.
  Do: catalogue root plans as current, completed or superseded, then move suitable
  documents to docs/ with updated links. Preserve required agent instruction
  entry points. Mark the old security access model obsolete and correct the beta
  roadmap's claim that admission checks are absent.
  Keep this file as the combined review checklist; use the roadmap for launch
  scope, legal/data-rights gates and product sequencing.
  Example/trade-off: a new contributor should not implement an obsolete security
  plan. Link cleanup takes time but prevents contradictory implementation work.
  Done when: one index points to current plans and superseded documents cannot
  reasonably be mistaken for active instructions.

## Refactor delivery design - incorporated by Codex, 27 October 2026

This section combines Codex's refactor recommendations with the user's supplied
eight-part alternative proposal (opening: "Finish the product-model migration").
It extends the existing task IDs, not a second checklist. Tags and urgency are
inherited from the mapped tasks. Reported customer incidents and file/test/branch
counts in the proposal are leads to reproduce, not independently verified facts.

### Qualifications and disagreements retained for visibility

| Proposal | Decision and reason |
| --- | --- |
| All products work from legs | Accept one contract model, but not ordinary vanilla-leg arithmetic for every instrument. Digitals and barriers retain their own payoff, monitoring and settlement rules behind a shared valuation interface. |
| Builders produce legs and solve rules | Accept for construction only. Resolve strikes and zero-cost quantities once; freeze them for scenarios and Greeks. Re-solving on a bump measures a different trade. |
| One telemetry table | Accept one typed event contract and shared run ID, not a mandatory physical table. Keep durable spend reservations separate from best-effort diagnostics; transcripts can need different access and retention. |
| Last-good snapshot fallback | Accept only with explicit stale-data status and an approved warn/block policy. Never silently present old data as current. |
| Generate grammar from the family registry | Share capability definitions and validate prompt/parser agreement. The current registry records family, policy and arity, not every ratio/anchor constraint; extend it first. Full automatic grammar generation is optional, not a prerequisite. |
| Return best partial result after round exhaustion | Return only a successful result linked to the current request, clearly labelled incomplete. Do not reuse an old trade as if the failed new request succeeded. |
| One-at-a-time UI helper | Useful for double clicks, but insufficient across tabs or servers. Retain server-side idempotency, ownership and version-conflict checks. |
| Delete legacy flow / prune branches | First inventory callers and decide Trade View's future. Migrate before removal; branch/worktree cleanup needs owner approval and recovery checks. No automatic deletion is authorised here. |
| Six tests fail on main | Not established by this review. The recorded run was agentic-workflow using the shared main virtual environment. Re-run the intended checkout and update counts rather than copy this claim. |
| Live data becomes an adapter swap | The interface reduces integration work, but does not settle feed licensing, quality, asset units or contract conventions. T18 and roadmap data-rights gates still apply. |

### Agreed implementation scope

1. **One resolved trade, with separate construction and valuation (T19, T22,
   T21, T23).** Extend the existing Structure/Leg model rather than add a rival
   model. Builders turn validated requests into signed legs and construction
   rules. Resolution produces fixed strikes, quantities, barriers, expiry and
   contract conventions. Add a PricingContext for market inputs and valuation
   assumptions. Payoff, scenario valuation, break-evens, economics and Kelly
   consume that resolved trade instead of reconstructing it from a label.
   Keep unit valuation, risk assessment and sizing separate; reuse TradeEconomics
   and sizing trace records without silently changing approved sizing policy.
   Example: a requested 1x1.25 spread must retain the 1.25 quantity in every
   table and chart, and still have that quantity after a spot bump.
   Completion evidence: family-by-family parity, signed quantity tests,
   no-shock valuation equality and fixed-contract bump tests. Any intentional
   financial correction gets separate approval and tests, not disguised parity.

2. **Typed requests and tool results (T15, T16, T22).** Use the same validated
   request for chat, forms and saved trades, preserving original spot/forward
   intent. Give results statuses such as ok, unsupported, invalid_request and
   error, with stable reason codes, request/run IDs and structured payloads.
   Carry pricing degradation as explicit warnings/status, not success prose.
   Centralise supported families, ratio limits, anchors and constraints; derive
   capability descriptions from that source and test parser agreement. Do not
   make the model retry an unsupported ratio as a spelling problem. On round
   exhaustion show a valid current partial result, or a specific failure.
   Completion evidence: unsupported ratios, invalid syntax, engine failures,
   exhausted rounds and stale-result rejection each have contract tests.

3. **Small application services and typed UI state (T11, T10, T19).** Extract
   evaluate, price, refresh and save operations behind UI-independent boundaries;
   split page rendering into modules and wrap session state with defined fields.
   Decide whether Trade View is retained; if retained, both screens consume the
   same calculation service and pack rather than parallel recommendation logic.
   Reuse a guarded operation helper for Apply, Refresh, Rename and Archive,
   backed by T10's server-side protections. Preserve sparse Kelly reseeding and
   reading widget state after rendering. Test navigation and double clicks as
   well as cross-tab conflicts before retiring any legacy path.

4. **Run record, typed events and complete cache identity (T09, T12).** Attach
   a RunContext and result ID to every pack: trusted user identity, request,
   code/config/prompt/model versions, snapshot identity, applied preferences and
   relevant distribution/valuation assumptions. Retain replayable input versions,
   not just fingerprints. Link selection audit, tool calls, paid calls, errors,
   feedback and saved outputs with typed events. Cache keys include every input
   that affects results and enforce user isolation. Verify changed data/config
   cannot reuse old output and a support report resolves to the exact calculation.

5. **One versioned config resolver (T09, T03).** Validate local defaults,
   remote overrides and personal settings against schemas, resolve precedence
   once, and freeze the complete bundle per run. Keep database/UI imports out
   of quant code. Support staged activation, promotion and rollback, including
   commentary so explanations cannot describe a different rules version.
   Test that mid-run edits affect the next run, never half of the current one.

6. **MarketDataProvider and explicit curve lookup (T17, T18).** Start with an
   adapter for the current file. Return validated versioned snapshots with
   component dates, source and quality status. Refresh must obtain a fresh
   validated snapshot before recalculation and atomically publish the new result;
   failure leaves the previous answer visibly historical, not partly refreshed.
   Introduce currency-keyed curves with migration compatibility and explicit
   errors for unsupported bases, not a USD fallback. Test unavailable/stale
   feeds, bad imports, changed snapshot refresh and historical answer retention.

7. **Server-side policy boundary (T05-T08).** Centralise trusted identity,
   admission, role and operation checks. Call it before every paid LLM attempt
   and expensive engine entry, including non-chat routes. Use durable atomic
   spending reservations for paid calls and separate concurrency/work limits
   for computation. A UI button guard is not authorisation. Test revoked users,
   retries, multiple tabs, storage outages and the global stop switch.

8. **Release and repository hygiene (T01-T03, T20).** Establish current test
   evidence and clean-install CI before moving core consumers. Correct stale
   tool lists and test counts in instructions; prefer reproducible commands over
   rapidly ageing counts. Index plans by current/completed/superseded status,
   preserve required instruction entrypoints and update moved links. Inventory
   branches/worktrees; propose cleanup separately, checking ownership, dirty
   files and unmerged/unpushed work before requesting deletion approval.

### Delivery order within the existing stages

1. Baseline tests, packaging and release gates; implement urgent policy controls
   without waiting for an app rewrite (T01-T08).
2. Establish request/result contracts, run/config identity and small service
   boundaries (T09-T12, T15-T16); start the T17 snapshot provider.
3. **Bring a narrow T19 slice forward to Stage 3/4, at U1:** vanilla and
   two-leg spreads become resolved contracts used by valuation and one output
   consumer. Require parity and no-shock tests before switching consumers.
4. Build charts on migrated families first, then migrate other supported
   families and their consumers incrementally. Do not declare all-family T21
   complete until its existing acceptance criteria pass. Extend to specified
   custom ratios and Greeks only when their contracts and tests are ready.
5. Finish remaining T19 migration and approved legacy retirement; introduce
   currency-map/asset conventions before XAU/BTC/ETH expansion. Full T19 remains
   U2; the early slice is a dependency for new calculation features, not a
   prerequisite for shipping access or spending fixes.

## Owner and completion evidence

### Product slice 1 checkpoint - Codex, 27 October 2026

Worktree: agentic-workflow, starting at merge commit fc2aac0 (includes main
7e70b5e). No production pricing, sizing policy or configuration changes made.

- Replaced the product parity harness's live snapshot dependency with synthetic
  dated market pillars. Covered both directions for USDBRL, EURPLN and GBPUSD,
  with flat and smile pricing. Smile construction errors now fail rather than
  silently substituting flat vol; each listed family must exercise variants.
- Added tests/test_product_engine_baseline.py with fixed vanilla/spread/butterfly
  constructions independent of production JSON. Checks signed quantities,
  premiums against a separate Black-76 formula, unchanged-market repricing,
  terminal payoff at strikes and in tails, Kelly payoff agreement and no mutation.
  Both call/put directions and positive/negative carry are covered.
- Initial full-suite command: `.venv/bin/python -m pytest tests -q -p no:cacheprovider`.
  Result before the new tests were collected: 1,082 passed, six failed, six
  skipped in 120.99 seconds, using this worktree's .venv. This is not a green
  release baseline and no failures have been quarantined or deleted.
- Focused post-edit command: `.venv/bin/python -m pytest tests/test_product_engine_baseline.py tests/test_product_model_parity.py tests/test_kelly_payoff_bridge_parity.py tests/test_trade_economics.py tests/test_sizing_trace.py tests/test_premium_basis.py tests/test_payoff_risk.py tests/test_scenario_pricer.py -q -p no:cacheprovider`.
  Result: 164 passed in 8.93 seconds. `git diff --check` passed.

| Existing failure | Observed evidence / disposition |
| --- | --- |
| test_roles_visibility: two Google-provider expectations | Configured provider list no longer includes Google. Admission workstream; confirm intended support before updating tests. |
| test_snapshot_overrides: forward_points_recomputed_from_spot | Current GBPUSD snapshot has no 1W forward. Test assumes it exists; replace with a controlled fixture in follow-up. |
| test_structure_scorer: vanilla_wins_far_target_high_carry | Current settings rank 1x1_spread first. Check intended config versus stale expectation; do not retune scoring to satisfy the test. |
| test_vol_surface_refactor: european_rko_entry_smile_moves_premium | Flat path returns six variants, smile path three. Fixture combines live USDTRY smile with fixed forward/vol constants. Reproduce with coherent inputs and inspect rejection reasons before classifying as fixture-only or pricing defect. |
| test_vol_surface_refactor: european_digital_entry_smile_moves_strike | Flat path returns three variants, smile path two; same mixed fixture concern. Investigate before specialised-product migration. |

Limits: parity with legacy is not independent proof that legacy math is correct.
The broad parity harness still intentionally uses the current variant catalog;
only the new fixed-construction tests are catalog-independent. Existing sizing
trace checks characterise policy, not approve it. Seagull full-precision versus
rounded wing quantities remain an explicit migration concern documented in the
Kelly parity test. No claim is made yet about no-shock smile valuation for every
specialised product. Keep T01 and T19 open; address these gaps before declaring
the full slice-1 gate or all-family migration complete.

### Slice 1 follow-up - controlled smile regression tests

Codex replaced the live USDTRY fixture in tests/test_vol_surface_refactor.py
with a dated synthetic market: spot 45, 91-day forward 46.7597, ATM vol 0.2121,
consistent forwards/discount factors and a mild, fixed delta smile. Both calls
and puts now exercise smile entry pricing with matching variant labels and no
warnings. Unsafe-smile tests remain and explicitly require variant rejection.
The digital scenario test can no longer pass by skipping all priced variants.
No production formula, interpolation method, guard or configuration was changed.

Diagnosis: the original mixed fixture caused three ERKO variants and one digital
variant to be rejected for negative digital values. Equal flat/smile counts were
therefore not a valid assertion for those inputs. A controlled fixture isolates
the intended regression: a usable smile changes prices/strikes without rejection.

**Separate open finding (T17/T19):** the current checked-in USDTRY snapshot also
rejects three ERKO variants with coherent market inputs (spot 48.652, 91-day
forward 52.539165, ATM vol 0.11305). Rejected labels are 25-delta/10-delta,
30-delta/10-delta and 20-delta/10-delta. The reported digital unit value is
approximately -0.002567 at barrier 63.4264. All three digital variants price
under that coherent-market check. This is evidence of an invalid value produced
by the current surface/pricer combination, not proof that the quoted market
itself contains arbitrage. Surface quality, interpolation and digital derivative
behaviour need a separate investigation; do not disable the rejection guard.

Verification: the previous focused command plus
`tests/test_vol_surface_refactor.py` passed **182 tests in 8.48 seconds**;
`git diff --check` passed. The full suite was not rerun in this follow-up;
the four other previously observed failures remain unresolved. The two smile
regression failures are resolved with fixtures, not suppressed or quarantined.
The next architecture slice can start with vanilla/two-leg resolved contracts;
the specialised-product surface finding remains a gate for that later migration.

### Resolved-contract migration checkpoint - Codex

Implemented an immutable ResolvedContract/ResolvedLeg representation for vanilla,
1x1, 1x1.5 and 1x2 spreads, retaining entry strikes, signed quantities and original
tenor. Added a separate PricingContext and fixed-leg revaluation/terminal-payoff
functions. Entry construction remains unchanged; the legacy and product pricers
retain the resolved contract rather than re-solving it during valuation.

Scenario valuation, expiry break-evens, downstream trade economics, engine Kelly
and the linked Kelly screen now consume this contract where present. Old objects
without it and unmigrated families retain compatibility paths. Product-entry
pricing/renderer reconstruction are not yet consolidated; do not claim the
entire product is defined in only one execution path yet. Package version is
0.2.51 for the source change; nothing has been deployed by this work.

New tests compare old/new consumers across spot/vol/time shocks, flat and smile,
calls and puts; assert immutability; forbid builder/anchor-resolution calls during
valuation; check explicit quantities rather than ratios inferred from labels;
and reject invalid inputs. Focused suite: 205 passed in 16.71 seconds.
Full-suite verification after migration: `.venv/bin/python -m pytest tests -q
-p no:cacheprovider` returned 1,131 passed, four failed and six skipped in
123.60 seconds. Remaining failures are the two Google-provider expectations,
missing GBPUSD 1W override fixture and tuned scorer expectation already listed
above. No additional failures were introduced. The source edits and evidence
remain uncommitted; no push or deployment was performed.

**Historical decision, now approved for correction (see below):** unchanged-market
smile revaluation does not always equal the entry price. Entry uses delta-pillar
volatility; the scenario path re-evaluates the fixed strike with sticky-delta
skew interpolation. In the frozen USDBRL 90-day ATM-forward call example, entry
premium is 2.9416594177% of base notional, whereas unchanged-market revaluation is
2.9375894555%, a difference of about 40.70 per million base notional. The legacy
and new revaluation paths give exactly the same value: this is pre-existing,
not a migration regression. Flat-vol no-shock tests pass.

Choose whether to preserve these conventions for the refactor and address the
gap separately, or approve a separately tested financial correction that makes
entry and scenario smile valuation consistent. Do not conceal the difference
with looser test tolerances or silently change premiums/rankings. Specialised
USDTRY ERKO surface validity remains a separate open issue. No new ratio UI,
Greek calculation, pricing-policy change or specialised-family migration has
been included in this checkpoint.

### Approved strike/smile correction - Codex, 27 October 2026

The user approved tests and implementation of the pricing correction separately
from the behaviour-preserving migration. Package version is now 0.2.52.

- Replaced approximate z-score interpolation with exact unadjusted forward-delta
  inversion, including the half-variance term. This applies to calls and puts,
  including non-pillar deltas such as 34-delta.
- Added explicit `atmf_legs` metadata to the catalogue and parser. Existing ATMF
  constructions retain K=F; their entry volatility is read at that strike.
  Explicit 50-delta requests are distinct. Removed the misleading ATMF (50-delta)
  catalogue label. The default recommendation menu has not been expanded.
- Added bounded root-solving fallback to strike-vol inversion. The previous
  ten-iteration limit could return an unconverged volatility on steep smiles;
  exact strike conversion alone did not fix that second source of mismatch.
- Preserved sticky-delta scenario behaviour, premium conversion and sizing
  policy. Corrections to prices/strikes can nevertheless change sizing and ranks.
  Previously retained resolved contracts are not reconstructed by valuation.
- The converged USDTRY smile still gives negative raw totals for three ERKO
  packages. The old pricer clipped these to zero after its digital-value guard.
  Negative smile package totals now raise the existing SmileArbitrageError and
  are omitted with warnings, rather than appearing as free trades. Flat pricing
  and tiny numerical-roundoff clamping are unchanged. This refines the earlier
  surface finding: unfinished inversion affected the digital-value warning, but
  convergence alone does not make the entire surface arbitrage-free.

Frozen USDBRL 90-day examples (premiums as percent of base notional):

| Trade | Before premium | After premium | Construction change |
| --- | --- | --- | --- |
| ATM-forward call or put | 2.94165942% | 2.93758946% | Strike stays 5.09961302; correct strike vol replaces the 50-delta quote. |
| Call ATMF/25-delta spread | 1.77553616% | 1.84323135% | Short strike changes from 5.37134926 to 5.38727845. |
| Put ATMF/25-delta spread | 1.89022105% | 1.81881016% | Short strike changes from 4.85461423 to 4.86756840. |
| Zero-cost seagull, either direction | 0% | 0% | Exact-delta wing strikes and solved quantities change; no-shock value remains zero within numerical tolerance. |

Regression evidence: the first new suite exposed 61 failures before correction;
the separate steep-smile cases exposed 19 failures before the convergence fix.
Tests cover exact forward-delta round trips, ATMF versus true 50-delta, both
directions, all vanilla-leg families' no-shock prices, forced exhaustion of the
iteration budget, invalid inputs, and negative ERKO package rejection.
The focused correction/pricing/parity suite passed 215 tests in 4.34 seconds.
An additional read-only check across all 13 current snapshot pairs at 90 days,
both directions and the curated vanilla variants, found maximum absolute
entry/no-shock difference 1.30e-12 per unit notional (about 0.0000013 per million).
This is evidence for those inputs, not a global surface-arbitrage guarantee.
Final full suite: `.venv/bin/python -m pytest tests -q -p no:cacheprovider --tb=short`
returned 1,248 passed, four failed and six skipped in 157.16 seconds. The four
failures remain the two login-provider expectations, missing GBPUSD 1W fixture
and scorer expectation recorded above. No new failures. Changes remain local
and uncommitted; no deployment or push was performed.

### Unified entry-pricing slice - Codex

Checkpointed the preceding work locally: 0a3acb2 contains the consolidated
documents; bed493e contains the tested resolved-contract and pricing correction.
The engine changes were checkpointed together because their shared files and
tests formed one verified baseline. Neither checkpoint was pushed.

Vanilla, 1x1, 1x1.5 and 1x2 entry pricing now goes through product_pricer once.
The PricedVariant compatibility/sizing envelope retains the resulting
`product_price` and its exact `resolved_contract`. Recommendation and custom
request displays reuse that product instead of rebuilding from family/label
and repricing. Existing scenario, break-even, economics and Kelly consumers
continue to use the retained contract. Package version is 0.2.53.

The old four entry functions are no longer selected by production dispatch;
they remain temporarily as independent test references. The broad parity test
uses those references rather than comparing the product pricer with itself.
Unmigrated families and older objects retain their existing display fallback.
No default menu, financial convention, eligibility policy or sizing policy was
changed in this slice. Legacy small-premium labels and reversed-input table
semantics are explicitly preserved, not endorsed as a new product specification.

New migration tests cover entry metrics, retained object identity, one pricing
call per custom request, target/minimum-distance gates, small premiums and
reversed inputs. Complete recommendation packs in both directions match the
old path's ranks, scores and notionals under fixed-loss and Kelly sizing.
The focused entry/parity/pricing suite passed 193 tests; the expanded migration
file then passed all 34 tests, including the four end-to-end pack comparisons.
Final full-suite command: `.venv/bin/python -m pytest tests -q -p no:cacheprovider --tb=short`.
Result: 1,282 passed, four previously recorded failures and six skipped in
167.03 seconds. No new failures; `git diff --check` passed. This slice is
checkpointed locally without pushing or deploying.

### Three-leg entry and valuation slice - Codex

Completed 2026-10-28 on agentic-workflow; package version 0.2.54.
Butterflies and seagulls now use the shared product entry path, retained display
result and immutable resolved contract for scenario valuation and Kelly payoffs.
Seagull solved wing quantities and butterfly derived strikes remain fixed under
market shocks. Stop-budget plumbing, target gates and nonpositive butterfly-wing
filtering are preserved. Old routines remain independent test references.

Verification: full suite 1,308 passed, the same four known failures, six skipped
in 146.27 seconds. Regression coverage includes both directions, flat/smile,
stressed valuation, expiry payoffs, stop budgets and retained object identity.
No cloud/UI verification or push. Handoff: PRODUCT_ENGINE_HANDOFF_CODEX.md.

Still outstanding: family-specific construction and risk rules remain; arbitrary
multi-leg construction is not enabled. Specialized binary/barrier contracts and
multi-expiry support are not migrated by this slice. Do not mark all of T19 done.

All items are open. Fill this table as work is assigned and completed.

| ID | Owner | Commit / test / live-check evidence |
| --- | --- | --- |
| T01 | Unassigned | Pending |
| T02 | Unassigned | Pending |
| T03 | Unassigned | Pending |
| T04 | Unassigned | Pending |
| T05 | Unassigned | Pending |
| T06 | Unassigned | Pending |
| T07 | Unassigned | Pending |
| T08 | Unassigned | Pending |
| T09 | Unassigned | Pending |
| T10 | Unassigned | Pending |
| T11 | Unassigned | Pending |
| T12 | Unassigned | Pending |
| T13 | Unassigned | Pending |
| T14 | Unassigned | Pending |
| T15 | Unassigned | Pending |
| T16 | Unassigned | Pending |
| T17 | Unassigned | Pending |
| T18 | Unassigned | Pending |
| T19 | Unassigned | Pending |
| T20 | Unassigned | Pending |
| T21 | Unassigned | Pending |
| T22 | Unassigned | Pending specification |
| T23 | Unassigned | Pending specification |

## Coverage: nothing dropped from either source review

| Source item | Combined tasks |
| --- | --- |
| Claude 1: tests and CI | T01, T02 |
| Claude 2: spending, input size, full history, provider limits | T06, T07 |
| Claude 3: database and ownership | T04 |
| Claude 4: admission, email normalisation, request recipient, revocation | T05 |
| Claude 5: staging, production and hosting | T03 |
| Claude 6: reproduction and live commentary | T09 |
| Claude 7: freshness and import checks | T17 |
| Claude 8: UI-independent engine config | T09 |
| Claude 9: threads, deadlines and load-driven queue decision | T08 |
| Claude 10: sizing decision and score gap | T13, T14 |
| Claude 11: pricers, legacy intake and Kelly bridge | T19 |
| Claude 12: split Agent page | T11 |
| Claude 13: document organisation | T20 |
| Codex 1: database protection | T04 |
| Codex 2: access, allowance and job boundaries | T05-T08 |
| Codex 3: uncertain sizing risk | T13 |
| Codex 4: packaging, tests, releases | T01-T03 |
| Codex 5: frozen resolved config | T09 |
| Codex 6: reliable save | T10 |
| Codex 7: explicit failure/fallback | T15 |
| Codex 8: original view versus engine direction | T16 |
| Codex 9: data quality | T17 |
| Codex 10: asset/contract definitions | T18 |
| Codex 11: product-model migration | T19 |
| Codex 12: UI services, run-linked support, retention | T11, T12 |
| Codex execution notes: roadmap corrections and sequencing | T20; stage gates |
| User addition: trade payoff/P&L charting in Agent chat | T21 |
| User addition: more flexible package construction / requested spread ratios | T22 |
| User addition: inception package delta, gamma and vega | T23 |
