# MacroTool — wider-beta readiness plan

Date: 7 October 2026  
Baseline reviewed: `agentic-workflow`, v0.2.43, commit `002aea2`  
Status: proposed plan for agreement; not an implementation or a production security certification.

## 1. Recommendation and scope

Launch a **controlled, invite-only research beta for professional users**, then expand in measured cohorts. Do not move directly from friendly testers to unrestricted Google sign-in. Audience and jurisdiction restrictions require legal agreement; professional status alone is not a regulatory exemption.

Separate two tracks:

- **Launch readiness:** access controls, isolation, spend limits, licensed and clearly dated data, reliable releases, correct outputs, privacy/legal terms, onboarding and feedback.
- **Product expansion:** live feeds, AUD/CAD, BTC/ETH, public APIs, referral rewards and richer analytics. These should not delay a safe FX beta, and should not be used to distract from launch blockers.

Suggested rollout, subject to founder approval: 20–30 invited users first, then approximately 100 after two weeks of satisfactory operating evidence. These are experiment sizes, not claims about hosting capacity. Measure peak simultaneous usage separately.

The beta remains decision support: no execution, no path-dependent products, no promise of executable prices or investment performance. Preserve the architectural rule that Python produces calculations, rankings and tables; the LLM interprets requests and narrates results.

## 2. What already exists, and what needs verification

The following is based on repository inspection and work completed in this task. Provider consoles, deployed secrets, production RLS, contractual data rights and deployment settings have **not** been audited for this plan.

| Area | Current evidence | Consequence for this plan |
|---|---|---|
| Login | `interface/security.py` requires configured OIDC login, but does not enforce a tester admission allowlist after login. Non-admin signed-in users become testers. Google/Auth0 provider settings may impose other restrictions. | Verify provider settings and add explicit application admission checks; authentication is not permission to join the beta. |
| Navigation | `interface/navigation.py` restricts testers to Agent and About; admins retain the other pages. | Do not reopen Trade View/Batch/Kelly as a tester fallback. Audit server-side actions as well as navigation. |
| Data access | The app uses a server-side Supabase service key. Workspace tables declare deny-by-default RLS, with user ownership enforced by application code. | Service-role requests bypass RLS; both database policy and application ownership checks need testing. |
| Critical schema warning | `db/migrations_tester_logging.sql` explicitly disables RLS on `chat_turns`, `app_errors` and `reactions`. | P0: inspect live permissions and policies immediately; do not infer exposure or safety from source alone. Replace the unsafe migration posture and verify direct access is denied. |
| LLM | Default `claude-sonnet-5-5`; agent loop defaults to six rounds and the adapter allows 9,000 output tokens per call. | “Six calls at 3K tokens” is not a valid cost model. Measure actual usage; caps are ceilings, not typical consumption. |
| Caching and telemetry | Prompt caching and per-call token/cache/latency/stop-reason metrics exist in `agentic/agent_llm.py`, `agentic/telemetry.py` and `PROMPT_CACHING.md`. | Verify live coverage and cache hit rates; do not rebuild this or assume caching makes output free. Logging is best-effort, not a spending enforcement ledger. |
| Market data | Checked-in snapshot date is 16 September 2026. Its note says EUR/GBP discount curves were retained rather than refreshed with the spot/forward/vol quotes. | Show component freshness, not just one reassuring headline date. Refresh or explicitly restrict mixed-date datasets before broader use. |
| Product surface | About, sizing explanations, contact email, chat logging and reactions already exist. | Improve onboarding and feedback delivery; these are not greenfield features. |
| Recent fixes | v0.2.42–43 add loss-bound reporting, target/budget displays and hard preference filters. Net-credit sizing still sets notional to 10×W by policy. | Verify deployed UI; narration changes do not make that sizing policy conservative or a guaranteed loss bound. |
| Commentary | Remote commentary can override local JSON. The USDJPY trace showed “Avoid seagulls” while Seagull was second in Structure Fit. | Audit and reconcile active remote prose. Prompt safeguards alone are not a complete content cleanup. |
| Releases/tests | Work has been pushed to `agentic-workflow`; older notes say `main` auto-deploys. No `.github/workflows` directory was found in this checkout. One known scorer expectation conflicts with current tuning. | Verify the actual deployed branch/version, add CI and deliberate promotion, and resolve or formally quarantine stale tests. |

Older `SECURITY_HARDENING_PLAN.md` describes an anon insert model that differs from today's service-role model. Reconcile that document during hardening; do not copy its old SQL into production.

## 3. Delivery sequence and ownership

Roles below are responsibilities to assign, not assumed staffing. Engineering includes application/platform work; quant owns financial correctness; founders own audience, commercial and product decisions; counsel owns jurisdiction-specific legal advice.

| Stage | Deliverables | Accountable roles | Exit gate |
|---|---|---|---|
| A — scope and audit | Audience/jurisdictions, legal brief, data-rights inventory, live access/RLS/deployment audit, risk register | Founders + counsel + engineering | No unknown critical exposure; approved scope and named owners for blockers |
| B — containment | Admission/revocation, cross-user isolation, quotas/circuit breaker, privacy controls, staging/rollback, licensed snapshot process | Engineering + data owner | Security, quota and restore/rollback tests pass |
| C — product readiness | Risk/narration acceptance suite, quick-start/user guide, score presentation, durable feedback, monitoring dashboard | Quant + product + engineering | Users can complete core tasks without guidance; material numerical defects closed |
| D — controlled pilot | Small invited cohort, daily issue/cost review, weekly product interviews, initial research report | Founders + support owner | Two weeks meeting agreed service/cost/quality gates |
| E — wider beta | Larger capped cohort, approved invite referrals, capacity adjustments | Founders + engineering | Capacity and cost evidence support the next cohort; no unresolved P0/P1 incident |
| F — expansion | Automated/live data, AUD/CAD, then separately assessed crypto and external API | Data + quant + engineering + counsel | Separate readiness and licensing review for each expansion |

Start legal/data-rights work and engineering audits in parallel. Do not wait for new assets or a positive backtest before improving security. Do not publish promotional performance claims before the methodology and claims have been reviewed.

## 4. P0 — access, security and confidentiality

### Admission and sharing the app

- Start with a server-enforced invite allowlist. Add invited/active/suspended/revoked status and accepted-terms version. Check admission before protected reads, expensive calculations and every paid API call, not only at login.
- Verify the identity provider's issuer, subject and verified email claims. Prefer a stable internal user ID derived from trusted identity; retain normalised email for contact. Never trust an email or owner ID supplied by the browser or LLM tool arguments.
- Test revoked users with existing sessions, multiple tabs, fresh sessions and saved deep links. Account for session expiry and reauthentication; disable development auth overrides in production with an explicit environment safeguard.
- A shareable **app/invitation link** should lead to a landing page or controlled signup, not grant access to another person's trade history. Later invite codes should be expiring, limited-use, revocable and redeemed atomically.
- Defer public trade/chat sharing. If added later, require explicit owner action, preview/redaction, a separate read-only export, expiry/revocation, access policy and data-redistribution permission. Never share internal IDs as if they were authorization tokens.

### Database and admin boundaries

- Inventory every table, storage bucket, function/RPC and grant, including logs, reactions, config history, workspace records and future quota tables. Enable appropriate RLS and deny public access to sensitive data; test with unauthenticated and ordinary-user credentials.
- Test two users attempting to read/update/delete each other's conversations, ideas, settings and exports. Service-role writes need explicit ownership predicates and server-derived identity; RLS alone cannot protect them.
- Gate config writes, user administration, exports, batch jobs and personal-weights access on the server. Record actor, old/new config version and timestamp for privileged changes. Navigation hiding is only a usability measure.
- Keep provider keys out of HTML, tool results, exceptions and logs. Scan current files and history; rotate anything actually exposed. Use separate staging/production secrets and restrict operator access.
- Review dependency vulnerabilities and untrusted rendering/link handling. Treat prompts and tool outputs as untrusted data; validate tool schemas, resource IDs and allowed operations in Python. The LLM must have no general DB/admin/network capability.
- Keep global caches limited to public/approved shared data. Include user/profile, snapshot hash, preferences and engine/config version in relevant private cache keys. Run simultaneous-user tests for identity/config bleed.

**Acceptance:** unauthorized access cannot consume paid calls or read/write private data; negative cross-user and admin tests pass; live RLS/grants are documented; no critical secret or dependency exposure remains. Consider a focused independent security review before enabling referral-led growth.

## 5. P0 — spend containment and abuse protection

- Create a dedicated production Claude workspace and scoped key, separate from development. Set the applicable provider workspace/organization spend limits and alerts; verify their actual enforcement. A separate key alone is not a spend cap. Provider and application controls are complementary, and neither should be described as zero-overshoot protection for already in-flight work. [Anthropic workspace controls](https://platform.claude.com/docs/en/manage-claude/workspaces), [rate/spend limits](https://platform.claude.com/docs/en/api/rate-limits).
- Add a durable per-user allowance for turns, paid tokens/cost and expensive engine runs, plus burst/concurrency controls. Enforce by stable user identity across devices and sessions. Limit message size, total context, tool-result size and tool rounds; charge or reserve each actual provider call, including follow-ups and retries.
- Use atomic admission/reservation and reconciliation, with idempotency keys, so concurrent requests cannot overspend a remaining allowance. Reserve a conservative maximum before the call. If usage is uncertain after a timeout, retain a conservative debit pending reconciliation rather than refund automatically. Reconcile against provider billing, including cache categories and pricing version.
- Add global daily and monthly budgets and an operator kill switch. If the quota store fails, new paid calls fail closed; telemetry failure must not remove spend controls.
- Give users friendly remaining-allowance/reset messages. On a global stop, leave About, support and authorized saved results available without new LLM calls. Do not expose tester-hidden admin pages as a fallback, and do not call deterministic computation “free”: it still consumes CPU and hosting resources.
- Keep the existing model pinned for beta; evaluate upgrades on cost, latency and acceptance tests. Do not assume an undocumented “Opus secret” exists. Keep output limits high enough to avoid broken responses, but monitor stop reasons and test truncation recovery.
- Scope the assistant to MacroTool tasks. An off-topic policy improves UX but is not abuse protection; quota enforcement remains server-side.
- Extend abuse controls to contact mail, feedback, invite redemption and CPU-intensive pricing. Per-session cooldowns can be bypassed by reconnecting.

**Acceptance:** parallel requests, refreshes, retries and new sessions cannot bypass allowances; exhaustion gives a coherent read-only experience; a simulated kill switch stops new paid work. Choose numerical budgets after a short measurement exercise, before issuing invitations.

## 6. P0 — data, privacy and legal launch gates

### Data and privacy

- Map what leaves the app and where it is stored: login identity, views, chats, `view_json`, tool results, engine outputs, feedback, email, provider requests and any tracing service. Saved `conversation_turns.llm_messages` and `chat_turns.tool_trace` can duplicate sensitive information.
- Put a short notice before first substantive use, with a full privacy policy: what is logged, purpose, authorized readers, processors, retention, deletion/export requests and a contact. Warn users not to submit confidential fund/client information without their employer's permission. Do not promise confidentiality arrangements that have not been contracted and implemented.
- Agree lawful bases, processor contracts, transfer/residency requirements and whether a DPIA is needed with counsel. Review current provider retention/training terms for the actual account arrangement rather than generalising from consumer products.
- Proposed starting retention for approval: raw diagnostic traces 30 days, minimized operational metadata 90 days, saved workspaces under an explicit user-facing retention/inactivity policy. Counsel may require different periods. Implement deletion across duplicated tables, email/support stores and backup lifecycle; document legal holds and post-restore deletion replay. Restrict raw-trace access and audit it.
- Collect minimum useful feedback metadata; do not automatically send full PM conversations through email. Pseudonymised IDs are not necessarily anonymous. Retention must be justified by purpose, not chosen as a supposed statutory default. [ICO storage-limitation guidance](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-protection-principles/a-guide-to-the-data-protection-principles/storage-limitation/).

### Legal, corporate and IP

- Decide the operating entity, incorporation jurisdiction, founders' agreement, equity and contractor/IP assignments; confirm who owns pre-existing code, research and branding and whether prior-employer obligations apply. Arrange appropriate commercial/privacy counsel and discuss professional/cyber insurance.
- Obtain a written perimeter assessment for intended countries, audience, derivatives, personalised structure rankings/sizing, marketing, referrals and later crypto. Decide whether permissions, restrictions or a different product scope are necessary before launch. A free beta, professional audience or “not investment advice” disclaimer does not by itself settle the regulatory question. [FCA perimeter guidance](https://handbook.fca.org.uk/handbook/perg8), [financial promotions](https://www.fca.org.uk/firms/financial-promotions-adverts).
- Have counsel draft/review beta terms, privacy notice, risk disclosures, complaints process and any performance claims. Explain indicative prices, snapshot timing, sizing assumptions and loss beyond premium where applicable. Disclaimers should communicate real limitations, not assert that liability can simply be waived.
- Inventory market-data, benchmark, exchange, open-source and vendor licences. Confirm rights to display raw quotes, derived analytics, historical results and any API/shared export to this audience. Buying or accessing a terminal/feed is not evidence of redistribution rights; obtain contractual confirmation.
- Protect scoring IP with server-only config, least-privilege admin access, restricted raw traces and reviewed output. Separate explainable economic risks from proprietary weights and rules: never conceal loss/tail information to protect IP. Minimise sensitive methodology sent to the LLM; prompt instructions alone cannot guarantee it will stay hidden.

**Acceptance:** founders and counsel sign off the audience, jurisdictions, entity/contracting position, terms, privacy and data rights. Do not broaden access while a material legal or data-licensing question is unresolved.

## 7. P0/P1 — reliability, hosting and release operations

- Keep Streamlit for the first controlled cohort **only if** measured reliability and isolation pass. Do not commit to a rewrite or migration based on a guessed user count. Test peak concurrent engine runs, long chats, memory growth, cold starts, disconnects and provider/database outages.
- If capacity or hosting control is inadequate, move the existing app to managed, always-on container hosting before rewriting the UI. Add a bounded job queue/worker for costly computation if measurements justify it. Multi-instance hosting requires shared durable state/quotas, deliberate cache/session handling and idempotent jobs, not just extra replicas.
- Verify current hosting plan limits, SMTP connectivity, regions, backup facilities and uptime commitments. Name an operations owner and support coverage; a beta is not a promise of 24/7 desk support.
- Set explicit request/compute timeouts, bounded retries with backoff, cancellation and friendly error IDs. Avoid duplicate mail or billable requests after uncertain timeouts. Preserve user drafts; distinguish failed generation from lost persistence.
- Add CI for unit/integration tests, two-user authorization tests, fixed-snapshot golden outputs, schema validation and secret/dependency scanning. Resolve the known scorer expectation against intended tuning; any temporary quarantine needs an owner, reason and expiry. Never silently bless all failures as “old”.
- Create separate staging and production deployments, credentials and preferably databases. Verify which branch each actually runs. Promote a tested commit deliberately; do not let routine branch pushes become accidental public releases.
- Version the application, prompt, model, scoring config and dataset. Pin approved config/data per run; a mutable remote JSON update can be as consequential as a code deployment. Retain rollback versions and audit every promotion.
- Exercise database restoration and app/config/data rollback before launch. Proposed recovery targets for agreement: recover service within four hours and lose no more than 24 hours of saved work; confirm backup frequency and user expectations support these targets.

**Acceptance:** staging smoke and failure tests pass; a production-like load test supports the initial cohort; restore and rollback are demonstrated; a responsible person can stop traffic and communicate an incident.

## 8. P0/P1 — updated data now, live data later

### Before the first wider cohort

- Appoint a data owner and refresh cadence. Choose either an explicitly dated demonstration dataset or a routinely refreshed research dataset; never imply either is live.
- Show prominent as-of timestamp/time zone and freshness status in chat, Market State, saved results and exports. Record component dates for spot, forwards, vols and discount curves; flag mixed dates and prevent unsupported combinations from being presented as current recommendations.
- Build a repeatable import/validation/publish workflow: schema and units, forward-point factors, maturity coverage, positive discount factors, surface plausibility, missing/stale values and pair convention checks. Quarantine failed imports rather than silently overwriting the last good dataset.
- Retain immutable raw observations and normalized snapshots with source, checksum and publication version. Keep the accepted historical interpolation/missing-date rules documented; do not carry historical gap-filling into live feeds without a separate policy.
- An active run uses one consistent snapshot, even if a new one arrives mid-chat. Refresh must be explicit and labelled; cached/saved results retain their original data/version references.

### Later feed/API track

- Distinguish **a vendor data API into MacroTool** from **an external MacroTool API for customers**. The former can automate refreshes first; the latter needs its own authentication, quotas, entitlements, licensing, versioning, support and security review.
- Select a feed after checking required spot/forwards/smile/rate coverage, timestamps, conventions, latency, budget, reliability and redistribution rights. Start with scheduled snapshots where they meet user needs; live streaming is not automatically more useful for this decision-support beta.
- For live ingestion add provider adapters, health/freshness monitoring, atomic snapshot publication and last-good/stale handling. Define when stale inputs merely warn and when new recommendations are blocked. Do not mix a fresh spot with old vols/curves without an explicit approved rule.

**Acceptance:** every displayed result can identify its actual input versions and freshness; no bad import silently becomes production market data. Full live data is not a first-cohort requirement.

## 9. P1 — product readiness and feedback

### Onboarding and detailed guide

- A one-screen first-run guide: supported use, example query, target/direction/tenor, fixed-loss versus Kelly, W, preferences and limitations. Make the first successful task possible without a founder explaining it.
- Add a detailed, versioned user guide linked from About: interpreting tables, net P&L versus gross payoff, score versus realised return, premiums/credits, caps, loss budgets versus contractual loss bounds, finite-loss filtering, tail direction, expiry-only products, data dates and troubleshooting.
- Include worked examples from frozen deterministic outputs, not hand-typed numbers. Test desktop and usable mobile layouts, keyboard access, readable tables and copy/export behaviour.
- Maintain a small live-chat acceptance set covering carry direction, no-path-dependent explanations, unknown versus bounded loss, net-credit sizing, zero expiry payoff, filters, contradictory context prose and follow-up table requests. Audit active remote commentary, not just checked-in files.
- Review the economics of the 10×W net-credit policy and misleading legacy labels such as modelled “max-loss capped” linear benchmarks. Decide whether to constrain or prominently qualify such outputs before inviting unfamiliar users. The recent explanatory fixes are not a sign-off of every sizing policy.

### Show separation in the P&L ranking

- Add a Python-rendered P&L score column plus an absolute difference from the top-ranked package, on the same sizing/currency/evaluation basis. State that it is a scenario-weighted ranking metric, not a probability, forecast, realised result or contractual payoff.
- Keep Structure Fit percentages separate from package P&L scores. Do not divide by a small, zero or negative best score to create misleading percentage gaps. If a normalized view is wanted, define a stable common denominator and show its basis.
- Define rounding and a tie/near-tie presentation tolerance so tiny differences do not imply strong superiority. Scope comparisons to the same view, inputs, sizing and config; scores from different queries are not automatically comparable.
- Quant/product owners approve wording and decide how much score detail to expose in light of IP concerns. Preserve deterministic rendering and test negative, zero, tied and differently sized cases.

### Feedback and contact

- Reuse the existing contact box and reactions. Add short closed questions: “Was this useful?”, “Did it match your view?”, “Were the numbers/risk explanations clear?”, with “not sure/not applicable” where appropriate. Add optional open text: “What was wrong or missing?”
- Attach feedback to a run/turn/version, not just a session. Use a durable feedback record and admin triage state (new, acknowledged, investigated, resolved); email both founders as notification, not the only system of record.
- Add per-user rate limits, idempotent submission and retry-aware delivery status. Give explicit control before attaching a full transcript. Keep marketing opt-in separate from product feedback.
- Publish a realistic response window, name a triage owner and review recurring issues weekly. Close the loop with users when fixes ship.

## 10. P1 — activity logs, evaluation and initial backtests

### Operational and selection audit

- Separate security/admin audit, minimal usage/billing events, application errors, detailed diagnostic traces and research datasets. Give each its own access and retention policy.
- Record a stable run ID linked to trusted user ID, timestamp, code/prompt/model/config/data versions, preferences, latency, outcome, tool-call count, token categories, estimated cost and provider request ID. Reconcile spend with billing rather than treating best-effort chat traces as complete invoices.
- Add the previously discussed **family-selection audit**: enabled/disabled, hard-gate result/reason, affinity score, nonpositive-score rejection, shortlisted status, variant pricing failures, finite-loss/tail exclusions and final rank. Keep family and variant counts distinct. This enables “last N queries” analysis without guessing from the top-five display.
- Dashboard: admitted/active users, activation, repeat use, completion rate, latency percentiles, errors, truncations, cost per successful task/user, cache performance, quota events, stale-data blocks and feedback themes. Restrict raw PM views to support staff who need them.

### Initial research report

- Inventory the existing independent backtest pipeline, datasets and viewer before commissioning new infrastructure. Preserve separation from the production application and reuse validated pricing/engine conventions through a defined interface.
- Freeze trade universe, point-in-time market data, engine/config version and outcome definitions before evaluation. Retain unit-notional and sized realised P&L, individual scenario valuations and entry ranks. Audit sizing caps, premiums, FX conversion and transaction-cost assumptions.
- Carry forward the agreed date rule: schedule non-overlapping starts by the relevant calendar horizon; when inputs are missing, move that start/end pair to the next valid pair, leave adjacent scheduled pairs unchanged, and record the shift and resulting overlap. Count calendar tenors consistently rather than assuming every quarter has 90 days. Report residual dependence, including simultaneous pairs and multiple ideas on the same date.
- Predefine what “better” means for each reported metric: e.g. sized realised P&L under the common budget, alongside unit-notional P&L and relevant risk/cost measures. Do not present one metric as universal PM suitability. Compare rank correlation, top-pick regret and success conditional on the expressed view, not only aggregate win rate.
- Use out-of-sample/time-split evaluation, disclose tuning on history, and account for dependence and repeated comparisons when presenting uncertainty. Choose transparent matched baselines; do not assume a vanilla is always an economically equivalent benchmark.
- Publish an initial methodology/results/limitations note only after quant review. If evidence is weak or inconclusive, say so. A beta can test workflow usefulness without claiming proven alpha; positive significance is not a launch prerequisite.

## 11. P2 — assets and distribution growth

### AUD and CAD

- Confirm exact crosses first; AUDUSD and USDCAD are proposals, not assumed requirements.
- For each chosen pair, add conventions, calendars, settlement, delta/premium conventions, forward units, rates/curves and smile inputs through the existing data/pricing interfaces. A supported-pair label is not sufficient.
- Validate calls and puts across tenors, carry/CIP interpretation, premium currency, sizing, bounds, scenario valuations and missing-data handling against independent references. Add deterministic regression fixtures and only then expose the pair to the agent and guide.

### BTC and ETH

- Treat crypto as a separate product stream, not two more FX symbols. Decide venue, instrument, settlement/collateral currency, linear versus inverse payoff, expiry conventions, underlying/index, market-data rights, funding/forward model and scenario assumptions before designing adapters.
- Do not reuse FX carry commentary, rates, loss normalization or Kelly distributions without validation. Consider exchange/collateral/liquidity risks and jurisdiction restrictions; consult counsel again before offering crypto-related recommendations.
- Start with a clearly scoped research prototype and benchmark pricing/risk independently. Release separately after FX beta evidence and sign-off.

### Sharing and incentives

- First share an invitation/landing link with approved professional contacts. Record referral attribution without revealing who uses the app or any trade content. Invitations remain capped and revocable.
- After the pilot, experiment with a modest, fixed, expiring allowance of future tool usage for accepted useful feedback or approved referrals. Do not reward trading activity, deposits, P&L, fabricated reviews or bulk account creation. Require legal review of the offer and clear reward terms.
- Set a global reward budget and anti-self-referral/multi-account controls. Prefer priority access or a founder session if usage credits are not yet a well-defined commercial unit.
- Measure retained, qualified users and quality feedback rather than raw signup counts. Stop the incentive if it attracts abuse or disproportionate cost.

## 12. Implementation surfaces and durable artifacts

Paths below are proposed locations relative to this repository unless explicitly marked existing. Do not put secrets, raw user traces or confidential legal advice in Git.

| Change | Main implementation/storage | Why |
|---|---|---|
| Admission/authorization | Existing `interface/security.py`, `workspace/store.py`; new server-side access policy and migration-backed user/invite records | Durable, revocable authority independent of UI or session state |
| Budgets/limits | New policy JSON for tunable limits; transactional DB usage/reservation ledger; checks around `agentic/agent_flow.py` and provider calls | Domain-tunable settings with atomic enforcement; not an in-memory counter |
| RLS/admin audit | New reviewed `db/` migrations, security integration tests, revised `SECURITY.md` | Fix schema posture and preserve auditable deployment evidence |
| Per-run evidence | Structured run/selection/usage tables linked to existing chat/workspace IDs; versioned schema | Queryable last-N analytics without parsing narration |
| Data ingestion | Existing `data/` schema/loader plus source adapters; immutable raw Parquet and snapshot manifests in private controlled storage | Reproducibility, provenance and atomic validated publication |
| UI/guide/feedback | `interface/about_content.json`, versioned guide Markdown, deterministic `agentic/shortlist.py`/`dashboard.py`, feedback/outbox tables | Editable copy, engine-owned numeric cells and reliable follow-up |
| Release process | New CI workflow, staging deployment configuration, release manifest and rollback runbook | Deliberate promotion of code, config and data together |
| Legal/IP | Restricted legal register outside source control; approved public terms/privacy in versioned content | Avoid leaking privileged work while tracking acceptance and policy versions |
| Research | Existing separate backtest workspace, documented manifests, reviewed public methodology/results note | Keep research independent and avoid untraceable production changes |

## 13. Go/no-go checklist

All launch gates need evidence and an owner; a checklist tick without a test/report is not completion.

- [ ] Founders/counsel approve audience, countries, terms, privacy and data redistribution rights.
- [ ] Live RLS/grants verified, unsafe checked-in migration corrected, service-role ownership/admin checks tested.
- [ ] Admission, revocation and cross-user isolation pass negative tests; production dev bypass disabled.
- [ ] Durable per-user/global budgets and provider limits are configured; concurrency and outage tests pass.
- [ ] Every result clearly identifies approved data timing; mixed/stale inputs follow the agreed policy.
- [ ] Material financial/display defects are closed; known test failures are resolved or explicitly quarantined with owners.
- [ ] Actual remote commentary and live-chat acceptance cases are reviewed against deterministic evidence.
- [ ] Staging/promotion, backup restore and rollback are demonstrated; deployed branch/version confirmed.
- [ ] Privacy notice, retention/deletion, quick start, guide and durable feedback are available.
- [ ] Monitoring/alerts and support ownership are in place; no critical/high-severity issue remains unmitigated.

Suggested pilot service targets, to agree rather than claim as current: at least 99% successful completion of valid supported tasks, no cross-user leakage or material incorrect risk figures, bounded spend, and p95 latency within a threshold chosen from measured pilot baselines. Track login, deterministic pricing and full agent response latency separately. Use scripted load/failure tests as well as real-user evidence; a tiny quiet cohort cannot establish reliability by itself.

## 14. Decisions needed from the founders

1. Intended user type, launch countries, initial cohort and who approves invitations.
2. Monthly total operating budget, daily emergency threshold and per-user allowance philosophy.
3. Demonstration snapshot versus regularly refreshed data; required cadence, source budget and data owner.
4. Legal/entity owner, counsel and desired confidentiality/retention commitments.
5. Hosting/support service expectations and who owns incidents and release approval.
6. Whether the proposed score-plus-absolute-gap display is acceptable, and the permitted IP detail.
7. Exact AUD/CAD crosses; whether crypto and an external customer API are genuine near-term demand or later discovery.
8. Whether to start with no rewards, priority access, or a tightly capped future-usage incentive after the pilot.

Recommended next work package: **read-only production admission/RLS/deployment audit plus a short measured usage/cost baseline**, while founders obtain legal and data-rights advice. Then implement containment before onboarding a larger cohort.

## Reference notes

External guidance was checked on 7 October 2026. Recheck provider capabilities, laws and contracts before implementation; this plan is not legal advice.

- Streamlit explicitly separates OIDC authentication from application authorization: [authentication documentation](https://docs.streamlit.io/develop/concepts/connections/authentication).
- Supabase service credentials can bypass RLS and must not be exposed to clients: [RLS documentation](https://supabase.com/docs/guides/database/postgres/row-level-security).
- Provider spending controls, retention principles and regulatory considerations are linked at the relevant work items above. No vendor pricing, licence entitlement or deployment limit has been assumed from an old checklist.
