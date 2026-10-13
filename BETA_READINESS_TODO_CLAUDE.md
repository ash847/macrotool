# Beta readiness — to-do list

> **Written by Claude (AI code assistant), not by a team member.**
> Source: a review of the `agentic-workflow` branch (v0.2.49, commit `bf049fc`) against
> `BETA_READINESS_PLAN.md`, done on 13 October 2026. Claude read the code and ran the test
> suite; it did not change any code, and it did not check live Supabase, Streamlit Cloud or
> Anthropic console settings. Treat this as a recommendation list to review, not a sign-off.

**Overall:** the engine's core rule holds — Python computes, the LLM narrates — and the
duplicate code paths have parity tests. The gaps are around the engine: who gets in, what it
costs, how a release reaches users, and whether you can later prove which config and data
produced an answer.

Items marked **Trade-off** describe a past design choice, its risk, and what changing it costs.

---

## A. Must fix before inviting anyone new

### 1. Get the tests green and run them automatically
- **Now:** 6 of 1,083 tests fail (run with the main `.venv`, so some could be dependency differences):
  - `test_roles_visibility.py` — 2 tests still expect Google login, which has been removed
  - `test_vol_surface_refactor.py` — 2 smile tests for European digital / European RKO
  - `test_snapshot_overrides.py` — forward points recomputed from spot
  - `test_structure_scorer.py` — the long-standing "vanilla wins far target" test
- There is no `.github/workflows`, so nothing blocks broken code from deploying.
- **Why it matters:** if "6 failures" becomes normal, a 7th (a real pricing bug) goes unnoticed.
- [ ] Fix or delete each failing test, with a one-line reason
- [ ] Add a GitHub Actions workflow that runs `pytest` on every push and pull request

### 2. Cap spending
- **Now:** no per-user cap, no message-length cap, no kill switch. Each message re-sends the
  whole chat history; one turn can make up to 6 model calls, each allowed 9,000 output tokens.
- **Example:** a tester pastes a long research note, then chats for 40 turns — every later turn
  costs far more than the first, and nothing stops it.
- [ ] Check a per-user daily allowance (stored in the database) in `_start_agent_turn` before the turn starts
- [ ] Add a global "stop all AI calls" switch in secrets
- [ ] Cap the length of a single message
- [ ] Use a dedicated Anthropic workspace with its own spend limit and alerts
- [ ] If the allowance check fails, refuse the turn (fail closed)
- **Trade-off:** full history was chosen so the Agent remembers the whole chat. Risk: cost grows
  every turn. Trimming or summarising old turns caps cost but loses early detail.

### 3. Lock down the database
- **Now:** `db/migrations_tester_logging.sql` turns row-level security **off** for `chat_turns`,
  `app_errors` and `reactions` — these hold PM views and tool output. Chat privacy relies on
  every query in `workspace/store.py` remembering its `user_email` filter. `save_conversation`
  upserts by ID alone, with no owner check.
- [ ] Turn row-level security on for the three logging tables (and check the live database, not just the file)
- [ ] Add an owner check to every save/upsert
- [ ] Add a test where user A tries to read and overwrite user B's chat
- **Trade-off:** the server-side service key was chosen because it's simple and the app does the
  filtering. Risk: one forgotten filter, or one leaked key, exposes every user's chats.
  Database rules cost some setup but protect you even when the app has a bug.

### 4. Tidy up admission
- **Now:** admission by verified magic-link email works and is checked on every page load. But
  approving someone means editing `allowed_emails` in Streamlit secrets and restarting; there's
  no suspended/revoked state and no record of who was approved when.
- [ ] Lower-case emails in the personal-weights check (`can_have_personal_weights` compares as-is; login lower-cases)
- [ ] Move the hard-coded fallback address in `interface/access_requests.py` into secrets
- [ ] Before ~100 users: move users into a database table with a status, and check it before every paid call
- Note: an Agent turn already running keeps going after you remove someone — acceptable now, fixed by the item above.

### 5. Separate staging from production
- **Now:** a push to `main` goes straight to users; there's no staging app or rollback button.
- [ ] Run a staging app from `agentic-workflow` and production from `main`; promote by deliberate merge
- [ ] Confirm which branch and version each live app is actually running
- **Trade-off:** Streamlit Cloud was chosen because it needs no servers. Risk: one shared process
  for all users, little control over restarts, no rollback. Keep it for the first cohort; decide
  on moving hosts from measured load, not guesses.

---

## B. Fix before growing to ~100 users

### 6. Make every answer reproducible
- **Now:** scenario weights, commentary and affinity scores can change remotely (admin saves).
  Each server process caches them until restart, and no chat turn records which version it used.
- **Example:** the USDJPY case where commentary said "avoid seagulls" while Seagull ranked 2nd —
  today you can't prove which commentary that user saw.
- [ ] Stamp each logged turn with config version, snapshot fingerprint, model and prompt version
- [ ] Audit the live remote commentary against the checked-in JSON

### 7. Show how fresh the market data is
- **Now:** the snapshot says 16 September 2026, but the EUR/GBP discount curves are older.
- [ ] Record a date per component (spot, forwards, vols, curves)
- [ ] Add a check script that rejects bad imports instead of overwriting the last good snapshot
- [ ] Show the as-of date in every chat answer
- **Trade-off:** a checked-in snapshot is simple and reproducible. Risk: it goes stale without anyone noticing.

### 8. Stop the engine reaching into the UI code
- **Now:** `knowledge_engine/loader.py` and `knowledge_engine/scenario_weighter.py` import
  `interface.supabase_logger` and `interface.security`.
- **Example:** any script or backtest that runs the engine quietly pulls in Streamlit and its secrets.
- [ ] Move config loading into one small module that doesn't depend on `interface/`, and pass config into the engine
- Makes item 6 easy and is what a future job queue would need.

### 9. Put limits on Agent background threads
- **Now:** each Agent turn runs on a background thread (`interface/agent_job.py`) so it survives
  the user clicking away — this fixed a real lost-reply bug.
- **Risk:** no limit on how many run at once, no timeout, no cancel; a restart loses turns in flight.
- [ ] Add a timeout per turn
- [ ] Cap concurrent turns per user
- [ ] Only build a full job queue if load tests show it's needed

### 10. Quant decision on odd sizing outputs
- **Now:** net-credit structures are sized at the 10× cap, and their "max loss" is just the premium.
- **Example (2m EURPLN 4.40):** net-credit 1x2s rank #2 and #4 at the full 1,000 notional, while
  European digitals are sized at a notional of 0.7–2 and so look worthless in currency terms
  despite a good per-unit result.
- [ ] Decide: size on scenario worst-case loss, or label these outputs clearly
- [ ] Add the plan's "score plus gap to the top pick" column (plan §9)

---

## C. Clean-up (lower priority, smooths later work)

### 11. Reduce duplicate code paths
- **Now:** two pricers (`analytics/structure_pricer.py`, `analytics/product_pricer.py`) kept in
  line by parity tests; the old intake chat path (`conversation/`, `config/override_detector.py`)
  is still present; there's a separate Kelly payoff bridge.
- **Trade-off:** keeping them let features ship fast. Risk: every pricing fix must land twice,
  and parity tests only catch the cases they cover.
- [ ] Decide whether the old intake path is needed for beta; retire it if not
- [ ] Plan a move to a single pricer

### 12. Split `interface/app.py` (2,094 lines)
- [ ] Move the Agent page into its own module so access and spend changes are easy to review (a split, not a rewrite)

### 13. Organise the planning documents
- **Now:** 37 `.md` plans in the repo root; some contradict the code (e.g. `SECURITY_HARDENING_PLAN.md` describes a database access model no longer used).
- [ ] Move them into `docs/` and mark each as current, done or superseded

---

## Suggested order

1. Tests green + CI (item 1)
2. Spend caps + kill switch (item 2)
3. Database protection + two-user tests (item 3)
4. Staging / production split (item 5)
5. Then items 6–10 before the cohort grows past ~20

*— Claude*
