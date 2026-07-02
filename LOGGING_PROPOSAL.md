# Logging & Observability Proposal — Tester Rollout

**Status:** proposal only (no code written for this doc).
**Context:** we are about to give external testers the Trade View + Kelly Sizing surface
(with the new in-context chat). We want the richest feasible capture so we can iterate on
the scorer, the recommendations, the sizing, and the agent's narration. This documents
what we store today, what we're missing, and the lift to close the gaps — ranked.

---

## 1. What we store today

Two Supabase tables, written via `interface/supabase_logger.py` (service key):

| Table | Written when | Columns |
|-------|--------------|---------|
| `queries` | every Trade View engine run (`_run_engines`, app.py) | `prompt` (summary string), `pair`, `direction`, `magnitude_pct`, `horizon_days`, `target_z`, `carry_regime`, `top_structure`, `llm_response`, `user_email` |
| `feedback` | tester submits the 3-question feedback expander | `prompt`, `pair`, `note`, `q1_text/q1_answer … qN`, `user_email` |
| `config_history` | admin saves scenario weights / affinity / commentary | `key`, `value` (JSON), `user_email`, version metadata |

**Key observations / current gaps in what's already flowing:**

- **`llm_response` is always `""` on the live Trade View path.** The visible screen runs
  the silent deterministic pipeline, so we persist a query row with an *empty* narration.
  We are recording that a run happened, but none of its **output**.
- **The recommendation itself is not stored.** We keep `top_structure` (one family id) but
  not the ranked shortlist, the per-variant strikes/premium/payoff/RR, the per-leg
  notionals, or which variant the tester actually looked at / selected.
- **The scores are not stored.** The affinity dimension scores, eligibility/gating, and the
  scenario-weighted `score_ccy` / driver decomposition are computed and thrown away.
- **The sizing decision is not stored.** Fixed-loss vs Kelly, λ, R:R, and — most valuable —
  the tester's **elicited Kelly distribution** (their subjective probabilities) vanish at
  rerun. That distribution is the single richest signal we could collect about a tester's
  actual view.
- **Feedback is not joined to a run.** `feedback` rows carry only a prompt summary string,
  not a foreign key to the `queries` row they refer to, so we can't reliably line up a
  rating with the exact recommendation it rated.
- **No session / trace identity.** Rows are independent; we can't reconstruct a tester's
  path through a session (view A → tweaked to view B → asked the chat → gave feedback).

## 2. The big one: chat transcripts

**Are they stored anywhere? No.** Not in Supabase, not on disk. They live only in process
memory for the life of the Streamlit session:

- **Agent tab** and the **new Trade View chat** both keep the full provider-native message
  history on `AgentSession.messages` — this includes the `tool_use` / `tool_result` blocks,
  i.e. the *engine ground truth the model saw*, which is exactly what we need to audit
  whether narration matched the numbers. `agentic/app.py:_agent_tool_trace` already
  reconstructs `(tool, args, result, is_error)` from it.
- A parallel display list (`st.session_state.agent_chat` / `tv_chat`) holds the rendered
  `(role, text)` turns.

When the tester closes the tab, all of it is gone. **For iteration this is the highest-value
missing capture** — the transcripts are how we'll see what testers actually ask, where the
agent is weak, and which explanations land.

### Proposed design

A per-chat **session id** (uuid, minted when a chat flow is created — one for the Agent tab,
one per Trade View trade-chat) threaded into a new table:

```sql
create table chat_turns (
  id            bigint generated always as identity primary key,
  chat_id       uuid        not null,      -- one conversation
  seq           int         not null,      -- turn order within the conversation
  surface       text        not null,      -- 'agent_tab' | 'trade_view'
  role          text        not null,      -- 'user' | 'assistant'
  text          text,                       -- rendered narration / prompt
  tool_trace    jsonb,                      -- assistant turns: [{name,args,result,is_error}]
  pair          text,
  view_json     jsonb,                      -- the live TradeView at this turn
  user_email    text,
  created_at    timestamptz not null default now()
);
create index on chat_turns (chat_id, seq);
```

- Log **two rows per exchange**: the user prompt, then the assistant reply with its
  `tool_trace` (reuse `_agent_tool_trace`). Or one row per exchange with both — either is fine;
  per-turn is simplest and append-only.
- `view_json` / `pair` tie every turn to the trade in context, so a transcript is
  self-describing without joining.

### Lift: small (~half a day)

1. `supabase_logger.log_chat_turn(chat_id, seq, surface, role, text, tool_trace, pair, view, user_email)`
   — mirrors the existing `log_query` pattern (service client, try/except, no-op if unconfigured). ~25 lines.
2. Mint a `chat_id` where each flow is built (Agent tab init; `_render_trade_chat` seed block)
   and stash it in session state; keep a per-chat `seq` counter. ~6 lines each.
3. Call `log_chat_turn` after each `advance()` in both surfaces (2 call sites). ~10 lines.
4. **You run the migration** (the `create table` above) in Supabase — the app uses the
   service key and won't create tables itself.

No new dependencies; the seam and write-path already exist. The only external step is the
one-time table creation.

## 3. Recommended enrichments, ranked by value ÷ effort

| # | What to add | Why it matters | Lift |
|---|-------------|----------------|------|
| 1 | **Chat transcripts** (§2) | The core iteration signal; nothing captured today | Small |
| 2 | **Kelly distribution + sizing spec** on the query row (`sizing_method`, `λ`, `R:R`, elicited `probs`/`bins`, conviction) | The tester's actual subjective view — richest data we can get; also lets us replay sizing | Small (data already in `st.session_state` / `flow.sizing_spec`) |
| 3 | **Store the recommendation output**: ranked shortlist + per-variant strikes/premium/payoff/RR/notional as JSON on the query row | We currently keep only `top_structure`; can't analyse what was shown | Small–Med (serialise `selector_result` / variants) |
| 4 | **`query_id` foreign key on `feedback`** (+ mint a query id and return it from `log_query`) | Joins a rating to the exact run it rated | Small |
| 5 | **Session id across the whole visit** (view runs + chats + feedback share it) | Reconstruct a tester's full path | Small |
| 6 | **Scores + driver decomposition** JSON on the query row | Correlate tester preference with our scoring; find where scorer disagrees with testers | Med |
| 7 | **Client-side events** (which variant expander opened, tab dwell) | Fine-grained UX signal | Med–Large (needs a component) |

Items 1–5 are all "small" and share the `log_query`/service-client pattern already in
`supabase_logger.py`; they're the recommended first batch.

## 4. Cross-cutting notes

- **PII / consent.** Transcripts + `user_email` are personal data. Confirm testers are told
  their sessions are logged for product improvement; consider a short retention window
  (e.g. 90–180 days) and a documented deletion path. The `admin_emails` gate already keeps
  raw logs admin-only in the UI.
- **Fail-open logging.** Keep every new write wrapped in try/except that no-ops on failure
  (as `log_query`/`log_feedback` do) — logging must never break a tester's session.
- **Schema migration discipline.** `log_query` already has a backward-compat fallback that
  drops `user_email` if the column is missing. Apply the same pattern to any new column so a
  not-yet-migrated table degrades instead of erroring.
- **Volume.** Chat turns are the only high-cardinality addition; at tester scale this is
  trivial for Supabase. No batching needed initially.

## 5. Suggested sequencing

1. Ship **chat transcript logging** (§2) — do this before/with the tester rollout so we
   capture from day one.
2. Add the **sizing spec + Kelly distribution** and **recommendation JSON** to the query row
   (items 2–3), plus the **`query_id` on feedback** and a shared **session id** (items 4–5).
3. Defer scores JSON (item 6) and client-side event tracking (item 7) until we know which
   questions the first tranche of data leaves unanswered.
