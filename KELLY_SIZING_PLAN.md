# Kelly Sizing — comparative variant sizing via the PM's edge

Engine + UI feature. Adds a **sizing-method toggle** so variants can be sized either by
**fixed loss** (today's behaviour, equal max-loss across variants) or by **Kelly** (each
variant sized to its growth-optimal bet given the PM's subjective distribution). Standalone
build doc; execute against this when green-lit.

**Status: IN PROGRESS (autonomous build).** Branch `feature/kelly-sizing` (reuses the retired
spot-anchored worktree). Scope this pass: **Trade View + Batch only**; the Agent stays
fixed-loss (guardrail untouched). Commit + push per phase; no PR / merge to main / deploy.

**Decisions (resolved):** D1 relocate the payoff bridge + Kelly math into `analytics/` (pure),
re-export from `kelly_v2` for back-compat — keeps `analytics/` free of `interface/` deps.
D2 bankroll `W=100` (== `LINEAR_NOTIONAL`). D3 accept two probability lenses (sizing = elicited
dist, scoring = scenario weights); documented. D4 Agent stays fixed-loss. D5 view-implied seed +
opt-in elicitation. D6 keep `target_rr` in session, relocate the R:R widget into the form. D7
Trade View writes the shared `Distribution`; Kelly screen reads it. Elicitation exposes the
existing CDF + PDF/fixed-range modes with an adjustable bin count.

---

## 0. The load-bearing invariant (why this is smaller than it looks)

For every variant, the scenario engine computes `pnl_ccy = pnl_pct × structure_notional`, and
the per-variant score is:

```
score_ccy = Σ_scenario  weight · pnl_ccy  =  structure_notional · score_pct
```

`score_ccy` is what ranks variants **everywhere** — comparator (`knowledge_engine/comparator.py`
~L474/L519), best-variant pick (`agentic/standard_pack.py` ~L154), Structure Evaluation sort
(`interface/structure_eval.py` ~L560). `score_pct` is notional-independent (it comes from the
scenario grid + scenario weights, which **do not change**).

**Consequence:** the sizing method only sets `structure_notional`. Everything downstream
(scenario pricing in ccy, scoring by `score_ccy`, ranking, render) consumes it transparently.
So this feature is **one seam swap** (`_size_variant`) + the inputs that feed it. No change to
the scenario/scoring machinery.

This also means the ranking is fully determined by the product `N_v · score_pct_v`:
- **Fixed loss:** `N_v = loss_budget / max_loss_pct_v` ⇒ rank ∝ `score_pct_v / max_loss_pct_v`
  (scenario-return per unit of max loss — edge-agnostic).
- **Kelly:** `N_v = λ · f*_v · W` ⇒ rank ∝ `f*_v · score_pct_v` (incorporates the PM's edge).

See §9 for the ordinal-ranking analysis this formula unlocks.

---

## 1. Goal & scope

1. User toggles **Sizing method: Fixed loss | Kelly** at trade-definition time.
2. **Fixed loss** → user sets the R:R explicitly (e.g. 1:3) in the same place they define the
   trade. *This input already exists* (`flow.target_rr`, intake R:R widget); the toggle just
   gates/relabels it.
3. **Kelly** → user specifies their edge as a distribution; the engine derives **relative
   notional sizes** per variant from it.
4. The engine then scores across scenarios using the new notionals (free, per §0).

**In scope:** Trade View (primary surface) and Batch (for the §9 sweep). **Out of scope for
phase 1:** the Agent path stays on fixed loss (see §7 — agent guardrail). Kelly v2 standalone
screen is reused as a library, not replaced.

---

## 2. Architecture / the seam

```
intake (app.py)
  └─ SizingSpec{method, target_rr | KellyDist, lambda}  ──► flow.sizing_spec
        │
        ▼
build_comparator_inputs(... sizing_spec ...)            (knowledge_engine/comparator.py)
        │   per shortlisted variant:
        ▼
price_variants(... sizing_spec ...)                     (analytics/structure_pricer.py)
        │
        ▼
_size_variant(pv, sizing_spec, linear_notional)         ◄── THE SEAM
        ├─ method == "fixed_loss": N = min(loss_budget/max_loss_pct, cap)   [today]
        └─ method == "kelly":      N = λ · kelly_fraction(dist, pv) · W,  capped
        │
        ▼   (unchanged below)
price_scenarios → score_structure (score_ccy = N·score_pct) → ranking → render
```

A single `SizingSpec` value object replaces the loose `loss_budget` / `target_rr` scalars
threaded through `price_variants` / `build_comparator_inputs` today. Default constructs to
`fixed_loss` with `target_rr=3.0` so **every existing caller and test is unchanged**.

---

## 3. Data model & plumbing

New module `analytics/sizing.py` (pure, no IO, no Streamlit):

```python
@dataclass(frozen=True)
class SizingSpec:
    method: Literal["fixed_loss", "kelly"] = "fixed_loss"
    target_rr: float = 3.0                 # fixed-loss only
    kelly_dist: "Distribution | None" = None   # kelly only (kelly_v2.kelly.Distribution)
    kelly_lambda: float = 0.5              # fractional Kelly multiplier
    bankroll: float = 100.0               # W; nominal, == LINEAR_NOTIONAL for continuity
```

- `Distribution` is imported from `interface/kelly_v2/kelly.py` (probs + outcome bins).
  *Refactor note:* if importing `interface/*` into `analytics/*` is undesirable (layering),
  lift `Distribution` into `analytics/` and have kelly_v2 re-export it. **Decision D1.**
- `flow.sizing_spec: SizingSpec` added to `conversation/flow.py`; **must be reset in `reset()`**
  alongside `target_rr` (existing invariant). Set from session state before each engine run,
  same pattern as `flow.target_rr` / `flow.structure_constraint`.
- Threaded as one param `sizing_spec: SizingSpec = SizingSpec()` through:
  `price_variants`, `build_comparator_inputs`, `interface/structure_eval.py` price paths,
  `agentic/standard_pack.py` (phase 1: always passes fixed_loss), `interface/batch_view.py`.
  `loss_budget` / `target_rr` / `linear_notional` are subsumed by / derived from the spec.

---

## 4. Component A — the sizing seam (`_size_variant`)

`analytics/structure_pricer.py`. Replace the `loss_budget`/`linear_notional` signature with
`sizing_spec`:

```python
def _size_variant(pv, sizing_spec, linear_notional=100.0):
    cap = 10.0 * linear_notional
    if sizing_spec.method == "fixed_loss":
        loss_budget = sizing_spec.bankroll * stop_pct   # stop_pct from target_rr, see note
        # ... existing premium-aware cap logic (net-credit → cap; debit → min(.., cap)) ...
    else:  # kelly
        x_star = kelly_fraction_per_notional(sizing_spec.kelly_dist, pv)   # §5
        notional = min(sizing_spec.kelly_lambda * x_star * sizing_spec.bankroll, cap)
    # ... derive net_premium_ccy / max_loss_ccy / payoff_at_target_ccy from notional (unchanged)
```

Note: `stop_pct = move_pct / target_rr` currently lives in the *callers* (app.py, batch_view,
structure_eval, standard_pack) that compute `loss_budget`. Keep that — `_size_variant` receives
the already-computed `loss_budget` for the fixed path (via the spec or a derived field), so the
move→stop logic isn't duplicated. The Kelly path needs no `stop_pct`.

---

## 5. Component B — per-notional Kelly primitive (return-basis generalization)

**The one piece that is genuinely new, not reuse.** `interface/kelly_v2/kelly.py:_returns`
defines `r = (DF·payoff − cost)/cost` — **return per unit premium**. This is correct for long
single options but **divides by premium**, so it is undefined/ill-behaved for spreads, zero-cost
seagulls, and net-credit structures — exactly the comparative set.

Add a **per-notional** return basis. Per unit of structure notional, the P&L fraction in a
terminal-spot scenario `S` is:

```
π(S) = DF · payoff_pct(S) − net_premium_pct          (dimensionless, per unit notional)
```

Kelly with bankroll `W`, notional `N`, `x = N/W`:

```
x* = argmax_x  Σ_k p_k · ln(1 + x · π(S_k))
domain: x < 1 / (−min_k π(S_k))   when the min is negative  (leverage / ruin bound)
N* = x* · W
```

Implementation: add `kelly_fraction_per_notional(dist, pv) -> float` to `analytics/sizing.py`
(or extend kelly.py). It builds `π` over `dist.outcomes` from the variant payoff bridge (§6)
and reuses the existing `minimize_scalar`-bounded optimizer pattern (the `r_min ≤ −1` leverage
clamp at kelly.py:96 generalizes directly — here the clamp is `x < 1/(−π_min)`).

Properties to preserve / assert (tests in §10):
- Long vanilla: `π_min = −net_premium_pct` ⇒ `x* < 1/net_premium_pct` ⇒ premium spent ≤ W.
- Zero/negative premium: well-defined (no division by premium); leverage bound set by the worst
  *scenario* loss, not premium. **This is the principled replacement for the deferred
  "scenario worst-case loss as denominator" fix** (CLAUDE.md Known Issues).
- `E[π] ≤ 0` ⇒ `x* = 0` (no edge, no bet).

**Probability source.** Kelly uses the **elicited terminal-spot distribution** (the PM's edge),
integrated against each variant's **terminal payoff**. It does **not** use the scenario-grid
weights (those are emphasis knobs, not beliefs, and the grid is tail-censored). So sizing and
scoring use two different probability lenses — a conscious decision (§11 D3).

---

## 6. Component C — payoff-bridge completeness + parity

`interface/kelly_v2/pricing.py:base_ccy_payoff_for_trade_rec` already maps: vanilla, 1x1, 1x1.5,
1x2, seagull, european_digital, european_digital_rko, european_rko. **Missing: `1x2x1_spread`**
(added after Kelly v2). Add it (mirror 1x2 + the long wing at `K3 = 2·K2 − K1`).

Risk: the bridge's payoffs must match `analytics/structure_pricer.py` base-ccy conventions
(esp. the european digital = base-ccy cash-or-nothing, payoff-at-target 100%; seagull wing
ratio; zero-cost). Build a **parity test** (mirroring `tests/test_product_model_parity.py`):
for each family × market, assert the bridge's terminal payoff at a set of spots matches the
structure_pricer/product-model intrinsic to tight tolerance. This is the long pole — exotics.

---

## 7. Component D — distribution source & elicitation

Two ways to obtain `kelly_dist`:

1. **Elicited (primary, the user's plan).** Reuse `interface/kelly_v2/elicitation.py` widgets
   embedded in the Trade View intake (CDF or PDF mode). Store the resulting `Distribution` on
   `flow.sizing_spec.kelly_dist`. This is the main UI lift.
2. **View-implied default (fallback, lower friction).** Synthesize a lognormal from the PM's
   existing inputs (target + conviction → drift toward target, width from conviction/ATM vol),
   reusing `interface/kelly_v2/baseline.py` machinery. Gives a real-world distribution *with
   edge* (it differs from market-implied) without extra elicitation. Useful as the default and
   for the Batch sweep (§9), where per-trade elicitation is impractical.

**Agent guardrail.** The agent system prompt forbids stating a Kelly/"optimal size" number
(defers to the Kelly screen). Phase 1: the Agent path keeps `method="fixed_loss"`, so the pack's
notionals stay max-loss-based and the guardrail is untouched. Relaxing it (agent relays
Kelly-sized notionals) is a later, separate decision (D4).

---

## 8. Component E — UI behaviour spec (Trade View)

This is a behaviour change to the **trade-definition flow**, not just a column relabel. Today the
trade is defined in the trade form (`interface/app.py` ~L1129: pair / direction / horizon /
target / prefs) and the R:R lives as a **sidebar slider** (`app.py` ~L221, `st.session_state.
target_rr`); after the engine runs, an "Implied stop / Loss budget" metrics row renders
(`app.py` ~L977–991) and then `render_structure_variants` (~L1056).

### 8.1 The control: Sizing method toggle
- A `Sizing method` **radio** — `Fixed loss` (default) | `Kelly` — placed **inside the trade
  form, where the trade is defined** (the user's point 2/3: terms set in the same place as the
  trade). Persists in `st.session_state.sizing_method`. Default `Fixed loss` ⇒ **zero behaviour
  change** until the PM opts in.
- The R:R slider **moves out of the sidebar into the form**, shown **only in Fixed-loss mode**
  (relabelled e.g. "Loss budget R:R (1:N)"). In Kelly mode it is hidden (R:R is irrelevant).
  The sidebar slider is removed/duplicated — *Decision D6: move vs mirror* (mirror is safer for
  the Agent path, which also reads `target_rr`; cleanest is to keep `target_rr` in session and
  just relocate the widget).

### 8.2 Fixed-loss mode (today's behaviour, made explicit)
- Form shows the R:R input. On run, the existing metrics row renders ("Implied stop (N× R:R)",
  "Loss budget") and the variants table sizes to equal max loss. **No functional change** — this
  path is the regression baseline.

### 8.3 Kelly mode — the new flow
**Ordering dependency:** the distribution is over **terminal spot for the chosen pair/tenor**, so
elicitation only makes sense *after* pair + horizon (+ ideally target) are set. So Kelly mode is
a two-beat flow inside the form:
1. PM sets pair / direction / horizon / target as usual, picks `Kelly`.
2. A **"Your edge (distribution)" expander** appears, reusing `interface/kelly_v2/elicitation.py`.
   It exposes the **existing elicitation modes** (no new math):
   - **CDF mode** — `elicit_from_cdf_anchors` (quantile→price pairs).
   - **PDF / fixed-range bucket mode** — `elicit_from_pdf_buckets` with **fixed-range sigma
     buckets** (`default_sigma_boundaries` / `sigma_boundaries_to_prices`), the existing
     definition the PM already uses.
   - **Bin count is adjustable** — surface the `n_bins` / `n_buckets` control (number_input,
     `>= 2`), so the PM can increase resolution. Defaults to `DEFAULT_N_BINS`.
   Plus a **fractional-Kelly λ slider** (default 0.5, range 0.1–1.0) and a read-only bankroll
   note (W = 100, nominal). Mode + bin-count are session-persisted like the curve itself.

**Default / seed (no dead-end before the PM customises):** the distribution is **pre-seeded with
the view-implied lognormal** (target + conviction → drift/width, via `kelly_v2/baseline.py`), so
the variants table renders immediately on run *without* the PM touching the elicitation. The
expander is "adjust your edge", not "you must elicit first". The PM's edits override the seed.

**Re-seed rules (mirror the existing Kelly-screen invariant, CLAUDE.md "Kelly baseline
reseeding"):** re-seed the baseline distribution **only on real context changes** — pair /
horizon / target / conviction change, or switching *into* Kelly mode. Do **not** re-seed on
ordinary reruns (e.g. λ slider moves, expander toggles), or the PM's elicited curve is wiped.

**Metrics row swap:** in Kelly mode the "Implied stop / Loss budget" metrics (fixed-loss
concepts) are **replaced** by Kelly equivalents — bankroll W, λ, and (optionally) the selected
trade's f* + expected log-growth. The stop/loss-budget block is fixed-loss-only.

### 8.4 Variants table / Structure Evaluation
- **Notional column relabel** by mode: "equal max loss (1:N R:R)" vs "sized to ½-Kelly on your
  distribution".
- **New optional columns in Kelly mode:** per-variant `f*` (the growth-optimal fraction) and
  expected log-growth `g(f*)` (both already computed by `kelly_growth_curve` /
  `kelly_discrete`). These make the *comparison* legible — the PM sees *why* one variant gets a
  bigger bet, not just that it does.
- **Meaning-shift banner:** a one-line caption stating the comparison changed — fixed-loss = "same
  risk, compare reward"; Kelly = "each at its growth-optimal size; bigger notional ⇒ better
  edge/odds, not just bigger". This is important UX: the dollar amounts are no longer
  apples-to-apples in the old sense.
- **Per-variant empty state:** a variant with **no Kelly edge** (`f* = 0`, i.e. `E[π] ≤ 0` under
  the PM's distribution) shows notional 0 / "no Kelly edge — your distribution doesn't favour
  this", greyed, rather than dropping it (it still has a scenario score; it just isn't a bet).

### 8.5 Blocked / empty states
- Kelly mode with **no target** set: the view-implied seed has no centre → show an inline prompt
  "set a target to seed your edge, or elicit a distribution", and fall back to fixed-loss sizing
  for the table until resolved (don't render blank).
- **Degenerate distribution** (all mass one bin / zero variance): Kelly → `SAFETY_F_MAX` clamp
  (already handled by `kelly_continuous`); surface a caption rather than a spike.

### 8.6 Relationship to the existing Kelly Sizing screen
- One source of truth for the distribution: the Trade-View Kelly elicitation and the Kelly
  Sizing screen's "From Trade Rec" mode should **share the same `Distribution` in session state**
  so they don't diverge. *Decision D7:* either (a) Trade View writes the dist that the Kelly
  screen reads (recommended — Trade View becomes the elicitation entry point), or (b) keep them
  independent for now and reconcile later. Reuse, don't fork, `kelly_v2` widgets either way.

### 8.7 State & invariants (carry over from Kelly v2)
- New session keys: `sizing_method`, `kelly_lambda`, `kelly_dist` (the elicited/seeded
  `Distribution`), plus the elicitation widget's own keys.
- **Re-read widget values from `st.session_state` after rendering** (existing Kelly invariant) so
  `+/-` edits don't leave the table on stale inputs.
- Build the `SizingSpec` from these keys **before `_run_engines`**, same point where
  `target_rr` / prefs are assembled today (`app.py` ~L795–798).

### 8.8 Batch & caps
- **Batch (`interface/batch_view.py`):** accepts a `sizing_spec` per batch (default fixed_loss);
  in Kelly mode it uses the **view-implied** distribution per trade (no per-trade elicitation) —
  this is the §9 sweep harness.
- **Bankroll / cap:** `W` nominal (== `LINEAR_NOTIONAL = 100`, visual continuity); the 10×
  notional cap stays as the guardrail (and is the only §9 ranking-flip driver — surface a small
  "capped" marker on any variant that hits it).

---

## 9. The ordinal-ranking analysis (answers "can we ignore the edge cases?")

From §0, ranking is driven by `score_ccy_v = N_v · score_pct_v`.

**Across the fractional-Kelly multiplier λ:** `N_v = λ · f*_v · W`, so λ scales *every* variant's
`score_ccy` by the same factor. **⇒ Ordinal ranking is invariant to λ by construction.** The
*only* thing that breaks this is the **10× notional cap binding asymmetrically** (it clips some
variants' N but not others, breaking proportionality). f* hitting its own leverage bound is
folded into f* and does not depend on λ; `f*=0` (no edge) just zeroes a variant (exclusion, not a
flip).

**⇒ The "extreme cases" = exactly the cap-binding trades, which are detectable per trade.**

**Kelly vs fixed-loss ranking** is a *different* comparison and *should* differ (Kelly ranks by
`f*·score_pct`, fixed-loss by `score_pct/max_loss`; if they agreed, Kelly would add nothing).
Measure the divergence to understand impact — do not "ignore" it.

**Quantification harness** (`tests/test_kelly_ranking_stability.py` + a Batch report):
1. Over the Batch trade set (many pairs/tenors/targets), with a view-implied distribution:
   - Compute per-variant `f*` and `score_pct`.
   - For λ ∈ {0.1, 0.25, 0.5, 1.0}: rank by `score_ccy`; assert **Kendall's τ = 1.0** (identical
     order) on all trades where the cap does not bind; flag and count trades where it does.
   - Report the **top-1 flip rate vs λ** (expected: 0% off-cap).
2. **Kelly vs fixed-loss:** report Spearman/Kendall correlation and top-1 agreement rate across
   the trade set — a descriptive metric for the PM (how often the method changes the pick), not
   an assertion.

Acceptance: if off-cap λ-invariance holds (it must, mathematically — the test guards the
implementation, e.g. that fractional Kelly is `λ·f*` and not a re-solve), and cap-binding flips
are rare and flagged, we treat λ as a pure multiplier and ignore the rest.

---

## 10. Tests (detailed)

### A. `tests/test_sizing_spec.py` (new) — the seam & spec
- `SizingSpec()` defaults to `fixed_loss`, `target_rr=3.0` → `_size_variant` reproduces today's
  numbers **byte-for-byte** (regression lock against the current `test_scenario_pricer.py`
  sizing cases — reuse the same asserted notionals: e.g. vanilla prem 0.02, budget 4.0 → 200).
- Fixed-loss net-credit / low-premium / zero-max-loss cases still hit the 10× cap exactly
  (port the existing `TestSizing` asserts).
- `kelly` method with a trivially-edged distribution produces a positive notional ≤ cap.
- Unknown method → raises.

### B. `tests/test_kelly_per_notional.py` (new) — Component B math
- **Long vanilla bound:** `x* < 1/net_premium_pct`; premium spent `= x*·W·prem_pct ≤ W`.
- **No-edge:** `E[π] ≤ 0` ⇒ `x* == 0` (e.g. a distribution centred at the forward, fair option).
- **Monotonic edge:** shifting the distribution further ITM (more edge) → `x*` strictly increases
  (until the leverage bound).
- **Zero-cost structure** (synthetic seagull with `net_premium_pct≈0`): `x*` is finite and
  well-defined; leverage bound is set by worst scenario π, not premium (no div-by-zero).
- **Net-credit structure** (1x2 with credit): `x*` finite; bound set by worst-case loss; notional
  positive and capped.
- **Parity with kelly_v2 on long options:** for a long vanilla, per-notional `x*·W·prem_pct/W`
  (fraction of bankroll spent on premium) matches `kelly_discrete`'s premium-basis `f*` to tol —
  i.e. the two bases agree where both are valid.
- **Closed-form sanity:** for a near-Gaussian small-edge case, `x*` ≈ Thorp `E[π]/Var[π]`.

### C. `tests/test_kelly_payoff_bridge_parity.py` (new) — Component C
- For each family (incl. `1x2x1_spread`) × {USDTRY, EURPLN, GBPUSD} × call/put, assert the
  bridge terminal payoff at a grid of spots == structure_pricer / product-model intrinsic
  (rel 1e-9). Specifically pin: european_digital → 100% base-ccy at-target; seagull wing ratio;
  zero-cost net premium.
- `base_ccy_payoff_for_trade_rec("1x2x1_spread", ...)` exists and reconciles to
  `+1@K1 −2@K2 +1@K3` with `K3 = 2K2−K1`.

### D. `tests/test_kelly_ranking_stability.py` (new) — Component §9
- Synthetic 3-variant set with hand-chosen `f*` and `score_pct`: assert ranking by `score_ccy`
  is identical across λ ∈ {0.1,0.25,0.5,1.0} **when no cap binds**.
- Construct a case where the cap binds for the top variant at λ=1.0 but not λ=0.1 → assert the
  ranking flips, and that the harness *flags* it (cap-bound marker present).
- Kendall-τ helper returns 1.0 for the off-cap sweep.

### F. `tests/test_kelly_ui_helpers.py` (new) — UI behaviour (logic, not pixels)
Streamlit widgets aren't unit-tested, so factor the behaviour into **pure helpers** and test
those (the app just wires session_state → helper → `SizingSpec`):
- `build_sizing_spec(session_like: dict) -> SizingSpec`: `Fixed loss` + `target_rr` round-trips;
  `Kelly` carries `kelly_dist` + `kelly_lambda`; missing dist in Kelly mode falls back to the
  view-implied seed (not an error).
- `should_reseed(prev_ctx, new_ctx) -> bool`: True on pair/horizon/target/conviction change or
  mode→Kelly; **False** on λ change / plain rerun (guards the "don't wipe the elicited curve"
  invariant).
- `notional_column_label(method)` / `meaning_banner(method)`: return the right copy per mode.
- `kelly_variant_row(pv, dist, lambda, W)`: a no-edge variant (`E[π] ≤ 0`) yields notional 0 +
  "no Kelly edge" flag; an edged variant yields positive notional ≤ cap and a `capped` flag when
  it hits 10×.
- Blocked state: Kelly mode + no target ⇒ helper signals "fall back to fixed-loss for the table".

Manual UI verification (checklist in the PR, not automated): toggle shows/hides R:R vs
elicitation; metrics row swaps; λ slider rescales notionals without reordering (off-cap);
seed renders before any elicitation; editing the curve survives a λ nudge.

### E. End-to-end / plumbing
- `tests/test_flow.py` (extend): `flow.sizing_spec` is reset in `reset()`; set-from-session
  round-trips.
- `tests/test_comparator.py` (extend): `build_comparator_inputs(..., sizing_spec=kelly)` returns
  variants with Kelly notionals and a coherent `score_ccy` ordering; fixed-loss default path
  unchanged (regression).
- **Full-suite regression:** the existing 486 pass with the default `SizingSpec()` (the seam is
  additive). The 1 pre-existing scorer failure stays parked.
- `demo.py` integration check: a `--sizing kelly` flag runs the pipeline with a view-implied
  distribution and prints variant notionals + f* (manual eyeball + smoke).

---

## 11. Risks & open decisions

- **D1 — layering:** import `Distribution` from `interface/kelly_v2` into `analytics/`, or lift it
  into `analytics/`? (Prefer lift + re-export to keep `analytics/` free of `interface/` deps.)
- **D2 — bankroll W:** nominal constant (recommend `W = LINEAR_NOTIONAL = 100` so Kelly notionals
  live on the same visual scale as today's). Document that absolute Kelly notionals are only
  meaningful *relative* to each other unless W is a real bankroll.
- **D3 — two probability lenses:** sizing uses the elicited distribution; scoring uses the
  scenario-grid weights. Coherent but worth a conscious sign-off. Unifying (Kelly *and* scoring
  on one distribution) is a larger, separate change.
- **D4 — agent guardrail:** phase 1 keeps the agent on fixed loss. Relaxing it later means
  letting the LLM relay (not invent) Kelly notionals from tool results.
- **D5 — elicitation friction:** full per-trade elicitation vs the view-implied default. Ship the
  default first; elicitation as an opt-in expander.
- **Probability quality:** Kelly is only as good as the distribution; start at fractional
  λ=0.5 (or lower) — full Kelly is fragile to mis-estimation.
- **D6 — R:R widget move vs mirror:** the R:R control moves from the sidebar into the form
  (Fixed-loss only). Keep `target_rr` in session (the Agent reads it) and just relocate the
  widget, rather than removing the key.
- **D7 — one distribution source of truth:** Trade-View Kelly elicitation and the Kelly Sizing
  screen should share one `Distribution` in session_state (recommend: Trade View writes, Kelly
  screen reads) so they can't diverge. Reuse `kelly_v2` widgets; don't fork them.

---

## 12. Phasing / suggested order

1. **B + C first (pure, testable, no UI):** per-notional Kelly primitive + payoff-bridge parity.
   De-risks the long poles behind tests before any wiring.
2. **A + data model:** `SizingSpec`, `_size_variant` seam, thread through comparator/price_variants
   with the fixed-loss default locked byte-for-byte.
3. **§9 sweep** on the Batch harness with the view-implied distribution — validate λ-invariance
   and measure Kelly-vs-fixed divergence *before* building elicitation UI.
4. **E (UI):** toggle + R:R relabel + elicitation embed.
5. **Labels/surfaces + demo flag.**

A good first PR is **steps 1–3**: the whole Kelly engine + the ranking evidence, behind the
default-off `SizingSpec`, with zero UI and zero change to existing behaviour.
