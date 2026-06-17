# MacroTool — Developer Reference

EM FX trade structuring & sizing tool for macro fund PMs. The target architecture is conversational: PM inputs a view in plain English and the tool recommends structures with sizing, entry/exit params, or critiques a PM-supplied structure. The current Streamlit Trade View screen is running a structured silent path while the deterministic pipes are being tested, so the visible UI does not currently depend on live LLM narration.

## Running the project

```bash
.venv/bin/streamlit run interface/app.py   # UI
.venv/bin/python demo.py                   # full pipeline without LLM
.venv/bin/python demo.py --pair USDTRY --direction base_higher --horizon 60
.venv/bin/python -m pytest                 # 373 tests
```

Python 3.13. Venv at `.venv/`. Requires `ANTHROPIC_API_KEY` (sidebar or secrets).

## Architecture — strict layer separation

```
data/           Pydantic models for market snapshots (spot, forwards, vols, df curves)
analytics/      Pure quant computation — MarketState, distributions, no IO
pricing/        Black-Scholes, forwards interpolation, scenario matrices
knowledge/      JSON knowledge base (facts + tunable defaults)
knowledge_engine/  Rule engine — scorer, sizing, critique, conventions
config/         Layered config system with session override support
conversation/   LLM state machine + prompt assembly + tracing
interface/      Streamlit app, charts, Supabase logger, debug log
```

**The single most important rule: LLM narrates only. All numbers are pre-computed by the engine before the LLM is called.** The LLM sees structured text blocks, never raw data objects.

## JSON vs Python

**JSON (`knowledge/`)** holds anything a domain expert should be able to tune without Python:
- `knowledge/facts/{pair}.json` — immutable per-pair conventions (settlement, fixing, instrument type). Never overridden at runtime.
- `knowledge/defaults/affinity_scores.json` — structure scoring: gates, per-bucket scores, thresholds. The main tuning surface.
- `knowledge/defaults/structure_profiles.json` — display names, overlay_only flag, major_risk text.
- `knowledge/defaults/sizing_defaults.json` — Kelly fractions, vol-regime adjustments, tranche schedules, TP rules.
- `knowledge/defaults/critique_defaults.json` — evaluation dimensions for PM structure critique.

**Python** handles all computation, type safety, and orchestration. JSON files are loaded via `knowledge_engine/loader.py` (lru_cache — process restart needed to pick up local edits; Streamlit Cloud redeploy clears cache automatically).

Affinity scores can also be fetched from Supabase (remote config) and fall back to local JSON.

## Conversation flow

State machine in `conversation/flow.py`:

```
INTAKE → DONE (recommend mode, 3 API calls combined)
INTAKE → DONE (critique mode, 3 API calls combined)
DONE   → DONE (follow-up Q&A, unlimited)
```

**INTAKE always makes exactly 3 API calls**, all sent with messages ending in the user turn (never assistant):
1. View extraction — LLM emits `[VIEW: {...}]` tag
2. Validation — market context, carry story, vol regime
3. Structure rec (recommend) or Critique (critique mode)

All three responses are concatenated and recorded as a single assistant message. This is intentional — it prevents message history from ending with an assistant turn.

This remains the intended conversation architecture. The current Streamlit Trade View page can also bypass `flow.advance()` and submit structured inputs directly into `_run_engines()` while we test the deterministic pipeline with the LLM path kept silent on that screen.

**After INTAKE**, the engine has already run:
- `compute_market_state()` → `MarketState`
- `score_structures()` → `StructureSelectionResult`
- `compute_sizing()` → `SizingOutput`
- `compute_flat_vol_distribution()`, `compute_smile_distribution()` (best-effort, non-fatal)
- `evaluate_structure()` (critique mode only)

## Direction convention

Always relative to the **base currency (ccy1)**:
- `"base_higher"` = base appreciates (USD up for USD* pairs; GBP up for GBPUSD; EUR up for EURPLN)
- `"base_lower"` = base depreciates

This convention runs through `TradeView`, `MarketState`, `with_carry`, stop levels, and delta labels.

## Carry and with_carry

`c = ln(fwd/spot) / (σ√T)` — normalised carry.

`with_carry = (c > 0) == (direction == "base_lower")`

**This formula is correct and intentional — do not change it.** Rationale:
- When `c > 0` (fwd > spot): the high-yield currency is the term (quote) currency. Carry trade = long term / short base = `base_lower`. So `base_lower` is with-carry when `c > 0`.
- When `c < 0` (fwd < spot): the high-yield currency is the base. Carry trade = long base = `base_higher`. So `base_higher` is with-carry when `c < 0`.
- Example: USDBRL has `c > 0` (BRL rates >> USD). Long BRL (`base_lower`) = with-carry ✓
- Example: GBPUSD has `c < 0` (GBP rates > USD). Long GBP (`base_higher`) = with-carry ✓

## Affinity scoring system

Replaces the old flat rules engine. Two steps per structure:

1. **Gates** — hard filters. Fail a gate → structure is ineligible regardless of score.
   - `target_z_abs_min/max` — minimum/maximum σ distance of target **from spot** (`target_z_spot`).

2. **Scoring** — sum affinity scores across 5 dimensions:
   - `target_z_abs` — how far the target is from **spot** (no_target / near / moderate / extended / far). Buckets `[0.5, 1.25, 1.75]` are σ-from-spot and manually tunable.
   - `carry_regime` — 0 / 1 / 2 based on |c| vs thresholds in JSON
   - `atmfsratio` — payout ratio of carry-capturing spread (low / medium / high). None when carry_regime=0.
   - `carry_alignment` — compound dimension: `with_{atm_bucket}` or `counter_{atm_bucket}`. Captures the interaction between carry direction and carry magnitude.
   - `structure_constraint` — PM preference gate/penalty (5th dimension).

**Two `target_z` fields on `MarketState` — distinct anchors for distinct purposes:**
- `target_z_spot = ln(target/spot)/(σ√T)` — **spot-anchored**, used by the scoring/selection layer. The "how far is the PM's target from where we are now" question.
- `target_z = ln(target/fwd)/(σ√T)` — **forward-anchored**, used by construction (put/call direction, variant eligibility gates in `structure_pricer`). Identity: `target_z = target_z_spot − c`.
- `put_call` is always derived from the forward (`"Call" if target > fwd`). **Never use `target_z_spot` for construction or option direction.**

All thresholds and scores are in `knowledge/defaults/affinity_scores.json`. Tunable without Python changes. Carry regime thresholds are also loaded from this file — no hardcoded defaults.

Primary structures (overlay_only=False) are capped at max_primary (default 3). Overlays ranked separately.

## Scenario grid — spot-anchored

`analytics/scenario_generator.py` builds a deterministic grid. The grid is **spot-centred**: the "no-move" column `S` holds spot unchanged, and σ-offset columns (`−½σ`, `−1σ`) are measured from spot. `scenario_fwd` is derived per-cell via CIP (`scenario_spot · e^{(r_d−r_f)·τ}`) and decays toward spot at expiry — it is used for MtM pricing only.

**Two anchors, deliberately separated:**
- **Spot** anchors the grid geometry (where scenarios sit, what "no move" means).
- **Forward** anchors pricing and construction (Black-76 MtM, `direction = sign(K/F)`).

Column set: `[S, t%→K, K−½σ, K, K+½σ, −½σ, −1σ, Δvol]`. `S` = spot unchanged (displayed as "No move" in the UI). `K−½σ`/`K`/`K+½σ` are **target-anchored** (`K±½σ = K·e^{±direction·½σ_t}`): `K−½σ` = half a sigma *short* of the target in the view's direction (a partial move that undershoots), `K+½σ` = overshoot. `t%→K` tracks progress spot→K with progress fraction `p = elapsed/T` (so 25%T→0.25, 50%T→0.50, early row→14/365 ÷ T).

Rows: `[2w, 25%T, 50%T, Expiry]`. The early **2w** row (was `1w`) only appears for tenors `> 6 weeks` (`valid_grid_rows`) and now exposes the **full directional move set** (same columns as the interim rows) so the "fast move" path — reaching/overshooting the target inside two weeks — is visible rather than greyed out. `−½σ`/`−1σ` stay spot-anchored adverse cells.

**Per-row roll-down forward:** the `S` cell's `scenario_fwd` gives the carry-derived forward at each time checkpoint. It decays toward spot at expiry and is surfaced in the UI row headers (`25%T · fwd 41.2`) so the carry tailwind remains visible without being the grid's centre.

**Direction** (`direction = sign(K/F)`) stays forward-relative so up-weighted "favourable" cells align with where the forward-constructed structure actually profits. On a carry-cross target (target between spot and forward), a put structure's adverse cells sit above the forward — this is P&L-correct, not a bug.

Config: `knowledge/defaults/scenario_definitions.json` (`_grid_cols`, `scenario_column_descriptions`, per-cell multipliers). JSON is the re-tune surface; Python handles computation.

## Rate context and df curves

`pricing/forwards.py: rate_context_for_snapshot()` builds `RateContext` for any pair:
- Identifies base currency from `pair[:3]`
- Reads `r_f` (base rate) from the appropriate df curve: `usd_df_curve` (USD base), `eur_df_curve` (EUR base), `gbp_df_curve` (GBP base)
- Derives `r_d` (quote rate) via CIP from the forward

To add a new base currency: add a `{ccy}_df_curve` field to `CurrencySnapshot` in `data/schema.py` and a branch in `rate_context_for_snapshot`.

NDF outrights already embed the full interest rate differential — use them as-is for CIP derivation. No additional fixing-lag adjustment needed.

## Supported pairs

| Pair | Type | Base DF curve | Character |
|------|------|--------------|-----------|
| USDBRL | NDF | usd_df_curve | High carry, topside skew |
| USDTRY | NDF | usd_df_curve | Very high carry, strong topside skew |
| EURPLN | Deliverable | eur_df_curve | Moderate carry, symmetric skew |
| GBPUSD | Deliverable | gbp_df_curve | Low carry (G10), mild negative skew |

Other pairs in snapshot (EURUSD, USDCNH, USDMXN, USDJPY) are not yet wired into the conversation flow.

## Option pricing and smile vol

`analytics/structure_pricer.py: price_variants()` prices every structure variant
in Black-76 and quotes premium/payoff as a fraction of base-ccy (USD) notional.

**Premium basis (load-bearing invariant):** `black76(F,K,T,σ,DF_d) / spot == DF_f × (forward_value / forward)`. Black-76 returns a premium in *quote* ccy discounted at the *quote* rate `r_d`; dividing by **spot** (not forward) collapses that into the *base* rate `r_f`, so the displayed `net_premium_pct` is base-ccy-notional %, effectively USD-discounted. This is correct and intended. The buggy basis (quote-discounted value ÷ forward) lands ~one quote-DF away. `tests/test_premium_basis.py` pins this on a high-carry TRY-like market — do not "fix" the `/spot` to `/fwd`.

**Pluggable vol surface (`analytics/vol_surface.py`):** all vanilla pricing depends on a `VolSurface` Protocol — `vol_at_strike(K, F, horizon_days)` and `vol_at_delta(delta, horizon_days)`.
- `SmileInterpolator` is the cubic-spline build method; its resolution/interpolation knobs (`pillars`, `call_deltas`, `tenor_days`, `bc_type`, `delta_clip`) are constructor params with the legacy values as defaults.
- `FlatSurface(atm_vol)` is the degenerate surface — one ATM vol everywhere — so the flat path is just a surface, not a separate branch. Pricing against it is byte-for-byte identical to the legacy scalar-vol path.
- `build_vol_surface(ccy, method="cubic_spline", **params)` is the only construction site. Add a new build method (e.g. `"sabr"`) by adding a class that satisfies the Protocol plus a branch here — no call-site changes.

**Where the smile applies (every vanilla, inception → completion):**
- **Entry pricing** — `price_variants(..., smile=<surface>)`: leg-based structures (`vanilla`, `1x1_spread`, `1x1.5_spread`, `1x2_spread`, `seagull`) price each leg at its own vol (delta legs `vol_at_delta`, strike legs `vol_at_strike`). A `_VolModel` wrapper hides the branch.
- **European package entry** (`european_digital`, `european_rko`) — also smile-aware at entry via a strike→vol seam (`vol_fn`, a plain `Callable[[float], float]` so the `pricing/` layer stays free of any `analytics` dependency). `european_rko`'s two vanilla legs price at their own strike vol; the **digital** (standalone and the strip inside `european_rko`) prices at the skew-consistent value `DF·N(d2(σ(K))) − vega·σ′(K)` for calls (`+vega·σ′` for puts), i.e. the strike-derivative of the call including the skew-slope term. `σ′(K)` is a central difference on `vol_fn`; `vega` is `black76_vega`. `european_digital`'s strike-solving bisection runs against the smile digital. A flat/`None` surface collapses `vol_fn` and `σ′→0`, reproducing the legacy flat price byte-for-byte. Guards: `tests/test_pricing.py` (analytic route-2 formula via a synthetic linear smile + call/put parity), `tests/test_vol_surface_refactor.py` (flat-identical + smile-moves for both packages).
- **atmfsratio** — `compute_market_state(..., surface=<surface>)`: the ATM-fwd / ATM-spot legs (vanillas on the high-carry ccy) price at `vol_at_strike`. `MarketState.surface` carries the surface downstream.
- **Scenario MtM** — `price_scenarios(..., surface=<surface>)` and the max-loss helper `_today_package_value_pct`: vanilla legs reprice under a **sticky-delta** smile — the scenario's ATM vol level plus `smile_skew_spread(K, scenario_fwd, tau)`, i.e. each fixed strike is re-deltaed at the scenario forward. Anchoring to the scenario vol keeps the existing term-structure / ±vol-shock plumbing intact.
- The surface is built once in `conversation/flow.py: _run_engines()` via `build_vol_surface(self.ccy)` (falls back to `None`/flat on an incomplete surface), stored on `MarketState.surface`, and reused by `comparator.build_comparator_inputs(..., smile=...)` and `interface/structure_eval.py` (both the variants table and the scenario-evaluation table).

**Still flat by design:** the path-dependent `rko` and `european_digital_rko` entry pricers (both currently gated via `enabled: false` in `structure_profiles.json`), and **all scenario-MtM legs** for the digital / RKO / european-RKO family — barrier/binary *scenario* skew consistency is out of scope. Guards: `tests/test_smile_pricing.py` (entry), `tests/test_vol_surface_refactor.py` (atmfsratio + scenario, flat-byte-identical and smile-moves-vanilla / digital-scenario-stays-flat).

**Digital smile-arbitrage guard (shipped):** `digital_call`/`digital_put` raise `SmileArbitrageError` (in `pricing/digital.py`) when the smile-implied digital escapes `[0, DF]` — the signature of a local butterfly arbitrage (negative risk-neutral density), typically cubic-spline overshoot in the extrapolated wings. `price_variants(..., warnings=[...])` catches it, drops the affected `european_digital`/`european_rko` variant, and records a note; `interface/structure_eval.py` renders a banner at the top of the Structure variants section. This is a *consumer-side* tripwire on the digital path, not a surface validator — it only fires where a digital is actually priced and in-range `[0,DF]` is necessary-not-sufficient, so it misses arbs that only corrupt vanillas/spreads, mild local butterflies, and all calendar arbs.

**European digital settlement — base-ccy cash-or-nothing (recommendation/treatment layer):** as a *recommended structure* the European digital pays a **fixed 1 unit of base ccy** (e.g. USD) if it finishes ITM, so its premium is a base-ccy % and its **payoff-at-target is exactly 100%** (not the old `spot/target`, which printed e.g. 110% on a TRY-strengthening target). `analytics/structure_pricer.py: _digital` builds this from the asset-or-nothing identity — `AON_call = call(K) + K·digital_call(payout=1)`, `AON_put = K·digital_put(payout=1) − put(K)`, then `/spot` — so it reprices in base-ccy terms while **reusing the unchanged quote-cash `digital_call/put` primitive** (and thus the smile correction + arb guard). `scenario_pricer.py` marks the digital on the same USD-cash basis (ITM ⇒ 100% of notional). **`european_rko` is unaffected:** its decomposition strip still uses the quote-cash (ccy2 / TRY) digital with `payout=1.0`, exactly as the `(H−K)` quote-cash step requires — the currency change is *only* at the `european_digital` structure level, never in `pricing/`. The strike-solving bisection brackets symmetrically **around the forward** (`K ∈ [F·e^{−4σ√T}, F·e^{+4σ√T}]`), not spot, so on high-carry pairs (fwd far from spot) the 10/20/30% strikes that legitimately sit between spot and the forward — downside *to the forward* — are reachable instead of railing to a spot-anchored bound. Guards: `tests/test_pricing.py` (base-ccy payoff/premium `DF_f·N(d1)` identity; deep-carry put + inverted-carry call assert distinct strikes at 10/20/30%).

### Roadmap / wishlist

- **Arbitrage-free vol surface via SSVI (deferred — `build_vol_surface(method="ssvi")`).** Replace the cubic-spline-in-delta surface with an SSVI parametrization, calibrated to the 5 delta pillars per tenor.
  - *What it gives us:* a globally arbitrage-free surface by construction (no butterfly, no calendar), so digitals are always in `[0, DF]` and the smile-arb guard above becomes a belt-and-braces backstop rather than something that fires. Proactive, whole-surface correctness — covers vanillas/spreads and calendar, which the guard does not. Add behind the existing `build_vol_surface(method=...)` seam → zero call-site changes.
  - *Trade-offs:* (1) **calibration is the hard part** — with only 5 quotes/tenor, raw 5-param SVI overfits, so use SSVI (θ(T) + φ(θ) + ρ); nonlinear solve that can fail to converge, needs robust seeds/fallback and the no-arb inequalities enforced during the fit, plus a delta→log-moneyness conversion. (2) **Full-surface reprice** — SSVI *smooths* rather than interpolating exactly, so it no longer reprices each broker pillar to the quote, and it moves **every** smile-based number in the tool (atmfsratio, all spread legs, scenario MtM), demanding broad re-validation and PM sign-off on "we no longer hit the 25Δ quote exactly." (3) keep a slim build-time no-arb assertion on the calibrated SSVI params.
  - *Decision (this session):* hold off. The shipped guard is sufficient for the highest-value slice (skew-sensitive digitals). Before committing to SSVI, do the one-afternoon empirical scan of how often / at what strikes the live spline actually violates no-arb — that sizes the prize and sets fit tolerances.
- **Generalize the pricing seam to a `PricingContext` (deferred).** The European-package work threads a `vol_fn` callable through individual pricer signatures. The longer-term ergonomic refactor is to pass one cohesive context object (or lean on `MarketState`) instead of N loose scalars + surface, across all pricers. Low-stakes (pure ergonomics, feature already works); no deadline.

## Logging and observability

- **Langfuse** — one trace per session, one generation per LLM call (step names: `INTAKE_view_extraction`, `INTAKE_validation`, `INTAKE_structure_rec`, `INTAKE_critique`, `DONE`). No-op safe if keys not set.
- **Supabase** — split client model in `interface/supabase_logger.py`:
  - anon key for `queries` / `feedback` inserts
  - service key for engine config reads and admin-only reads/writes
- Both are initialised from Streamlit secrets injected into `os.environ` before session state init.

## Key invariants

- **Messages must end in a user turn** before any API call. Never bypass `flow.advance()`.
- **Generator must be fully consumed** before calling `advance()` again — state updates happen at exhaustion.
- **Distributions are non-blocking** — if they fail, the conversation continues without them.
- **carry_regime 0 → atmfsratio is None** — do not compute carry-capturing spread premiums in noisy regimes.
- **Vol surface delta labels are always relative to the base currency.**
- **`target_rr` must be cleared in `reset()`** alongside all other view state.
- **Scoring tuple type is `float`** — affinity scores use fractional values.
- **`target_z_spot` drives scoring; `target_z` (forward) drives construction.** Never swap them. `put_call` and variant eligibility (`min_target_z` in `structure_pricer`) always use the forward-anchored `target_z`. The scorer's `target_z_abs` bucket and gates read `target_z_spot`.
- **Scenario grid is spot-centred; pricing is still forward-anchored.** The `S` (no-move) cell = spot unchanged; `scenario_fwd` is derived per-cell for MtM. `direction` is forward-relative (`sign(K/F)`). Do not re-anchor MtM pricing to spot.
- **Kelly widget values must be re-read from `st.session_state` after rendering.** This prevents Streamlit `+/-` edits from updating the visible input while leaving charts / edge / Kelly on stale values.
- **Kelly baseline reseeding must only happen on real context changes.** Re-seed on source-mode switch, pair/tenor change, or Trade Rec selection change; do not re-seed on ordinary reruns.

## Kelly screen

`interface/kelly_v2/` now supports two user entry modes:

- `Standalone` — choose a supported pair and tenor from the live snapshot, then elicit a subjective distribution for a single vanilla option.
- `From Trade Rec` — if the current session already has a live `Trade View` recommendation, surface up to 20 concrete recommended variants (the first `TRADE_REC_DROPDOWN_LIMIT` of the Trade View `selector_result.shortlist`, in the same order) in a dropdown and size the selected trade.

Implementation notes:

- The Trade Rec linkage must stay variant-level, not family-level, because Kelly needs a fully specified payoff.
- The payoff bridge in `interface/kelly_v2/pricing.py` should stay consistent with the structure pricer’s base-ccy payoff conventions, especially for digitals, seagulls, and zero-cost structures. **Note:** the European digital is now a base-ccy cash-or-nothing structure (fixed base-ccy payout, payoff-at-target 100%) — any Kelly digital payoff must match this, not the old `spot/target` basis. (`interface/kelly_v2/` is not present on every branch.)

## Config system

Three layers merged at session start: base defaults (JSON) → user profile → session overrides (in-memory).

Session overrides are triggered by `[PREF_CHANGE: {"field_path": ..., "value": ...}]` tags emitted by the LLM. The override detector parses these, validates against an allowlist, and re-resolves config. Overrides are ephemeral — they don't persist across sessions.

### Per-user scenario-weights profiles

Scenario weights (`scenario_definitions`) can be **forked per user** for a select few, so PMs can iterate on their own weighting and surface differences of opinion. Admin-managed via the **Profile** picker on the Scenario Weightings page.

- **Gating:** the `personal_weights_emails` secret allowlists who may have a personal profile. The check lives in `interface/security.py:can_have_personal_weights` and is enforced **inside the loader** (single source of truth) — a de-allowlisted user reverts to global immediately even if a personal row lingers.
- **Storage:** same `config_history` table, composite key. Global keeps `"scenario_definitions"`; a personal profile uses `f"scenario_definitions::{email}"` (`interface/supabase_logger.py:personal_weights_key`). No schema change; versioned/audited like any other config.
- **Resolution** (`knowledge_engine/scenario_weighter.py:load_scenario_weights_config(user_email)`): personal (allowlisted + non-sentinel config) → global → local JSON.
- **Revert to global:** writes a sentinel personal row `{"_inherit_global": true}`; the loader treats it as "behave as global" (reversible — Save again to re-fork). The UI exposes a "Revert this user to global" button.
- **Cache:** `_weights_cache`/`_weights_source` are **keyed by resolved profile key**, not a single global. This is load-bearing — Streamlit Cloud runs one process for all sessions, so an unkeyed cache would bleed one user's weights to another. `clear_scenario_weights_cache(profile_key=None)` clears all or one.
- **Threading:** `user_email` flows interface → engine via `flow.user_email` (set before `_run_engines`) and the `user_email=` param on `compute_family_weights`, `build_comparator_inputs`, and `build_recommendation_pack`. Trade View and Batch use the logged-in user's profile; the Agent path defaults to global for now (param plumbed through `build_pack`). Default `None` everywhere → global, so all existing callers are unchanged. Guard: `tests/test_personal_weights.py`.

## Deployment

GitHub: `ash847/macrotool` (private). Streamlit Community Cloud auto-redeploys on push to `main`. The `feature/security-hardening` branch deploys a separate app instance.

### Required Streamlit secrets

```toml
ANTHROPIC_API_KEY = "..."
SUPABASE_URL = "..."
SUPABASE_ANON_KEY = "..."
SUPABASE_SERVICE_KEY = "..."
admin_emails = ["name@fund.com"]
personal_weights_emails = ["pm1@fund.com"]   # optional: users allowed a personal scenario-weights profile

[auth]
redirect_uri  = "https://<app-slug>.streamlit.app/~/+/oauth2callback"
cookie_secret = "<64-char hex string — run: openssl rand -hex 32>"

[auth.google]
client_id            = "<from Google Cloud Console>"
client_secret        = "<from Google Cloud Console>"
server_metadata_url  = "https://accounts.google.com/.well-known/openid-configuration"
```

The Google OAuth app (Cloud Console → APIs & Services → Credentials) must have the exact `redirect_uri` above registered as an Authorised Redirect URI.

### Auth architecture notes

- `[auth]` holds shared config (`redirect_uri`, `cookie_secret`). Provider-specific keys go under `[auth.google]`. Mixing them causes "missing keys" errors.
- The redirect URI for Community Cloud is `/~/+/oauth2callback`, **not** `/oauth2callback`. Wrong path → persistent `MismatchingStateError`.
- `cookie_secret` is used only to sign the provider JWT token and the final identity cookie. OAuth state lives in server-side memory (`_STARLETTE_AUTH_CACHE`), not in a cookie — so changing `cookie_secret` does not fix state-mismatch errors.
- `st.login("google")` must match the subsection name `[auth.google]`. Calling `st.login()` (no argument) uses the flat `[auth]` default-provider format instead.
- authlib ≥ 1.6.6 has a known regression that breaks `st.login()`. Pin to `authlib==1.6.5` in `pyproject.toml`.

### Operational notes

- App auth is fail-closed. If the `[auth]` block is missing or incomplete, `interface/security.py` stops the app at startup with a config error.
- Adding or removing an admin: edit `admin_emails` in Streamlit Cloud secrets and restart the app.
- **Python source changes require a `pyproject.toml` version bump** to trigger Streamlit Cloud package reinstall. JSON file changes deploy immediately.
- **After changing `pyproject.toml` dependencies, run `uv lock`** and commit the updated `uv.lock`. Community Cloud may use the lockfile; a stale lockfile overrides `pyproject.toml` pins.
- `SUPABASE_SERVICE_KEY` now backs app writes (`queries`, `feedback`) as well as engine/admin operations. `SUPABASE_ANON_KEY` is retained only for direct REST smoke tests and Security Advisor validation.

## PM preference roadmap

1. **UI capture first** ✅ — PM preference inputs were first added on the intake form.
   - `Primary objective`: `Balanced`, `Keep cost low`, `Hold up if the path is slow/noisy`, `Keep risk clean`
   - `Structure constraint`: `No restriction`, `Avoid capped structures`, `Avoid complex structures`, `Avoid tail-risky structures`
   - `Trade management style`: `Standard hold`, `May monetise early`, `Need defendable mark-to-market`
   - Note: "Keep upside if I'm very right" removed — redundant with "Avoid capped structures".

2. **Selection plumbing** ✅ — `Structure constraint` wired into affinity scoring as a 5th dimension.
   - `structure_constraint` field added to every structure in `affinity_scores.json`.
   - Bucket = the preference string directly (no numeric conversion needed).
   - Default scores: 0 = compatible, −5 = penalised (strong soft preference, but not a hard guarantee by itself).
   - Editable via **Structure Constraint** tab in the Structure Selection page.
   - `score_structures(market_state, structure_constraint=...)` — defaults to "No restriction" so all existing callers are unchanged.
   - `flow.structure_constraint` set from `pref_structure_constraint` session state before each engine run.
   - Additional hard gates now enforce the strongest exclusions directly in Python:
     - `Avoid complex structures` excludes `seagull`, `rko`, `european_rko`, `european_digital_rko`
     - `Avoid tail-risky structures` excludes `seagull`, `1x2_spread`

3. **Context plumbing** ✅ — `Primary objective` and `Trade management style` are routed into context selection.
   - 5 preference-aware contexts at the top of `scenario_weights.json`: `classic_carry`, `cheap_carry`, `conservative_carry`, `delta_carry`, `big_move`. First-match selection — exactly one context fires per trade.
   - New supported fields in conditions: `primary_objective`, `trade_management`. New `in` operator accepts a list of allowed values (used for enum prefs, e.g. `primary_objective in ["Balanced", "Hold up if the path is slow/noisy"]`).
   - `compute_family_weights(ms, primary_objective="Balanced", trade_management="Standard hold")` — defaults preserve existing behaviour.
   - `flow.primary_objective` / `flow.trade_management` set from session state before each engine run, alongside `flow.structure_constraint`.
   - All 5 new contexts ship with empty `adjustments: {}` — they fire and surface in the UI but don't yet bend weights. Tune via the Context weights tab.
   - Old market-state-only contexts are retained as fallbacks for cases the 5 don't cover (counter-carry, carry=0, no-target, edge tenors). The three that required `with_carry=true` (`carry_capture`, `directional_with_carry`, `carry_momentum_extended`) are unreachable for typical preferences and remain dormant.
   - Trade View shows weighted P&L for each structure both **(baseline)** (1/8 each) and **(context)** (after the active context's adjustments) so the deviation is visible per structure.

Design intent:
- `Balanced` / `No restriction` / `Standard hold` remain the defaults — current behaviour unchanged when no PM preference is chosen.
- Selection-layer constraints live alongside `affinity_scores.json`.
- Context/evaluation-layer preferences will live alongside `scenario_weights.json`.

## Known Issues

- Structure Evaluation variant expander headers are still partially markdown/styling-sensitive in Streamlit. `$` amounts do not render reliably in the header and some fragments still pick up red text styling. Preferred fix: keep expander titles plain (variant + strikes + maybe notional) and move weighted P&L summary to the first line inside the expander body.
