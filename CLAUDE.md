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
   - `target_z_abs_min/max` — minimum/maximum σ distance of target from forward.

2. **Scoring** — sum affinity scores across 4 dimensions:
   - `target_z_abs` — how far the target is from the forward (no_target / near / moderate / extended / far)
   - `carry_regime` — 0 / 1 / 2 based on |c| vs thresholds in JSON
   - `atmfsratio` — payout ratio of carry-capturing spread (low / medium / high). None when carry_regime=0.
   - `carry_alignment` — compound dimension: `with_{atm_bucket}` or `counter_{atm_bucket}`. Captures the interaction between carry direction and carry magnitude.

All thresholds and scores are in `knowledge/defaults/affinity_scores.json`. Tunable without Python changes. Carry regime thresholds are also loaded from this file — no hardcoded defaults.

Primary structures (overlay_only=False) are capped at max_primary (default 3). Overlays ranked separately.

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
- **atmfsratio** — `compute_market_state(..., surface=<surface>)`: the ATM-fwd / ATM-spot legs (vanillas on the high-carry ccy) price at `vol_at_strike`. `MarketState.surface` carries the surface downstream.
- **Scenario MtM** — `price_scenarios(..., surface=<surface>)` and the max-loss helper `_today_package_value_pct`: vanilla legs reprice under a **sticky-delta** smile — the scenario's ATM vol level plus `smile_skew_spread(K, scenario_fwd, tau)`, i.e. each fixed strike is re-deltaed at the scenario forward. Anchoring to the scenario vol keeps the existing term-structure / ±vol-shock plumbing intact.
- The surface is built once in `conversation/flow.py: _run_engines()` via `build_vol_surface(self.ccy)` (falls back to `None`/flat on an incomplete surface), stored on `MarketState.surface`, and reused by `comparator.build_comparator_inputs(..., smile=...)` and `interface/structure_eval.py` (both the variants table and the scenario-evaluation table).

**Still flat by design:** the digital / RKO / european-RKO pricers (and their scenario MtM legs) stay on flat ATM vol — barrier/binary skew consistency is out of scope. Guards: `tests/test_smile_pricing.py` (entry), `tests/test_vol_surface_refactor.py` (atmfsratio + scenario, flat-byte-identical and smile-moves-vanilla / digital-stays-flat).

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
- **Kelly widget values must be re-read from `st.session_state` after rendering.** This prevents Streamlit `+/-` edits from updating the visible input while leaving charts / edge / Kelly on stale values.
- **Kelly baseline reseeding must only happen on real context changes.** Re-seed on source-mode switch, pair/tenor change, or Trade Rec selection change; do not re-seed on ordinary reruns.

## Kelly screen

`interface/kelly_v2/` now supports two user entry modes:

- `Standalone` — choose a supported pair and tenor from the live snapshot, then elicit a subjective distribution for a single vanilla option.
- `From Trade Rec` — if the current session already has a live `Trade View` recommendation, surface up to 10 concrete recommended variants in a dropdown and size the selected trade.

Implementation notes:

- The Trade Rec linkage must stay variant-level, not family-level, because Kelly needs a fully specified payoff.
- The payoff bridge in `interface/kelly_v2/pricing.py` should stay consistent with the structure pricer’s base-ccy payoff conventions, especially for digitals, seagulls, and zero-cost structures.

## Config system

Three layers merged at session start: base defaults (JSON) → user profile → session overrides (in-memory).

Session overrides are triggered by `[PREF_CHANGE: {"field_path": ..., "value": ...}]` tags emitted by the LLM. The override detector parses these, validates against an allowlist, and re-resolves config. Overrides are ephemeral — they don't persist across sessions.

## Deployment

GitHub: `ash847/macrotool` (private). Streamlit Community Cloud auto-redeploys on push to `main`. The `feature/security-hardening` branch deploys a separate app instance.

### Required Streamlit secrets

```toml
ANTHROPIC_API_KEY = "..."
SUPABASE_URL = "..."
SUPABASE_ANON_KEY = "..."
SUPABASE_SERVICE_KEY = "..."
admin_emails = ["name@fund.com"]

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
