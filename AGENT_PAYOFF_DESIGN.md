# Design — engine-authored payoff-region text for the Agent pack (Step 4)

Status: **SHIPPED** on branch `feature/agent-payoff-narration` (off `main`).
Implemented: `knowledge_engine/payoff_profile.py` + 3 render sites + prompt rewrite +
`tests/test_payoff_profile.py`. Steps 2 (ERKO tag) and 3 (premium sign) landed alongside.

## Open decisions / TODO

- **#14 — regime-text vs eval-numbers precedence (OPEN, deferred by PM).** When the agent
  explains "why", the prompt tells it to *synthesize* the CONTEXT GUIDANCE regime text **and**
  the per-structure scenario `findings` / `deciding_axis` — so today it blends both, by design.
  PM to decide whether the explanation should be *anchored* on the per-structure eval numbers
  (regime text as framing) or vice-versa. One-paragraph prompt precedence rule once decided.
  Recommendation on file: anchor on the eval numbers, use regime prose only to frame.
  **Status: PM is testing the shipped changes first before deciding.**
- **Step 1 — Opus flip:** PM is setting `ANTHROPIC_MODEL = "claude-opus-4-8"` as a Streamlit
  secret (no code change).
- **Terminology glossary (Step 5, deferred):** a banned/preferred-terms list in the system
  prompt for the pure-lexical residue (#4 "high-touch", #6 "tighter KO"). Not added — PM
  testing whether Opus + the shipped premium-sign labels are enough first.

---

Original design (as signed off) follows.

## 1. Problem

The agent gets the *primitives* of a structure (per-leg side/strike/premium, a few
`findings` tags) plus a **prohibition**: the system prompt says *"NEVER invent payoff
geometry, exposure regions, or which side has residual exposure — you will get the
levels and the direction wrong"* ([agent_flow.py:33](agentic/agent_flow.py#L33)).

When a PM asks "what happens if spot goes through X / where do I lose / is the tail
capped," the model must answer, has no engine-authored payoff text to quote, and a bare
"don't" loses to its urge to help — so it confabulates. This is the direct cause of the
feedback items:

- **#8** "thinks a 1x2 is short below K2" — it's short *above* the upper breakeven; below
  K1 it simply expires worthless.
- **#5** "assumes selling the 2-leg necessarily creates negative bleed."
- **#13** target/barrier swap ("knocks out through 4.0283" when the target is 4.21).
- **#3** "roll-down offsets directional losses" (counter-carry).
- **#1** "premium collection makes sense in a big-move regime."
- **#7 / #12** ERKO path-dependence (also fixed by the Step-2/3 bug fix; the payoff line
  reinforces it by stating product nature positively).

**Goal:** replace the prohibition with a deterministic, engine-authored **PAYOFF** line
per structure that the model *relays verbatim*, the same way it already relays legs and
numbers. Turn "don't author geometry" into "here is the geometry — relay it."

## 2. Design principle — compute, don't author

The load-bearing facts (which side is the tail, where the breakevens are, does it pay
above or below, is it path-dependent) are **directional** and must be *computed from the
actual priced legs*, never templated per family. A per-family template silently inverts
on a put-vs-call flip or a variant change; a computed profile reads the real legs and
cannot disagree with the per-leg breakdown shown right above it.

This mirrors the house pattern in `knowledge_engine/structure_attributes.py`: derive facts
from numbers, then render a fixed vocabulary over them. Here the "numbers" are the signed
leg notionals + strikes we already have at render time.

## 3. New pure module — `knowledge_engine/payoff_profile.py`

A pure function that turns the priced legs into a small, family-agnostic profile.

```python
@dataclass(frozen=True)
class PayoffProfile:
    profit_region: str        # e.g. "between 5.57 and 6.10", "above 5.57", "at/near 5.90"
    breakevens: tuple[float, ...]   # already-shown numbers; 0, 1, or 2
    max_payoff_where: str     # "at 5.90 (the short strike)", "above 5.57 (uncapped)"
    tail: Literal["capped", "uncapped_up", "uncapped_down"]
    tail_where: str | None    # "on a move above 6.10 (through the 2× short strike)"
    product_nature: Literal["expiry_only", "path_dependent", "binary_expiry"]
    premium_flow: Literal["debit", "credit", "zero_cost"]
    pre_expiry_note: str | None = None   # FORK FLEX — reserved for a one-line pre-expiry
                                         # / MtM story (e.g. ERKO "comes back to life",
                                         # counter-carry roll-down). Left None for now;
                                         # the renderer emits it only when populated, so
                                         # turning it on later is a pure additive change.
```

**Fork flex (per sign-off):** the pre-expiry / MtM narrative is scoped in but **not
populated**. `PayoffProfile` carries an optional `pre_expiry_note` slot and the renderer
appends it *only when non-None*, so the terminal-only line ships now and a future
pre-expiry note is a one-field, additive change with zero churn to the call sites or the
prompt.

### 3a. Computation

For the **vanilla-leg families** (`vanilla`, `1x1_spread`, `1x1.5_spread`, `1x2_spread`,
`1x2x1_spread`, `seagull`, `risk_reversal`) build the exact terminal payoff from the legs
we already iterate in `render._legs_breakdown`:

```
P(S_T) = Σ_leg  sign(notional)·|ratio| · intrinsic(right, S_T, K_leg)  −  net_premium
```

Sample `P` on a spot grid spanning `[min(K)·(1−δ), max(K)·(1+δ)]` and read off, purely
geometrically:

- **breakevens** — sign changes of `P` (prefer the already-computed `variant.breakeven`
  when it's the single relevant one; the grid catches the second breakeven a ratio spread
  adds).
- **profit_region** — the contiguous span(s) where `P > 0`.
- **max_payoff_where** — argmax of `P` (a point/plateau for capped, "uncapped" when the
  slope stays positive at the grid edge).
- **tail** — sign of the slope of `P` at each grid edge: negative slope at the top edge on
  a net-long-call-count-negative structure ⇒ `uncapped_down` with `tail_where` naming the
  strike the tail opens beyond (for a 1x2, `K3_upper = 2·K2 − K1` region). `capped` when
  both edges are bounded.

This is the piece that kills #8/#5/#1: the tail direction and the "worthless below K1 vs
short above the upper breakeven" facts fall straight out of the sampled payoff and can't
be stated backwards.

### 3b. Non-piecewise-linear families (explicit branches, still computed)

- **`european_digital`** — step payoff. `product_nature = binary_expiry`, `tail = capped`,
  `profit_region` = "above/below `K` at expiry", payoff-at-target is the fixed 100%
  base-ccy already in the pack. No breakeven (matches `variant.breakeven is None`).
- **`european_rko`** — vanilla payoff *between strike and barrier*, **zero beyond the
  barrier**, checked **only at expiry**. `product_nature = expiry_only` (read from the
  `path_dependent=False` profile flag — see Step 2/3), `profit_region` = "between `K` and
  the barrier `H`", and the payoff line states the barrier explicitly so the model stops
  swapping it with the target (#13).
- **`rko` / `european_digital_rko`** — `path_dependent=True`; both are `enabled=false` so
  they never reach the recommended set, but the profile still handles them for a PM-named
  `price_structure` request, stating `product_nature = path_dependent`.

Product nature comes from the `path_dependent` bool already in
`structure_profiles.json` — the data model already knows the truth; we are only surfacing
it. (This is the same flag whose *mis*-application in `structure_attributes._BARRIER`
causes #12; Step 2 fixes that tag, Step 4 states the flag positively.)

## 4. Rendering — one generic templater, not N per-family strings

A single `render_payoff(profile) -> str` composes the profile into one labelled line, e.g.:

```
     PAYOFF: pays between 5.5694 and 6.1030; best at 5.9000; net debit; turns
     negative only on a move above 6.1030 (through the 2× short strike); settles on
     the expiry level only.
```

For an ERKO:

```
     PAYOFF: pays like a put between the 4.2100 strike and the 4.0283 knock-out; pays
     nothing if spot finishes beyond 4.0283; European — the barrier is tested at
     expiry only, not on the path.
```

Every number in the line is one the pack already prints (strikes, breakeven, barrier,
premium sign) — the templater only *connects* them. No new number is minted, so the
render stays inside the "numbers come from a tool result" invariant.

## 5. Integration points

All three render sites gain one `PAYOFF:` line, built from the priced legs already in hand:

- `render._legs_breakdown` caller in `render_pack` (recommended set, top-3).
- `render_recommended` (a recommended structure restated via `price_structure`).
- `render_priced_structure` (a PM-named off-menu structure) — uses the *same*
  `payoff_profile()` on its `priced_structure.priced_legs`, so an off-menu 1x2 is described
  in the identical vocabulary as a recommended one.

`payoff_profile()` takes the product-model `priced_legs` + `variant` + `structure_id`
(for the profile flag / family branch). All three sites already have both objects.

## 6. Prompt change (small, and the point of the whole exercise)

In `agent_flow.SYSTEM_PROMPT`, replace the RISK-block prohibition
*"NEVER invent payoff geometry, exposure regions, or which side has residual exposure"*
with a **positive** instruction:

> Each structure prints a `PAYOFF:` line stating where it makes and loses money, where the
> tail is, and whether it settles on the expiry level only or is path-dependent. Relay that
> line's facts verbatim — the profit region, the breakevens, the barrier, the tail side. Do
> NOT author payoff geometry, exposure regions, breakevens, or path/expiry behaviour
> yourself: if the `PAYOFF:` line does not cover what the PM asks, price the structure or say
> so — never reconstruct it from memory.

The risk-on-request rule (surface `risk (engine)` only when asked) is unchanged; PAYOFF is
neutral geometry, not the risk caveat, so it can render by default.

## 7. IP-cleanliness

The profile is pure terminal-payoff geometry (strikes, breakevens, premium sign, product
nature) — **no scores, weights, thresholds, or scenario aggregates**. It is strictly less
sensitive than the `findings` tags already shipped. Nothing new leaks.

## 8. Testing

New `tests/test_payoff_profile.py`:

- **Directional correctness (the regression guards for the feedback):**
  - 1x2 call: `tail == "uncapped_down"`, `tail_where` names the region above the upper
    breakeven; profit region does **not** extend below K1 — pins #8/#5.
  - Vanilla / 1x1 debit: `tail == "capped"`, single breakeven == `variant.breakeven`.
  - Seagull: profit region between the two bought strikes; wing names the funded tail.
  - ERKO: `product_nature == "expiry_only"`, profit region bounded by the barrier, barrier
    ≠ target in the string — pins #13/#7/#12.
  - Digital: `binary_expiry`, no breakeven, payoff-at-target 100%.
- **Consistency:** every number in the rendered PAYOFF line also appears in the
  `_variant_summary` / legs breakdown for the same structure (no minted numbers).
- **Put/call symmetry:** flipping `is_call` flips the tail side and the profit region
  correctly (the anti-template guard).
- A `FakeToolLLM` adherence check: given a pack with a PAYOFF line, the model's narration
  is graded only on *not contradicting* the line (kept lightweight; the real value is the
  deterministic profile, not the LLM test).

## 9. Scope — explicitly out

- No change to pricing, sizing, scoring, or scenario weighting. Projection only.
- No per-scenario / path P&L narrative — the scenario grid stays where it is; PAYOFF is the
  *terminal* profile only.
- `rko` / `european_digital_rko` remain `enabled=false`; the profile handles them only for
  the off-menu `price_structure` path.

## 10. Composition with Steps 2–3

Step 4 assumes Steps 2–3 have landed, because the payoff line consumes both:
- **Step 2** (drop `european_rko` from `structure_attributes._BARRIER`) makes the findings
  tags agree with `product_nature = expiry_only`. Without it the pack self-contradicts.
- **Step 3** (premium-sign label + surface `path_dependent`) supplies `premium_flow` and
  `product_nature`; the PAYOFF line is where they get *used*.

Recommended order stays: 1 (Opus) → 2 (ERKO tag) → 3 (premium sign + path flag) → **4
(this doc)**.

## 11. Worked examples — the exact feedback items, killed

**#8 (1x2 call, K1=5.57 / K2=5.90, net debit):**
Old: model says "short below K2." New PAYOFF line: *"pays between 5.5694 and 6.1030; best
at 5.9000; net debit; worthless below 5.5694; turns negative only above 6.1030."* The model
has nothing left to invert.

**#13 (ERKO put, target 4.21, barrier 4.0283):**
Old: "knocks out through 4.0283 (further than your target) and pays nothing" — stated as if
4.0283 were the target. New: *"pays like a put between the 4.2100 strike and the 4.0283
knock-out; nothing beyond 4.0283; barrier tested at expiry only."* Strike and barrier are
named as distinct roles, so the swap can't happen.
