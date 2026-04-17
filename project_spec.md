# EM FX Trade Structuring & Sizing Tool — Project Spec

---

## What this is

A conversational tool for macro fund PMs that takes a trade view in plain English and returns recommended structures with sizing, entry, and exit parameters — or critiques a structure the PM has already identified. The tool covers spot, forwards, and vanilla options across a defined set of EM FX currencies, with first generation exotic structures available as on-demand comparison instruments.

The core value proposition is not a pricer. Pricers exist. The value is the opinionated layer above the pricing: which structure expresses this view most efficiently given the budget and conviction level, how large the position should be given the PM's edge and the instrument's vol regime, where to place the stop, and how to scale in and out. All of it explained in specific terms with the reasoning made explicit.

---

## The problem it solves

Macro PMs typically have the view and the qualitative risk framework. The gap is in the systematic application of structure selection, sizing discipline, and entry/exit parametrisation at the moment of trade construction. Bloomberg has the data and the pricing but requires significant expertise to navigate. Generic AI tools have the language but not the domain judgment. This tool combines a conversational interface with encoded domain expertise to fill that gap.

The specific pain points validated with PM contacts:
- Knowing what structures are available for a given view and budget, and what each one is actually optimised for
- Understanding what a position's profile looks like across a range of spot and vol outcomes — not just at expiry, but through the life of the trade
- Applying sizing disciplines (Kelly, vol adjustment, scaling) systematically rather than intuitively
- Getting an honest, specific critique of a structure they have already identified

---

## What it does

**Input**

The PM provides four things in plain English:

- A view — direction, magnitude, time horizon
- A budget or risk limit — either premium budget in dollar terms or maximum loss
- Conviction on timing — high, medium, or low
- Any specific hedge or structure preference if they have one

**Step 1 — View validation**

Before surfacing structures, the tool assesses whether the stated move is in the price. What does the vol surface imply about consensus positioning. Whether the view is differentiated or crowded, and how that affects structure choice. This step changes the recommendation — a crowded view in a high-skew currency points toward a different expression than an out-of-consensus view in a flat vol environment.

**Step 2 — Structure recommendation**

Two or three structures that express the view given the budget, with:

- Conventions resolved explicitly — the PM sees the NDF fixing, settlement basis, premium currency, and cut for their specific instrument
- A scenario matrix showing P&L in dollar terms across a grid of spot and vol outcomes at multiple time horizons — not a payout diagram at expiry, but a view of how the position behaves as it evolves
- A specific explanation of what each structure is optimised for, referencing the PM's stated view parameters — not a generic description

**Step 3 — Sizing**

- Kelly-adjusted notional, with half-Kelly as the default and the reasoning stated explicitly in terms of the specific instrument and vol regime
- Vol-regime adjustment — if the currency's vol is elevated relative to its own history, the sizing adjusts down with the reason explained
- Scaling schedule if timing conviction is low — how to enter in tranches given the horizon and conviction level rather than a single entry

**Step 4 — Entry and exit parametrisation**

- Stop level derived from the instrument's vol — the level at which the market has indicated the view is wrong, not an arbitrary round number
- Catalyst-based time exit — the PM is asked to state the catalyst explicitly; this becomes a time-based exit condition alongside the vol-derived stop
- Take-profit scaling rules — at what P&L levels to reduce the position and by how much, given the remaining time horizon

**Critique mode**

The PM supplies their own structure. The tool evaluates it against their stated view parameters honestly and specifically:

- Expected value of the PM's structure versus the recommended alternatives, given the stated view
- Where it underperforms — not generically, but in the specific scenarios that matter for this view
- Execution considerations the payout diagram does not show — liquidity in the strikes chosen, gamma management requirements, whether the hedge is hedging what the PM thinks it is
- The same sizing and entry/exit logic applied to the PM's structure once evaluated

---

## Exotic comparison layer

The tool does not recommend exotic structures proactively. When a PM asks to compare a recommended vanilla structure to an exotic, the tool engages with that comparison specifically.

**Structures supported for comparison**

- Reverse knock-out (RKO) — barrier call or put with knock-out at a specified spot level
- European digital — fixed cash payout if spot is above or below the strike at expiry
- European digital with RKO — fixed cash payout contingent on spot not having touched the barrier during the life

**What the comparison shows**

Not an explanation of how the structure works. Three specific outputs:

1. **EV comparison** — given the PM's stated view on direction and magnitude, the expected value of the exotic versus the recommended vanilla, probability-weighted using the vol surface
2. **Price difference made tangible** — the premium saving in dollar terms, and specifically what the PM is selling to achieve it. "You are saving $X in premium. What you are giving up is the payout in the scenario where spot trades through the barrier before reaching your target — the vol surface implies a Y% probability of that path."
3. **Decay profile** — how the position's value evolves across spot levels and through time, with particular attention to the dynamics specific to that structure in that currency. For an RKO in TRY this means the behaviour as spot approaches the knock-out level. For a digital this means the binary gamma profile near the strike as expiry approaches.

**Pricing methodology**

Closed form Black-Scholes solutions for all three structures. Sufficient for comparison and decay profile purposes. Not offered as tradeable prices.

---

## Currency scope — Phase 1

Three currencies selected to cover the meaningful variation in EM FX:

| Currency | Type | Carry | Vol | Skew | Primary cross |
|---|---|---|---|---|---|
| BRL | NDF | High | High | High | USDBRL |
| TRY | NDF | Very high | Very high | One-directional | USDTRY |
| PLN | Deliverable | Low | Low | Symmetric | EURPLN |

Together these cover: NDF versus deliverable, high versus low carry, high versus low vol, asymmetric versus symmetric skew, USD-quoted versus EUR-quoted primary cross.

---

## Asset class scope

**Phase 1**
- FX spot — offshore markets only
- FX forwards — offshore NDF for BRL and TRY, deliverable for PLN
- Vanilla FX options — calls, puts, risk reversals, call spreads, put spreads
- First generation exotics — RKO, European digital, European digital with RKO, available as comparison instruments only

**Future phases**
- Additional currencies — ZAR and MXN are natural next additions, adding commodity linkage and high-liquidity US macro beta respectively
- Additional products — IRS, CCS, and sovereign CDS to be scoped once Phase 1 is validated

---

## Design principles

All architecture and data model decisions must support expansion of both currency coverage and product coverage without structural rework. Specifically:

- The conventions layer must be currency-agnostic in structure — adding a new currency means populating a defined schema, not modifying core logic
- The structure selection logic must be product-agnostic in structure — adding a new product class means extending the decision rules, not rebuilding them
- The pricing engine must be modular — adding a new instrument or exotic type means adding a pricer module, not modifying the scenario matrix or output layer
- The knowledge layer must be separable from the reasoning layer — domain expertise is encoded as explicit rules and parameters, not embedded in prompts that are hard to audit or update

---

## Data approach

**POC**
Synthetic snapshot. Hardcoded spot rates, forward points, and vol surfaces for the three currencies. Updated manually. Sufficient to test the workflow and conversation with PM contacts.

**Production**
Bloomberg integration. OVML for FX options pricing and vol surfaces, FXFA for spot and forwards. The tool sits on top of the PM's existing Bloomberg infrastructure, using their internal data rather than a third party feed. This eliminates the data legitimacy question — the PM is working from the same prices they would trade on.

---

## The knowledge layer

The tool's defensibility sits in five encoded knowledge components:

**1. Conventions**
Complete instrument and settlement conventions for each currency and asset class in scope. NDF mechanics, fixing details and fixing risk, holiday calendar effects on forward date calculation, deliverable versus NDF distinctions, premium currency and delta conventions for options.

**2. Structure selection logic**
Explicit decision rules mapping view parameters to appropriate structures. Direction conviction, timing conviction, budget, current skew environment, and vol regime together determine which structures make the shortlist and in what order.

**3. Statistical disciplines**
Specific implementation of Kelly criterion, vol-regime adjustment to notional, scaling schedules for low-timing-conviction entries, vol-derived stop placement, and take-profit scaling rules — as applied to these specific instruments, not as general theory.

**4. Critique framework**
The evaluation dimensions applied when assessing a PM-supplied structure. EV against view, scenario underperformance, execution considerations, gamma and path dependency implications, hedge effectiveness.

**5. Exotic comparison judgments**
Currency-specific context for exotic comparisons. Which exotic structures are commonly used and liquid in which currencies. Where the skew environment makes a barrier structure attractive or expensive. What the typical path dynamics are in each currency that affect barrier probabilities.

---

## Build sequencing

**Step 1 — Knowledge extraction** *(~2 weeks, no code)*
Conventions document for BRL, TRY, PLN across spot, forwards, and options. Structure selection decision tree. Statistical disciplines parameterisation — Kelly inputs, vol adjustment methodology, stop and take-profit rules. Exotic comparison judgment layer.

**Step 2 — POC build** *(~2 weeks)*
Synthetic data snapshot. Black-Scholes pricer with correct NDF conventions. Closed form pricers for RKO, European digital, European digital with RKO. Kelly and sizing calculations. LLM conversational layer built on the knowledge documents. Basic conversational interface.

**Step 3 — PM testing** *(~2 weeks)*
Three PM contacts, live trade ideas, both recommend and critique modes tested. Evaluation criteria: does the conversation have the right elements, does the scenario matrix format work, does the sizing output change behaviour, does the exotic comparison output add clarity rather than confusion.

**Step 4 — Iteration**
Refine knowledge layer based on PM feedback. Extend to second and third currency if not already covered. Tighten structure selection logic based on what PMs actually asked for versus what was recommended.

**Step 5 — Bloomberg integration**
Replace synthetic data with live Bloomberg feed. Move from POC to usable tool.

---

## Success criteria for POC

Two out of three PM contacts report that the tool changed something specific about how they structured, sized, or parametrised a trade. Not "this is interesting" — something changed.

---

## Open questions

- **Outcomes logging** — whether and how to record trade inputs and subsequent market outcomes to build a feedback dataset over time. This is the compounding data asset that makes the tool more defensible at scale.
- **Vol surface data for non-Bloomberg users** — Bloomberg solves the data problem for funds with Bloomberg access. A separate solution is needed for funds without it. Not a Phase 1 problem but worth tracking.
- **Expanded currency list** — ZAR adds commodity linkage. MXN adds high-liquidity US macro beta and a currency where the political risk / vol surface relationship is distinctive. Both are natural Phase 2 additions.
