# MacroTool — Scoring & Evaluation Framework

A guide for testers with trading knowledge. This document explains how the tool goes from a plain-English trade idea to a shortlist of recommended option structures, and how those structures are then evaluated across market scenarios.

---

## Market Data

The tool runs on a representative synthetic snapshot for seven pairs: USDBRL, USDTRY, EURPLN, EURUSD, USDCNH, USDMXN, USDJPY.

For each pair the snapshot contains:

- **Spot** rate
- **Outright forwards** at standard tenors (1W through 1Y)
- **Implied vol surface** — ATM, 25Δ and 10Δ calls and puts at each tenor, capturing the skew and butterfly structure that characterises each pair (BRL/TRY with strong topside skew and high carry, JPY with mild downside skew, EURUSD near-symmetric)
- **Discount curves** — used to derive the domestic and foreign interest rates via CIP from the forward outright

The data is fixed, not live-streamed. It is calibrated to be broadly representative of each pair's character at a typical point in the cycle — enough to demonstrate how the scoring responds to different carry and vol regimes across the pair universe.

---

## From Trade Idea to Market State

The PM enters a view in plain English — pair, direction, magnitude, and horizon. The system extracts this and derives the following from the snapshot:

**Normalised carry (c)**
The ratio of the forward premium to the one-standard-deviation move. Concretely: how much of the vol budget is consumed by the carry. A high positive c means the forward is well above spot — carry is substantial and directional.

**Carry regime**
Bucketed from c into three levels:
- *Noisy (0)* — carry is small relative to vol; forward is close to spot; carry signal is uninformative
- *Potential (1)* — carry is discernible; spread structures start to make sense
- *High (2)* — carry is dominant; structures that monetise the drift are clearly favoured

**Target z-score**
How far the target is from the forward, expressed in standard deviation units. A target at +1σ is an extended move; a target at +0.3σ is modest. This drives how much optionality is needed and whether capped structures are viable.

**ATM/FS ratio**
The payout ratio of the spread that exactly captures the forward drift — long at-spot, short at-forward, in the carry direction. A high ratio means carry is cheap to monetise via a spread. A low ratio means the market is already pricing carry tightly and there is less edge in a spread structure.

**Carry alignment**
Whether the trade view is with or against the carry. This matters because with-carry positions benefit from both the view being right and time passing (theta works in your favour via the drift). Counter-carry positions require a stronger view to overcome the bleed.

---

## Affinity Scoring — How Structures are Shortlisted

There is no machine learning here. Each candidate structure is scored by a rule engine across four independent dimensions. The scores sum to a total affinity score; the top three eligible structures become the shortlist.

### Step 1 — Gates (hard filters)

Before scoring, structures are checked for basic eligibility:

- Structures that require a target (spreads, seagull, 1×2) are excluded if no target is given, or if the target is too close to the forward (less than 0.5σ away — not enough room to structure a spread)
- Risk reversal is permanently gated — the tool does not recommend outright risk reversals at this stage

A structure that fails a gate is excluded regardless of how well it would score.

### Step 2 — Four scoring dimensions

**1. Target distance**
How far the target is from the forward in σ terms. Vanilla scores consistently at all distances. Spread structures score well at moderate distances (there is room to capture the move without capping too early) and poorly at very far targets (the short leg becomes deep OTM and the structure becomes expensive relative to payoff). The 1×2 spread is additionally gated out at far targets where the uncapped short exposure becomes material.

**2. Carry regime**
How strong the carry signal is. High carry strongly favours spread structures — you can sell the far leg cheaply and reduce cost while still capturing the drift. Low/noisy carry removes this advantage; vanilla dominates because there is no reliable drift to monetise.

**3. ATM/FS ratio**
How efficiently carry can be harvested via a spread at this moment. A high ratio means the market is offering good value on the spread; the scoring rewards structures that benefit from this. When the ratio is low, carry-capturing spreads become less attractive.

**4. Carry alignment**
The interaction of carry direction and carry magnitude. A with-carry trade in a high-carry regime benefits from three things simultaneously: the view, the drift, and time passing. This compounds the advantage of spread structures. A counter-carry view in the same regime faces headwinds on all three axes, pushing the scoring back toward simpler structures where the cost of being wrong is more contained.

Each dimension contributes a positive, negative, or zero score. The total is the structure's affinity score. The top three become the shortlist, ranked in order.

---

## Structure Evaluation — Scenario MtM

The affinity score tells you which structure suits the current market environment. The evaluation layer asks a different question: **how does the chosen structure actually behave as the trade evolves?**

### What stays fixed

Once the trade is on, the following are held constant across all scenarios:

- Strikes and barriers (set at entry, never re-solved)
- Interest rates and discount factors
- Skew (currently recorded but not yet shocked — placeholder for future)

### What moves

- **The forward** — each scenario specifies a rule for where the forward lands (e.g. halfway to target, full target, 1σ adverse). Spot is then back-derived from the new forward using the original rates.
- **Time** — each scenario sits at 25%, 50%, 75% or 100% of the tenor elapsed
- **Vol** — flat (unchanged), down 1%, or up 1%

### What is reported

For each scenario: the MtM option value as a percentage of entry spot, and the P&L relative to the premium paid at entry.

---

## Scenario Families

The 22 scenarios are grouped into eight families. Each family is asking a distinct question about the trade.

---

### Correct Path
*Is the structure accruing value at a reasonable pace as the view plays out?*

| Scenario | Time elapsed | Forward | Vol |
|---|---|---|---|
| 25% of the way to target | 25% | Interpolated 25% toward target | Flat |
| 50% of the way to target | 50% | Interpolated 50% toward target | Flat |
| 75% of the way to target | 75% | Interpolated 75% toward target | Flat |
| Target reached at expiry | 100% | At target | Flat |

---

### Early Target
*If the target is hit well before expiry, how much residual time value is left to monetise?*

| Scenario | Time elapsed | Forward | Vol |
|---|---|---|---|
| Target hit at 25% of tenor | 25% | At target | Flat |
| Target hit at 50% of tenor | 50% | At target | Flat |
| Target hit at 75% of tenor | 75% | At target | Flat |

---

### No Move
*How much does theta bleed cost if nothing happens?*

| Scenario | Time elapsed | Forward | Vol |
|---|---|---|---|
| No move at 25% of tenor | 25% | At original forward | Flat |
| No move at 50% of tenor | 50% | At original forward | Flat |
| No move at 75% of tenor | 75% | At original forward | Flat |

---

### Wrong Way
*How resilient is the structure to a move against the view?*

| Scenario | Time elapsed | Forward | Vol |
|---|---|---|---|
| Small adverse move | 50% | 0.5σ against view | Flat |
| Large adverse move | 50% | 1σ against view | Flat |

---

### Overshoot
*For capped structures (spreads), does the short leg cap the upside materially if the move exceeds the target?*

| Scenario | Time elapsed | Forward | Vol |
|---|---|---|---|
| Overshoot mid-life | 50% | 0.5σ beyond target | Flat |
| Overshoot at expiry | 100% | 0.5σ beyond target | Flat |

---

### Vol Sensitivity
*How exposed is the structure to a change in implied vol level?*

| Scenario | Time elapsed | Forward | Vol |
|---|---|---|---|
| Vol down, halfway to target | 50% | 50% toward target | −1% |
| Vol up, halfway to target | 50% | 50% toward target | +1% |
| Vol down, at target | 50% | At target | −1% |
| Vol up, at target | 50% | At target | +1% |

---

### Skew Sensitivity
*How exposed is the structure to a change in the skew of the vol surface?*

| Scenario | Time elapsed | Forward | Vol |
|---|---|---|---|
| Skew flatter | 50% | At target | Flat, skew ×0.75 |
| Skew steeper | 50% | At target | Flat, skew ×1.25 |

> Skew is recorded in the scenario but not yet priced — the current pricer uses flat vol for all strikes. These scenarios are placeholders for when smile-aware pricing is added.

---

### Expiry
*What are the terminal payoffs at the extremes — no move and a large miss?*

| Scenario | Time elapsed | Forward | Vol |
|---|---|---|---|
| No move at expiry | 100% | At original forward | Flat |
| Large adverse move at expiry | 100% | 1σ against view | Flat |

---

## What's Next

The scenario output currently shows raw MtM and P&L per scenario. The next step is to introduce a **scenario scoring rubric**: weight each family by its importance to the PM's objective (e.g. upside capture vs. robustness vs. cost of carry), compute a weighted score per structure, and feed that back into the overall ranking alongside the affinity score.
