# Kelly Criterion Visualization Tool — Spec

## Objective

Build a simple interactive web tool that helps me **visually understand the Kelly criterion** and the **sensitivity of optimal bet size** to the main input parameters.

This is an **educational / intuition-building tool**, not a production trading app.

The app should let me adjust the main parameters on the right-hand side of the Kelly equation and immediately see how the recommended sizing fraction changes.

---

## Primary User Story

As a user, I want to:

- Adjust Kelly inputs with sliders
- See the resulting Kelly fraction instantly
- Understand how sensitive the output is to small changes in assumptions
- Compare full Kelly vs fractional Kelly
- Build intuition visually rather than just reading a number

---

## Product Requirements

### Core Functionality

Implement a single-page interactive app with:

#### 1. Input Sliders / Controls
- Win probability `p`
- Payout multiple / odds `b`
- Fractional Kelly multiplier
- Optional max bet cap
- Optional toggle to clamp negative Kelly to zero

#### 2. Current Output Display
- Full Kelly fraction
- Applied Kelly fraction after multiplier / cap
- Break-even probability
- Optional text explanation of whether the edge is positive or negative

#### 3. Primary Visual
- A fixed-width horizontal bankroll bar showing:
  - Bet fraction
  - Unallocated cash fraction

#### 4. Sensitivity Visual
- Chart showing **Kelly fraction vs win probability**
- Hold `b` fixed
- Show marker at current `p`

#### 5. Secondary Sensitivity Mode
- Toggle chart to **Kelly fraction vs payout multiple**
- Hold `p` fixed
- Show marker at current `b`

#### 6. Fractional Kelly Comparison
- Show:
  - Full Kelly
  - Half Kelly
  - Quarter Kelly
  - Current applied Kelly

---

## Formula Requirements

### Kelly Formula