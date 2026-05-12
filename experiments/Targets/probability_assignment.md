# FX Trade Structuring Tool — View Distribution & Target Precision Spec

## Objective

Build an interactive tool that converts an FX view into an option-structure-ready representation.

Example view:
“EURUSD to 1.05 in 3 months”

The tool should not treat this as a single-point forecast. Instead, it should help the user express:

1. Broad probability distribution at expiry  
2. Precision / conviction around the target  
3. Overshoot and tail tolerance  

This module does NOT recommend trade size.  
It is only for trade structuring.

---

## Core Inputs

User enters:

- Currency pair  
- Spot  
- Forward  
- Implied volatility (annualised)  
- Time horizon (in years)  
- Target level  

Example:

Pair: EURUSD  
Spot: 1.0350  
Forward: 1.0380  
Implied vol: 9.0%  
Tenor: 3 months (0.25 years)  
Target: 1.0500  

Direction is inferred:

- If target > forward → upside view  
- If target < forward → downside view  
- If target ≈ forward → neutral / range  

---

## Core Concept

The tool starts from a market-implied distribution:

Market distribution Q

Then allows the user to reshape it into:

User distribution P

The goal is to capture:

“Where does the user disagree with the market?”

---

# Part 1 — Market Distribution

## Horizon Volatility

Compute:

horizon_vol = vol × sqrt(T)

Example:

vol = 9%  
T = 0.25  
horizon_vol = 4.5%  

---

## Distribution Assumption

For v1:

ln(S_T / F) ~ Normal(0, horizon_vol²)

This is a simplified lognormal model used for:

- visualisation  
- bucket construction  
- probability estimation  

---

# Part 2 — Adaptive Broad Buckets

## Objective

Create 6–8 adaptive buckets that maximise information for trade structuring.

Buckets must NOT be equal-width.

They must be aligned with:

- forward  
- target  
- partial move levels  
- overshoot  
- tails  

---

## Key Principle

Buckets should answer:

1. Is the view directional or target-specific?  
2. How important is hitting the exact target?  
3. Is overshoot likely?  
4. Can upside/downside be capped?  
5. Are tails important?  

---

## Bucket Construction Algorithm

Inputs:

F = forward  
K = target  
sigma_T = horizon_vol  

---

### Step 1 — Generate sigma levels

Compute:

z = [-2, -1, -0.5, 0, 0.5, 1, 2]

For each z:

level_z = F × exp(z × sigma_T)

---

### Step 2 — Add target

Include:

K

---

### Step 3 — Add halfway level

In log space:

halfway = exp((ln(F) + ln(K)) / 2)

---

### Step 4 — Add overshoot levels

If K > F:

overshoot_1 = K × exp(0.5 × sigma_T)  
overshoot_2 = K × exp(1.0 × sigma_T)  

If K < F:

overshoot_1 = K × exp(-0.5 × sigma_T)  
overshoot_2 = K × exp(-1.0 × sigma_T)  

---

### Step 5 — Combine and sort

Collect all levels:

- sigma levels  
- target  
- halfway  
- overshoot levels  

Sort ascending.

---

### Step 6 — Merge nearby levels

If two levels are very close (e.g. < 0.25σ apart), merge them.

---

### Step 7 — Select final bucket boundaries

Choose 6–8 buckets prioritising:

- one adverse tail bucket  
- one mild adverse bucket  
- one around forward  
- one partial move  
- one target zone  
- one overshoot  
- one extreme overshoot  

---

## Example Buckets (EURUSD 3m, target 1.05)

< 0.995  
0.995 – 1.020  
1.020 – 1.038  
1.038 – 1.050  
1.050 – 1.065  
1.065 – 1.085  
> 1.085  

---

## User Interaction

For each bucket:

- user assigns probability  
- all probabilities must sum to 100%  
- enforce via:
  - auto-normalisation OR  
  - constraint-based sliders  

---

## Output of Bucket Module

Produces:

P_i = probability of each bucket  

Also compute:

- cumulative probabilities  
- probability above forward  
- probability above target  
- probability in target zone  
- tail probabilities  

---

# Part 3 — Target Precision Module

## Objective

Capture what the user means by the target.

The same target implies very different structures depending on intent.

---

## Target Interpretation Modes

User selects one:

1. Threshold  
   “I care about finishing above the target”

2. Zone  
   “I think the target is the most likely landing area”

3. Waypoint  
   “The target is directional — overshoot is fine”

---

## Precision Input

User selects precision:

- Tight  
- Medium  
- Loose  

Translate into ranges using volatility:

Tight ≈ ±0.25σ  
Medium ≈ ±0.5σ  
Loose ≈ ±1.0σ  

Convert into price space around the target.

---

## Overshoot Tolerance

User selects:

- Low (prefer capped payoff)  
- Medium  
- High (prefer uncapped payoff)  

---

# Part 4 — Derived Metrics

From bucket probabilities compute:

1. Directional probability  
   P(S_T > F)

2. Target probability  
   P(S_T > K)

3. Target zone probability  
   P(lower_target < S_T < upper_target)

4. Overshoot probability  
   P(S_T beyond overshoot level)

5. Adverse tail probability  
   P(S_T in worst bucket)

6. Difference vs market  
   Delta_i = P_i - Q_i  

---

# Part 5 — Interpretation Layer (for future use)

This module should enable downstream logic to classify:

- Broad directional view  
- Target-concentrated view  
- High overshoot view  
- Low tail-risk view  
- Convexity preference  

This will map to:

- vanilla options  
- call/put spreads  
- butterflies  
- risk reversals  
- seagulls  
- strangles  

---

# Part 6 — UI Requirements

## Layout

1. Top bar  
   Inputs: spot, forward, vol, tenor, target  

2. Main chart  
   - grey: market distribution  
   - blue: user distribution  

3. Bucket table  
   - ranges  
   - sliders for probabilities  

4. Target module  
   - interpretation toggle  
   - precision selector  
   - overshoot tolerance  

5. Diagnostics panel  
   - key probabilities  
   - summary metrics  

---

## Interaction Flow

1. User inputs market data  
2. Market distribution is displayed  
3. Buckets are generated  
4. User assigns probabilities  
5. User defines target interpretation  
6. Metrics update in real time  

---

# Part 7 — Implementation Notes

- Keep calculation logic separate from UI  
- Use log space for stability  
- Ensure probabilities always sum to 100%  
- Allow reset to market distribution  
- Avoid overfitting precision (7 buckets max)  

---

# Final Principle

The goal is not to perfectly model the distribution.

The goal is:

“To extract enough structured information about the user’s belief to choose the right payoff shape.”