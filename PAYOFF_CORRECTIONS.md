# Payoff consistency — v0.2.44

Scope: fix target payoff on 1×1.5/1×2 spreads, retain solved seagull wing precision, and align expiry breakevens with the base-currency P&L convention. No sizing-policy, eligibility or target-selection changes.

- Both variant and product pricers subtract all short-leg intrinsic value at the target. Legacy gross payoff/premium R/R follows the corrected gross payoff. Agent net target P&L remains net of premium and uses the full package.
- Both pricers retain unrounded seagull wing ratios; actual priced legs, scenario valuations and Kelly payoff bridges use the same ratio. Rounding belongs only in display. This can change scenario rankings and Kelly allocations; it is not merely a cosmetic fix.
- Shared breakeven calculation solves intrinsic(spot)/spot minus entry premium = 0 across strike intervals. No premium financing/accrual is introduced. It returns all isolated positive finite crossings and distinguishes zero-P&L regions. Digital and expiry-barrier jumps are not invented zero crossings.
- `breakevens` carries the full list; the compatibility `breakeven` scalar is populated only for one isolated crossing without zero-P&L regions. Trade View and agent context display the complete result. Unsupported path-dependent families return unavailable rather than a fabricated strike-based breakeven.
- A capped quote-currency payoff may have two base-currency crossings because its value in base currency falls as expiry spot rises. This follows the existing reporting convention, not a new pricing assumption.

Still separate: legacy `max_loss_pct` remains a sizing proxy for open-tail structures. Its semantics and remaining UI labels need a distinct review before altering sizing policy. Options with zero payout at the requested target remain eligible, as requested.
