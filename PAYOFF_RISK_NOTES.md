# Variant-specific payoff / risk notes

The optional dashboard `risk_note` field is labelled **Payoff / risk**. Standard
packs retain its deterministic text in `RecommendedStructure.major_risk` for
compatibility; it no longer comes from generic structure-profile risk prose.
The same generator supplies agent-facing PAYOFF lines, including custom trades.
This does not add a risk column to the upfront Top structures list.

`knowledge_engine/payoff_risk.py` reads actual signed option legs and priced terms.
Exposure starts at the outer strike, not the PM's target. Tail net-loss crossings
include entry premium and use the scenario engine's base-currency convention:

`net P&L per unit = quote-currency intrinsic payoff / expiry spot - entry premium`

Within a tail, quote intrinsic is `slope * spot + intercept`; its net-P&L zero is
`-intercept / (slope - premium)`. Only positive roots in that tail region are
reported. No grid search, target substitution or LLM arithmetic is used. Entry
premium is not accrued, consistent with current scenario economics.

These are expiry facts, not pre-expiry MtM thresholds. Option-payoff caps and peaks
describe quote-currency intrinsic geometry, not a promised peak in base-currency
net P&L. Unequal butterfly wings are not assumed to have zero outer payoff.
Barrier notes distinguish path-touch and expiry-only conditions. Missing terms
produce an unavailable note rather than a generic family-level assertion.

Pricing, sizing, scenario ranking and tail-filter policies are unchanged. Legacy
JSON profile prose remains available to other consumers, but no longer supplies
the recommendation risk notes. The existing legacy payoff-profile module is not
used by the agent renderer for these notes.
