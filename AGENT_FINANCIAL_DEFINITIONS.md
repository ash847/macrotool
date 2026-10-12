# Agent financial definitions — Step 1

The agent receives explicit `TradeEconomics` records on priced variants, carried
inside the standard pack. Computation lives in `analytics/trade_economics.py`;
`agentic/render.py` emits deterministic labels; `agentic/agent_flow.py` constrains
narration. These are additive fields, not changes to the legacy sizing policy.

## Meanings

- Premium: signed entry cashflow as a fraction of base-currency notional;
  positive means paid, negative means received.
- Loss budget: the input budget, not a guaranteed bound on contractual loss.
- Sizing loss proxy: the existing engine measure used by fixed-loss sizing,
  subject to existing caps. The legacy `max_loss_pct` and `max_loss_ccy` fields
  retain their values for compatibility, but the agent no longer calls them
  contractual maximum loss.
- Contractual maximum loss: separate status, value and explanation. Long vanilla,
  European digital and European RKO have non-negative terminal payoffs; their
  premium bounds loss on the stated base-currency basis. Other packages are
  explicitly unknown pending the later risk-geometry work, not inferred from
  premium or a family label.
- Target return on premium: net P&L at the specified target divided by premium
  paid. Zero-cost and net-credit trades report “Not applicable — no premium
  outlay”, while retaining target net P&L and risk information.

## Horizon and valuation

The normal pack's trade horizon is its option expiry. The additive
`evaluation_days` argument to `price_variants` supports an earlier evaluation
without changing expiry, entry price or sizing. Target valuation uses the existing
scenario-pricer full-package valuation: prevailing target-spot conversion minus
entry premium, without accruing premium, matching the scenario engine. Before
expiry it uses current rates and ATM vol with the supplied sticky-delta surface
where the existing scenario pricer supports it. These are valuation assumptions,
not predictions of future volatility. Unsupported path-state-dependent target
valuations are unavailable rather than silently returned as zero.

The legacy gross `payoff_at_target_pct` can count only the long leg on ratio
spreads. It and legacy `rr_at_target` remain untouched for compatibility but are
not narrated as the canonical target P&L or net-return ratio. The new target P&L
uses the scenario engine's full signed package. It may therefore differ from
legacy gross-payoff displays, even beyond the subtraction of premium.

## Scope

No change to pricing primitives, scenario scores, rankings, notional calculations,
Kelly optimisation, loss budgets or caps. No stop execution is assumed by agent
narration. The existing seagull sizing stress remains unchanged and is explicitly
identified as a proxy. No tail ontology, preference-parity or full sizing-audit
implementation is included in this step. No deployment is implied by these edits.

Regression coverage: `tests/test_trade_economics.py`, including fixed-loss and
Kelly numerical/ranking invariance in both directions, net-credit/zero-cost,
full-package ratio payoff, pre-expiry MtM, explicit unavailable values and currency
scaling.
