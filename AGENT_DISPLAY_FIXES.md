# Agent display and narration corrections — 0.2.42

- Target (%) is the signed move from spot, not forward.
- Market State and chat Settings show the applied trade's common fixed-loss input budget. A shared footnote distinguishes this from actual package sizing amounts and contractual losses. Kelly displays N/A; fixed-loss fallback retains its budget. Staged settings take effect on Apply.
- The net-credit policy directly fixes notional at 10×W instead of budget division. No sizing formula or cap has changed.
- The supported universe contains no path-dependent products. Legacy context prose must not imply touch triggers or path-locked payoffs. Intermediate mark-to-market remains relevant for discretionary monetisation.
- Context prose is not evidence of a hard exclusion. The agent receives the actual affinity shortlist alongside regime guidance.
- Supabase chat turn 536, dated 2026-10-06, confirms the USDJPY 90d / 150 contradiction: context guidance supplied “Avoid seagulls”, but Structure Fit placed Seagull second at 50%. No seagull gate or score has been changed. Remote commentary has not been edited; the runtime scope and evidence instructions also apply to remote overrides.
- 1×1 and 1×2×1 maximum losses use strike-boundary and tail-limit extrema of intrinsic payoff / expiry spot minus entry premium, matching the existing base-currency P&L convention. Fees and premium financing are excluded. Unequal butterfly wings are evaluated, not assumed premium-only. Currency conversion can make a negative quote-currency lower-tail payoff unbounded in base currency as spot approaches zero. No bounded-loss inference is added for 1×2 or 1×1.5 spreads.
- Zero expiry payoff at target displays N/A for return on premium, with an explicit reason. Net P&L still records the premium loss. Pre-expiry mark-to-market and other negative-return cases are unchanged.
