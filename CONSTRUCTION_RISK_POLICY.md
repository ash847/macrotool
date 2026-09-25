# Construction-config no-tails policy

The single classification source is
`knowledge/defaults/structure_variants.json`. Each construction has a boolean
`can_lose_beyond_premium` declaration:

- `true`: can lose more than initial premium paid on at least one side of spot.
- `false`: cannot lose more than initial premium paid.
- Missing/non-boolean: unknown, never assumed safe.

For zero-cost or net-credit constructions, classify whether the complete trade
can incur a net loss, including any entry credit. This is a domain-authored
construction classification, not a numerical maximum-loss calculation. A `true`
flag does not imply mathematically unlimited losses. A `false` flag does not mean
no risk: the premium can still be lost.

Initial classifications:

| Construction | Can lose beyond premium |
|---|---|
| Vanilla, ordinary 1x1 debit spread | false |
| Symmetric long 1x2x1 butterfly | false |
| Long European RKO and digital, long digital RKO | false |
| 1x1.5 and 1x2 ratio spread | true |
| Seagull | true |

Declarations apply to the actual configured construction. Reassess the declaration
when changing ratios, leg direction, strike ordering or funding construction.
Do not inherit it automatically from a family name. Existing disabled-product
settings remain unchanged.

The `Avoid tail-risky structures` preference allows only explicit `false`.
Family selection checks that at least one configured construction passes; entry
pricing filters individual constructions before scenario ranking. Both the agent
comparator and Trade View evaluation/table pass the same filter, as does the
agent's no-target fallback. Other preference scores/gates remain unchanged.

Custom requests get an approved flag only when all their construction parameters
exactly match a configured construction, ignoring the display label. Otherwise
they remain unknown. Without the no-tails preference they can still be priced and
are labelled unknown; with it, the tool returns an explicit exclusion reason.
The agent cannot declare a custom construction safe or override the preference.

No payoff/risk math, market data, premium calculation or sizing policy is changed.
Unknown contractual maximum-loss amounts remain separate from this flag. The
existing linear reference in Trade View remains a benchmark, not an eligible
option recommendation.

Restart the app and rerun the trade after changing construction configuration,
so existing session packs are rebuilt. Automatic configuration revision tracking
belongs to the later pack-parity step.

Tests: `tests/test_construction_risk_policy.py` covers declared/unknown flags,
mixed eligible/ineligible constructions within a family, custom terms, fallback,
comparator and Trade View parity, both directions, and unchanged allowed pricing.
