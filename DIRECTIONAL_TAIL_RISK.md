# Directional tail constraints (0.2.26)

The chat now accepts lower-spot, higher-spot, both-sided and view-relative tail
exclusions. This is a hard construction filter, not a narration-only preference.

`knowledge/defaults/directional_tail_risk.json` declares lower/higher exposure for
each supported call/put construction family. Rules apply only to exact catalog
terms (or catalog labels for retained priced variants). Unmatched custom terms
remain unknown. Missing/non-boolean declarations also remain unknown. A zero or
long financing wing is not classified as an uncovered short seagull wing.

Tail means unprotected terminal losses beyond premium as spot moves further in
that direction. It is not a claim of mathematically infinite loss, a modelled
stop guarantee, or all possible mark-to-market risks. Long linear exposure has a
lower-spot tail, short linear exposure a higher-spot tail, despite modelled caps.

The separate `tail_constraint` preference accepts `none`, `lower_spot`,
`higher_spot`, `both`, `against_view`, `with_view`. Python resolves relative
constraints against the engine's base-higher/base-lower view. Absolute constraints
remain absolute after a view change. The older "Avoid tail-risky structures"
setting remains stricter: both sides must be safe. Clearing the directional
preference does not clear that separate setting.

`set_tail_constraint` updates the active view without inventing new target/tenor
inputs. `run_standard_pack` can set it alongside a new view; omission preserves
the preference. It is included in the cache key, shortlist reference, saved idea
preferences and chat settings. Changing sizing settings preserves it. No database
migration is needed because existing settings/preferences JSON fields carry it.

Filtering happens before assigning displayed ranks and computing deciding-axis
commentary. Eligible variants retain unchanged prices, sizing and P&L scores.
The no-target fallback is also filtered. If all variants fail, no fallback can
reintroduce them. Unknown on an excluded side fails closed. Exclusion reasons are
retained per variant and supplied to the agent. Custom trades can still be priced
for inspection but a conflicting or unknown tail is explicitly flagged; their
construction is never silently changed.

Python-generated tables and agent context carry both tail labels and the active
constraint. For USDCNH to 7.20, excluding lower-spot tails excludes short-put
seagulls and long linear exposure. Call ratio spreads pass that tail filter,
but still need to pass the existing regime eligibility rules (the current 7.20
snapshot does not shortlist them). No tail preference bypasses those rules.
Scenario weights, pricing and sizing formulas are unchanged.

Tests cover catalog directions, custom unknowns, both-sided filtering, cache and
resume persistence, view reversals, fallback exhaustion and the USDCNH example.
Live natural-language extraction still needs UI acceptance testing.
