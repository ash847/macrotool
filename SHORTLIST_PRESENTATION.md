# Trade View-style initial chat presentation (0.2.20)

The initial answer now contains Market state, 2–3 LLM-written sentences explaining
the regime and selection, Shortlisted structures (top three eligible primary
families with fit percentages), and Top structures (top five individual variants).
Top-structure columns match compact Trade View: rank, structure, variant, strikes,
notional, signed premium percentage and Kelly risk where applicable. Detailed
variant tables are not included in the initial answer.

All comparator-evaluated variants are retained, rather than selecting one per
family. The same Trade View linear benchmark is included, at reference notional W,
with explicitly modelled capped scenario losses. It is not a Kelly-sized option,
a guaranteed executable stop, or a contractually protected trade. No-tail rules
continue to apply to option constructions; the linear comparison remains labelled
as a benchmark, matching Trade View. No new risk classification is inferred for it.

Fit percentages are approved public data, not probabilities. Numeric scenario
scores and weights remain private. The detailed financial definitions, sizing
audits and per-variant contributors/detractors remain available for follow-ups.
Family lookup returns every retained matching variant with its exact rank;
comparisons reference Top structures ranks, not family-shortlist ranks.

No-target/fallback packs are not presented as scenario-ranked recommendations.
Parity tests compare all retained variants with Trade View's evaluation under
fixed-loss and Kelly sizing in both directions. Pricing and scoring formulas
remain unchanged; the displayed candidate selection intentionally changes.

The notes below describe the earlier 0.2.19 presentation, superseded for initial
answers. Its detailed financial comparison table remains available on request.

## Previous presentation

Python renders the default top-five table directly from the retained engine
recommendations. Columns: rank, structure/key terms, sized notional, premium,
net P&L at target, target return on premium, and additional loss beyond premium.
Amounts use explicit base-currency codes; the P&L cell states its horizon.
Unknown values are unavailable/unknown, not replaced with zero. No internal
numeric ranking scores or weights are exposed.

The table preserves engine order and selects the best retained variant per family,
not a new global variant ranking. The agent is instructed to add one short
top-pick paragraph (maximum 120 words), not five essays. Python inserts the table
even if the model omits its optional `[[SHORTLIST]]` marker and removes duplicate
model-authored Markdown tables from that reply. Narration length is a prompt
constraint, not a hard truncation that might remove important qualifications.

The displayed final answer is also stored in model history and the conversation
transcript. Custom-pricing answers and ordinary follow-ups do not automatically
repeat the top-five table.

`inspect_recommendations` retrieves one or several already-priced ranks for
explanation, risk, sizing or comparison. It can also look up a named family,
including retained recommendations below fifth place. It does not reprice or
rerun the engine. Rank references carry a lightweight shortlist fingerprint;
references to an older/different current list are rejected rather than silently
mapped to different trades. This is not the deferred shared-pack/live-data refactor.

If a family was never retained, the tool distinguishes “not shortlisted” from
“shortlisted but no priced recommendation retained”. Detailed gate/pricing
failure reasons are not historically retained in the pack; the tool explicitly
states that limitation rather than inventing a reason or consulting changed config.

The detailed engine context and sizing audits remain available internally.
This step changes public presentation and lookup, not pricing, sizing, eligibility,
ranking or scenario weights. Provider-free tests validate table fidelity, lookup
without repricing, stale-reference rejection, missing values and saved history;
live-model adherence to the prose-length instruction still requires a user trial.
