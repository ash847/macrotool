# Deterministic follow-up tables

`inspect_recommendations.display` selects Python-rendered output:

- `none` (default): retrieve facts for prose, without inserting a table.
- `trade_details`: existing financial comparison table.
- `contributors` / `detractors`: one side of the stored scenario drivers.
- `drivers`: contributors and detractors.
- `both`: trade details plus both driver tables.

The model selects only display intent and ranks (or family), never numerical cells.
The tool schema requires an explicit display choice; omitted values from legacy
Python callers default to `none` for compatibility.
The tool validates the shortlist reference and selection before a display is queued.
Missing ranks select the current top five; a family selects its retained variants.
Multiple successful requests within a turn merge by table kind and engine rank.
Changing the view or pricing a custom trade clears pending inspection tables.
New turns start with no pending tables. Rejected requests cannot queue new rows.

`agentic/shortlist.py` renders narrow per-variant driver tables directly from stored
`cell_drivers`, using the full absolute-contribution denominator retained in the pack.
Unavailable denominators show N/A; missing driver evidence is distinct from an empty
positive/negative list. No pricing, sizing, weights, ranks or engine outputs change.
The displayed text is saved verbatim in conversation history. Model-authored tables
are still removed in these controlled table responses; requested driver tables now
replace them instead of an unrelated trade-summary table.

The initial market-state/shortlist/ranking layout is unchanged. A new-view request
for drivers should run the pack then inspect it with `display=both`. Ordinary prose
follow-ups are unchanged. Custom-pricing rendering is outside this change.

Version 0.2.25. Historical saved replies are not rewritten. UI/live model acceptance
still needs verification; deterministic tests use a scripted model, not paid API calls.
