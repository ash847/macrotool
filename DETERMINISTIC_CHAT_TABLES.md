# Deterministic follow-up tables

`inspect_recommendations.display` selects Python-rendered output:

- `none` (default): retrieve facts for prose, without inserting a table.
- `trade_details`: existing financial comparison table.
- `contributors` / `detractors`: one side of the stored scenario drivers.
- `drivers`: contributors and detractors.
- `both`: trade details plus both driver tables.
- `dashboard`: one combined table, with trades as columns by default.

Dashboard presentation controls (no cell values are accepted):

- `layout`: `trades_as_columns` or `trades_as_rows`.
- `fields`: ordered, unique field IDs from the tool schema. Omit for the standard
  summary: legs, notional, premium, target P&L, target return on premium, loss
  budget, sizing loss proxy, additional-loss flag, directional tails, top contributor.
  Optional fields include the retained risk note and top detractor.
- `driver_count`: 1–3 retained drivers per side per trade; defaults to 1.

These controls require `display=dashboard`; incompatible or unknown options are
rejected rather than silently ignored. Field order is preserved; trades remain
in engine rank order. A dashboard request replaces pending separate tables for
that turn, producing one table. The latest explicit dashboard request wins.
All original display modes remain supported. No user cell values, formulas,
weights or private score fields are accepted.

`agentic/dashboard.py` maps stored engine facts to cells, then transposes the same
cell matrix if requested. Currency scaling of canonical per-notional target P&L
and sizing proxy is deterministic Python, matching the existing detail context.
No premiums, Greeks, scenarios, rankings or sizing decisions are recomputed.
Missing facts stay unavailable; net-credit/zero-premium return ratios stay N/A;
linear's missing option economics are not fabricated. Loss budget is explicitly
a reference, never a contractual bound. Driver percentages retain the full
absolute-contribution denominator, regardless of the number shown.

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
for drivers should run the pack then inspect it with `display=dashboard` for one
combined table or `display=both` for separate tables. Ordinary prose
follow-ups are unchanged. Custom-pricing rendering is outside this change.

Version 0.2.27. Historical saved replies are not rewritten. UI/live model acceptance
still needs verification; deterministic tests use a scripted model, not paid API calls.
