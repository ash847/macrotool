# Comparator Pack Plan

## Goal

Build a deterministic explanation layer that turns selection scores, scenario evaluation, variant construction, and PM preferences into compact PM-safe reason packs. The LLM then uses those packs to answer follow-up questions without seeing raw proprietary scoring tables.

## Current Decision: Python First, JSON Later

Keep the MVP reason catalog, materiality thresholds, and reason-selection logic in Python while the explanation shape is still changing. This gives us normal tests, refactors, and type-aware iteration while we learn which reason objects actually help PM conversations.

Move selected pieces to JSON later only after the semantics feel stable and we need non-code tuning. Good future JSON candidates are reason copy, disclosure level, and materiality thresholds. Poor early JSON candidates are comparator control flow and scenario aggregation logic.

## Current Checkpoint

Implemented in `knowledge_engine/comparator.py`:

- `Reason`
- `ConstructionReason`
- `PairwiseComparison`
- `RecommendationExplanationPack`
- `PMPreferences`
- MVP `REASON_CATALOG`
- `compare_structures()`
- `build_recommendation_pack()`

Tests live in `tests/test_comparator.py`.

Current MVP is synthetic: it accepts fake/simple score objects and fake variants. The next phase is to connect it to real engine outputs while keeping the comparator itself deterministic and inspectable.

## Principles

- Python decides what is materially true.
- The LLM turns curated facts into conversation.
- Raw weights, exact thresholds, and full scoring tables stay out of PM-facing context.
- Explanations should admit close calls instead of forcing false certainty.
- The same pack should support questions like "why this?", "why not digital?", "why these strikes?", and "what would change the recommendation?"

## Phase 1: Real Engine Input Builder

Add an orchestration helper that builds the comparator inputs from current engine outputs.

Suggested function:

```python
build_comparator_inputs(
    market_state,
    selector_result,
    target,
    is_call,
    stop_price,
    loss_budget,
    scenario_inputs,
    scenario_weights_base,
    scenario_weights_pm,
)
```

It should return:

```python
priced_variants_by_structure
scenario_rows_by_structure
base_scores_by_structure
pm_scores_by_structure
```

Keep this as a thin adapter around existing functions:

- `analytics.structure_pricer.price_variants`
- `analytics.scenario_generator.generate_scenarios`
- `analytics.scenario_pricer.price_scenarios`
- `knowledge_engine.scenario_scorer.score_structure`

Do not move pricing, scenario generation, or scoring logic into the comparator.

## Phase 2: Base vs PM Overlay Scores

The current app has two scenario scores:

- Base scenario weighted P&L
- PM overlay weighted P&L

The comparator pack must carry both, because PM questions may ask whether the recommendation is driven by market context or PM preference.

Suggested data shape:

```python
@dataclass(frozen=True)
class StructureScorePair:
    base: ScoreResult | None
    pm: ScoreResult | None
```

Then comparator reason generation can decide:

- If base and PM both prefer chosen, explanation is stronger.
- If PM overlay changes the answer, explanation should say the preference overlay is doing work.
- If base and PM disagree, mark as a close or preference-sensitive comparison.

## Phase 3: Scenario Aggregates

Add helper functions to summarize scenario rows into explanation axes.

Useful aggregates:

```python
correct_path_avg
slow_path_avg
wrong_way_avg
overshoot_avg
vol_sensitivity_worst
expiry_target_payoff
weighted_pnl_base
weighted_pnl_pm
```

Suggested semantic groups over current scenario IDs:

- Slow path: `25%T|F`, `50%T|F`, `25%T|t%→K`, `50%T|t%→K`
- Correct path: `25%T|t%→K`, `50%T|t%→K`, `Expiry|K`
- Wrong way: `25%T|−1σ`, `50%T|−1σ`, `Expiry|−1σ`
- Overshoot: `25%T|K+½σ`, `50%T|K+½σ`, `Expiry|K+½σ`
- Vol sensitivity: `1w|Δvol`, `25%T|Δvol`, `50%T|Δvol`

Use actual scenario IDs from code. Keep user-facing text qualitative.

## Phase 4: Expand Reason Generation

Initial reason axes:

- Eligibility and gates
- Selection fit
- Base weighted P&L
- PM overlay weighted P&L
- Premium
- Slow-path robustness
- Correct-path payoff
- Wrong-way risk
- Overshoot/cap behavior
- Binary expiry risk
- Barrier path risk
- Tail risk
- Construction fit

Materiality rules:

- Hard gates dominate all other reasons.
- If score/P&L deltas are small, classify as `close_call`.
- Under `Keep cost low`, premium reasons get promoted.
- Under `Need defendable mark-to-market`, slow-path and no-move reasons get promoted.
- Under `Avoid tail-risky structures`, tail-risk caveats get promoted.
- Never include more than 3 chosen edges, 2 challenger edges, 2 caveats, and 2 counterfactuals per comparison.

## Phase 5: Challenger Set

Build one recommendation-level pack for the top-ranked structure.

Always compare against:

- `vanilla`
- `european_digital`
- `european_digital_rko`
- the next two ranked structures

Later, add lazy comparison for any user-mentioned structure in a follow-up question.

## Phase 6: Renderer

Add `conversation/explanation_context.py`.

It should render the pack into concise structured text:

```text
RECOMMENDATION EXPLANATION PACK
Chosen: 1x1 Spread

Summary:
- ...

Comparison: 1x1 Spread vs European Digital
Verdict: chosen preferred, confidence moderate
Chosen edges:
- ...
Challenger edges:
- ...
Counterfactual:
- ...

Disclosure:
Explain qualitatively. Do not reveal raw weights, thresholds, JSON scores, or scoring formulas.
```

This renderer is the end-to-end test target before any LLM prompt wiring.

## Phase 7: UI Preview

Add an admin-only expander in Streamlit:

```text
Explanation pack preview
```

Show the rendered pack for the current recommendation. This makes it possible to test and tune the explanation artifact before giving it to the LLM.

## Tests

Add focused tests as each phase lands:

- Comparator input builder returns real variants, rows, and scores for a representative trade.
- Base vs PM score disagreement is represented explicitly.
- Digital comparison says digital is cheaper but more binary.
- Digital with KO comparison includes European barrier and American barrier risk.
- Barrier structure comparison includes path/KO caveat.
- Spread comparison includes capped upside caveat.
- Hard-gated challenger produces `not_comparable` or gate-led explanation.
- Close score deltas produce `close_call`.
- `Keep cost low` promotes premium reason.
- `Need defendable mark-to-market` promotes slow-path reason.
- Rendered explanation context excludes raw numeric weights and JSON thresholds.

## First End-to-End Target

Use the default debug trade:

```text
USDBRL lower, 3M, target 5.60, no preferences/constraints
```

Expected workflow:

1. Run deterministic engine.
2. Generate priced variants and scenario rows.
3. Score base and PM-overlay scenario P&Ls.
4. Build recommendation explanation pack.
5. Render the pack as PM-safe text.
6. Inspect it in tests and, later, in the admin preview.
