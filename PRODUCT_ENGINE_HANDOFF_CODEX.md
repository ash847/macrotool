# Product engine handoff - Codex

## Current checkpoint

- Worktree: /Users/ash/Documents/Coding work/agentic-workflow
- Branch: agentic-workflow. Do not implement this work in main.
- Last completed commit: df7a9c9 (unified vanilla/two-leg entry pricing).
- No push requested for this slice.
- Previous full suite: 1,282 passed, 4 known failures, 6 skipped.
- Known failures: two Google login button tests in test_roles_visibility;
  missing GBPUSD 1W in test_snapshot_overrides; ranking expectation in
  test_structure_scorer::TestRanking::test_vanilla_wins_far_target_high_carry.

## Current authorized slice

Migrate butterflies and seagulls onto product_pricer.price and retain the
resolved vanilla-leg contract for scenario repricing and Kelly payoffs.
Keep existing ranking, sizing and financial conventions. Freeze the seagull's
solved wing quantity, NOT its construction placeholder. Preserve butterfly
target eligibility and nonpositive-wing filtering. Pass the stop into seagull
entry pricing so its loss-budget calculation is unchanged.

## Completed slice - 2026-10-28

- Butterfly and seagull entry dispatch now uses product_pricer.price.
- freeze_contract accepts actual priced-leg quantities and rejects a seagull
  without explicitly supplied solved quantities.
- Seagull stop price is forwarded; butterfly target gates and invalid-wing
  filtering are retained. InvalidButterflyWing is a narrowly caught error.
- Consumer tests cover both new families under spot/vol/time shocks, expiry
  payoffs, fixed contract identity and no reconstruction.
- Independent legacy parity includes both new families, including stop budgets.
- Version 0.2.54. Changes are currently uncommitted; no push or deployment.
- Full suite: 1,308 passed, four known failures listed above, six skipped,
  146.27 seconds. Command: .venv/bin/python -m pytest tests -q
  -p no:cacheprovider --tb=short. No UI/cloud check performed.

## Resume here

1. Review/checkpoint this tested slice before further changes.
2. Separate family construction/solving from generic resolved-package metrics;
   some payoff/risk policy still dispatches by family. Do not change policy
   silently (notably seagull stop budget and ratio-spread tail-risk treatment).
3. Flexible same-expiry ratios are a later product feature requiring validation
   and user specification; this slice does not expose arbitrary packages.
4. Specialized binary/barrier migration requires explicit contract semantics;
   preserve the negative-smile-package rejection guard. Multiple expiries need
   per-leg expiry support, not just extra legs.

## Scope boundaries

- Generic leg lists have no hardcoded maximum; current contract has one expiry.
- Arbitrary ratios/user construction require later validation and product design.
- Barriers/digitals need specialized semantics, not vanilla-leg flattening.
- Old private entry functions remain independent regression references.
- Broader roadmap: BETA_IMPLEMENTATION_TODO.md.
