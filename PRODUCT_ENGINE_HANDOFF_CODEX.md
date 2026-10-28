# Product engine handoff - Codex

## Current checkpoint

- Worktree: /Users/ash/Documents/Coding work/agentic-workflow
- Branch: agentic-workflow. Do not implement this work in main.
- Last completed commit: d593b1c (construction/policy separation).
- No push requested for this slice.
- Latest full suite: 1,325 passed, zero failures, six skipped (163.48 seconds).
- The four previous recurring failures are resolved by user-approved test cleanup:
  Auth0-only expectations, GBPUSD 1M instead of missing 1W, and removal of the
  far-target/high-carry vanilla-winner test. Production behavior unchanged.
- Test cleanup is uncommitted; no push. Historical results below are retained.

## Previous authorized slice

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
- Version 0.2.54. Checkpointed in 3fd4ea9; no push or deployment.
- Full suite: 1,308 passed, four known failures listed above, six skipped,
  146.27 seconds. Command: .venv/bin/python -m pytest tests -q
  -p no:cacheprovider --tb=short. No UI/cloud check performed.

## Resume here

1. Completed construction/policy slice (version 0.2.55): construct_package
   owns solving; _price_constructed aggregates resolved legs and entry quotes.
   product_policies.py explicitly owns legacy sizing and display conventions.
   Seagull stop valuation now uses generic value_resolved. Entry breakevens
   are reused by the compatibility envelope. Financial policy is unchanged.
2. Boundary tests prove no construction during aggregation/revaluation and
   generic four-leg valuation does not silently imply an approved sizing rule.
   Full suite: 1,321 passed, the same four known failures, six skipped in
   153.38 seconds. The final breakeven-reuse cleanup was made while that run
   was active; a fresh focused run covering construction, entry and resolved
   consumers passed all 97 tests after the cleanup (8.42 seconds).
   git diff --check passed. No cloud/UI run. No push.
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

## Next decision

The approved behaviour-preserving slice is complete. Before enabling arbitrary
packages, specify allowed ratios/legs, risk constraints and sizing policy with
the user. Existing generic valuation does not mean arbitrary package sizing is
approved. Old private entry functions remain test references; do not delete
without retaining independent regression evidence. Check git log for the local
construction/policy checkpoint commit following 3fd4ea9.
