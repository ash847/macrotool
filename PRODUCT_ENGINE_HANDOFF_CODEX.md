# Product engine handoff - Codex

## Current checkpoint

- Worktree: /Users/ash/Documents/Coding work/agentic-workflow
- Branch: agentic-workflow. Do not implement this work in main.
- Pre-integration checkpoint: d561a32 (flexible ratios, clean tree).
- No push requested for this slice.
- Integrated source full suite: 1,416 passed, zero failures, seven skipped
  (160.55 seconds). Eight additional combined-feature tests then passed.
- MCP optional suite: 13 passed using /Users/ash/macrotool-mcp/.venv/bin/python,
  with fake model and real stdio transport. No API calls or live deployment.
- The four previous recurring failures are resolved by user-approved test cleanup:
  Auth0-only expectations, GBPUSD 1M instead of missing 1W, and removal of the
  far-target/high-carry vanilla-winner test. Production behavior unchanged.
- Test cleanup is committed; no push. Historical results below are retained.

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

## Main integration - 2026-10-28

Merged main/origin/main bf12508 into checkpoint d561a32 in an isolated managed
worktree before updating agentic-workflow. No main worktree edits or push.
Version 0.2.57. Eight textual conflicts reconciled; main changes were broader
than parser/MCP wrappers and included construction, risk and shared telemetry.

- Bare leg numbers are strikes; explicit d/delta markers are required for delta.
  Example: 1x1.3 ATMF/25d. Ratio token remains separate from leg inputs.
- Main's public request delta range is 10-50; mathematical exact resolver still
  supports (0,1). Exact delta inversion and ATMF versus 50-delta are preserved.
- Strike construction accepts strike, delta and mixed anchors, including
  flexible ratios. Retained contracts continue to own quantities and valuation.
- Main's implausible-strike/reversed-leg checks apply to ratio_spread too.
- Risk derivation reads resolved contract quantities, not hardcoded family ratios.
  Catalog disagreement remains unknown; no automatic weakening of preferences.
- MCP and shared telemetry changes retained. MCP test environment already existed;
  no installer run, secrets edited, remote logging exercised or Claude config changed.
- tests/test_main_ratio_integration.py covers eight explicit combined cases.
- Update is via tested merge commit and fast-forward, not a second merge attempt.

Historical slice notes below predate this integration. Next user check: named
strike/delta requests in the app; push only if explicitly requested.

User approved supported flexible ratios WITH existing recommended sizing;
arbitrary combinations remain price/analysis-first. Version 0.2.56 implements
off-menu agent 1xR requests for finite R >= 1. Existing ratios keep their IDs;
others use ratio_spread with retained short_ratio and resolved +1/-R quantities.
Fixed-loss uses the existing absolute premium proxy; Kelly uses resolved payoff.
No recommendation-menu expansion, arbitrary leg editor or Kelly dropdown addition.
22 new flexible-ratio tests pass. Final full suite on the finished source tree:
1,347 passed, zero failures, six skipped in 154.60 seconds. git diff --check
passed. No cloud/UI verification. Changes are not committed or pushed.
Spec updated accordingly. Next: user testing of named 1x1.3 requests in the
agent, then checkpoint/push when requested. Do not describe this as a new
Kelly dropdown option or a general arbitrary-leg editor.

The approved behaviour-preserving slice is complete. Before enabling arbitrary
packages, specify allowed ratios/legs, risk constraints and sizing policy with
the user. Existing generic valuation does not mean arbitrary package sizing is
approved. Old private entry functions remain test references; do not delete
without retaining independent regression evidence. Check git log for the local
construction/policy checkpoint commit following 3fd4ea9.
