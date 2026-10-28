# Product engine handoff - Codex

## Current checkpoint

- Worktree: /Users/ash/Documents/Coding work/agentic-workflow
- Branch: agentic-workflow. Do not implement this work in main.
- Pre-integration checkpoint: d561a32 (flexible ratios, clean tree).
- Current source release: f276056, v0.2.57, pushed to origin/agentic-workflow.
- User acceptance testing is pending; cloud deployment has not been independently verified.
- Integrated source full suite: 1,416 passed, zero failures, seven skipped
  (160.55 seconds). Eight additional combined-feature tests then passed.
- MCP optional suite: 13 passed using /Users/ash/macrotool-mcp/.venv/bin/python,
  with fake model and real stdio transport. No API calls or live deployment.
- The four previous recurring failures are resolved by user-approved test cleanup:
  Auth0-only expectations, GBPUSD 1M instead of missing 1W, and removal of the
  far-target/high-carry vanilla-winner test. Production behavior unchanged.
- Post-merge check on agentic-workflow: 30 ratio/integration tests passed.
- Test cleanup and all engine/integration work through f276056 are committed and
  pushed. Historical results below are retained; earlier no-push statements
  describe those checkpoints only, not the current release.

## Resume here

T23 release checkpoint (28 October, v0.2.58): user chose inception
Greeks ahead of T21. Added analytics/inception_greeks.py and the on-demand
inception_greeks agent tool. Vanilla-leg contracts only; actual signed quantities
and sized notional; no reconstruction, resizing or automatic UI reporting.
Delta is conventional spot delta in base units, gamma is delta change per 1%
spot rise, vega is base currency per one vol point at fixed inception FX.
Rates and entry leg vols stay fixed. Digitals, ERKOs and linear trades return
unavailable. Custom trades retain their inception market identity; stale ranks
or custom contexts are rejected. Final full suite: 1,482 passed, seven skipped
in 165.76 seconds, including 58 new Greeks cases and fake-model tool routing.
Delta/gamma bump checks use Richardson extrapolation, tolerance 0.02% relative
or 0.05 base units per million; vega checks use 0.5% relative or 0.5 base units
per million across half/standard/double bumps. These are numerical test
tolerances, not risk limits. Live agent/cloud acceptance has not been run.

1. Get user test feedback on familiar recommendations, explicit strike versus
   delta inputs, ratio sizing and Kelly widget updates. README links current scope.
2. After user testing of T23, next product slice: T21 selected-trade expiry payoff/net P&L charts, using the
   retained contract, premium and actual notional. No LLM numerical curve building.
3. T23 exotic Greeks remain deferred; do not substitute vanilla sensitivities.
4. Broader T22 arbitrary packages remain proposed; supported 1xR requests already
   receive existing family sizing. Specialized contracts and legacy retirement
   remain incomplete under T19. Do not mark whole workstreams done.
5. Keep urgent launch controls separate; a green pricing suite is not beta sign-off.

Current input example: `1x1.3 spread ATMF/25d`. Bare `25` is strike 25, not a delta.
No new recommendation menu or Kelly dropdown entries were introduced.

## Historical evidence

Sections below preserve the sequence and test evidence. They are not a second
current task list; the resume instructions above and BETA_IMPLEMENTATION_TODO.md
take precedence over earlier plans.

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

## Construction-policy checkpoint (historical)

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
3. At that checkpoint flexible ratios were deferred; they were subsequently
   approved and implemented in d561a32. Arbitrary packages remain deferred.
4. Specialized binary/barrier migration requires explicit contract semantics;
   preserve the negative-smile-package rejection guard. Multiple expiries need
   per-leg expiry support, not just extra legs.

## Scope boundaries

- Generic leg lists have no hardcoded maximum; current contract has one expiry.
- Arbitrary combinations require later validation and product design; supported
  two-leg 1xR ratios are implemented.
- Barriers/digitals need specialized semantics, not vanilla-leg flattening.
- Old private entry functions remain independent regression references.
- Broader roadmap: BETA_IMPLEMENTATION_TODO.md.

## Main integration - 2026-10-28

Merged main/origin/main bf12508 into checkpoint d561a32 in an isolated managed
worktree before updating agentic-workflow. No main worktree edits. The integration
was subsequently pushed as f276056 to origin/agentic-workflow.
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

Historical slice notes below predate this integration and its completed push.

User approved supported flexible ratios WITH existing recommended sizing;
arbitrary combinations remain price/analysis-first. Version 0.2.56 implements
off-menu agent 1xR requests for finite R >= 1. Existing ratios keep their IDs;
others use ratio_spread with retained short_ratio and resolved +1/-R quantities.
Fixed-loss uses the existing absolute premium proxy; Kelly uses resolved payoff.
No recommendation-menu expansion, arbitrary leg editor or Kelly dropdown addition.
22 new flexible-ratio tests pass. Final full suite on the finished source tree:
1,347 passed, zero failures, six skipped in 154.60 seconds. git diff --check
passed. No cloud/UI verification. That slice was later committed as d561a32 and
included in the f276056 push. User testing of named requests is pending.
Do not describe this as a new
Kelly dropdown option or a general arbitrary-leg editor.

The approved behaviour-preserving slice is complete. Before enabling arbitrary
packages, specify allowed ratios/legs, risk constraints and sizing policy with
the user. Existing generic valuation does not mean arbitrary package sizing is
approved. Old private entry functions remain test references; do not delete
without retaining independent regression evidence. Check git log for the local
construction/policy checkpoint commit following 3fd4ea9.
