# Flexible package construction - Codex

Status: supported-ratio slice implemented; arbitrary-package spec remains proposed,
2026-10-28. See the scope distinction below.
Worktree: agentic-workflow. Roadmap item: T22 in BETA_IMPLEMENTATION_TODO.md.
Release: v0.2.57 / f276056, pushed to origin/agentic-workflow. User testing pending.

## Approved refinement and first implementation

The user approved automatic sizing for flexible ratios within supported families,
while arbitrary leg combinations remain price/analysis-first. The blanket sizing
gate below applies to arbitrary combinations, not supported ratio spreads.

Implemented first slice: explicit agent requests `1xR` with finite R >= 1,
including `1x1.3 spread ATMF/25d`, delta/delta and existing anchor/target forms.
After main integration, bare leg numbers mean strikes; explicit delta markers
are required. The ratio token itself is never a strike or delta.
Main integration also enables literal strike and mixed strike/delta legs through
the same construction path. Implausible strikes and reversed spreads are refused,
never reinterpreted as deltas. General arbitrary-leg construction remains deferred.
Ratios 1, 1.5 and 2 retain existing family IDs; other ratios use ratio_spread.
Fixed-loss uses the existing ratio-family premium proxy and cap; Kelly uses the
actual +1/-R contract. No new financial sizing rule, automatic candidate search,
structured leg editor or Kelly dropdown integration is introduced. Direction
continues to come from the active session. Ratios below one are explicitly
rejected in this slice, rather than being misinterpreted as an existing family.
Unknown tail/loss bounds stay unknown; existing preference gates remain active.
The arbitrary-package specification below remains proposed, not implemented.

## Proposed arbitrary-package scope (not implemented)

- Same supported pair and expiry for every leg; European vanilla calls/puts.
- Explicit buy/sell, positive ratio, and call/put per leg.
- Strike anchor: outright strike, absolute option delta, or ATMF.
- Mixed calls/puts and mixed anchor types are allowed when explicitly supplied.
- Start with a structured leg editor; keep the existing named-family requests.
- Custom trades are user-requested, not new candidates automatically generated
  or ranked by the recommendation engine.
- No calendars, barriers, digitals, cross-pair packages or new zero-cost solver.
- Existing curated seagull funding and butterfly construction remain unchanged.

Example: buy 1 call at 25 delta, sell 1.3 calls at 10 delta. Construction
resolves both strikes using the current smile. Scenario and Kelly calculations
then keep those strikes and the 1:-1.3 ratio fixed.

## Input and validation

Use a separate typed CustomPackageRequest, not a permissive extension of the
legacy text parser. Every leg carries its own option right, side, ratio and
anchor. Pair and expiry are taken explicitly from the active trade context.

- Reject missing/ambiguous side or right; do not infer either from session direction.
- Reject nonfinite/zero ratios, nonpositive strikes and out-of-range deltas.
- Delta magnitude must be strictly between zero and one; call/put is separate.
- ATMF is a distinct anchor, not a synonym for 50 delta.
- Preserve user ratios without silently rescaling them. For package notional N,
  a +1/-1.3 package has leg notionals +N/-1.3N in the pair's base currency.
- Show entered legs and resolved strikes/quantities before treating a package
  as the active trade. Preserve order and a stable identity, not just a label.
- Reject an empty or completely offsetting package. Duplicate-leg netting must
  preserve the submitted description and audit record.
- A maximum leg count and ratio limits are input controls, not pricing limits.
  Proposed initial UI cap: four legs. Numeric ratio limits need explicit approval.

## Existing code and required extensions

1. agentic/structure_request.py strips call/put/long/short words and obtains
   ratios from named families or explicit 1xR requests. Do not send arbitrary
   per-leg buy/sell/right combinations through it.
2. product_model.Structure and ResolvedContract support variable leg counts.
   product_pricer.construct_package allowlists supported vanilla-leg families
   including ratio_spread. Add an
   explicit custom-vanilla construction path with validation, not a blanket
   bypass of the family guard. No derived wings or quantity solving there.
3. product_policies.sizing_loss_proxy deliberately rejects unknown families.
   Separate a valid price from unavailable sizing: do not fail pricing or
   invent a max-loss value of zero just because sizing is not approved.
4. Generic resolved valuation and terminal payoff can be reused. Breakevens
   must use actual quantities, not a familiar family name's assumed ratios.
5. agentic/price_structure.py uses family-based constraints/attributes. Custom
   risk exclusions must use verified package properties; unknown means unknown,
   not safe. Do not classify every non-vanilla package as capped.
6. interface/kelly_v2/app.py filters known family IDs. Add an explicit custom
   handoff that carries the resolved contract and stable identity. No rebuilding
   from labels. Existing sparse baseline reseeding rules still apply.
7. Persist request and resolved contract together so a reload cannot resolve
   a different trade against a newer market snapshot.

## Risk and sizing boundary

Price, terminal payoff and scenario P&L are separate from permission to size.
Show gross payoff, entry premium and net P&L as distinct quantities.
Show the risk basis/currency alongside every loss statement.

Derive capped gains, losses beyond premium and tail exposure from the resolved
piecewise payoff, including strike breakpoints and both spot limits. A finite
scenario grid or a familiar family label is not proof of bounded loss. In
particular, quote-currency and base-currency tail limits differ because terminal
normalization divides by terminal spot. Preserve the existing premium basis.

Proposed release gate: price and analyze custom packages first; expose automatic
fixed-loss or Kelly sizing only after a custom-package sizing policy is approved.
For open-tail packages, a finite PM distribution does not establish a worst-case
loss bound. Any future sizing must state the distribution support, tail treatment,
budget denominator and notional cap explicitly. Keep all existing curated-family
sizing unchanged. Never silently borrow the 1x2 premium-only proxy for custom legs.

## Implementation slices

- [ ] 1. Typed custom request and deterministic validation; explicit rejection tests.
- [ ] 2. Custom construction plus price-only output, preserving resolved leg identity.
- [ ] 3. Risk classification and scenario results with honest unavailable-sizing state.
- [ ] 4. Structured editor, persistence and contract-based Kelly handoff, subject
  to the sizing release gate above. No new automatic recommendation candidates.
- [ ] 5. Approved sizing policy for arbitrary packages. T21 charts and T23 Greeks
  can proceed independently on already supported resolved contracts.

## Acceptance tests

- 1:1, 1:1.5 and 1:2 explicit legs match equivalent existing named trades.
- Mixed calls/puts and four-leg packages equal independent weighted leg prices.
- Repricing never changes strikes/quantities, including after changing PM bins.
- Leg permutation preserves value; common ratio scaling scales values linearly.
- Exact offsetting legs, malformed inputs and missing sizing policy are handled explicitly.
- Tail classification covers debit, credit, zero-cost, short-call and short-put
  examples in both quote and base currency; no finite-grid safety assumption.
- Recommendations, custom display, saved/reloaded trade and Kelly see the same
  resolved contract. Pair/expiry edits invalidate the old construction deliberately.
- Current curated recommendations, preference gates and sizing stay unchanged.

## Decision before arbitrary-package implementation

Approve price-and-analysis first with automatic custom sizing withheld, or define
the custom sizing rule now. Four-leg UI cap is proposed; ratio limits also need
agreement before exposing inputs. No new financial policy is implied by this doc.
