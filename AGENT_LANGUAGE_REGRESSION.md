# Agent language and regression checks

`knowledge/defaults/agent_vocabulary.json` holds tunable shortlist caveats and
narration rules. `knowledge_engine.loader.load_agent_vocabulary` uses the existing
cached JSON loading convention: restart the process after editing. The prompt
includes these rules and the public shortlist consumes the shared caveats. No
computed financial values, risk classifications or pricing formulas live here.

Numerical ranges are engine-authored. European RKO ranges now display the lower
level first, preserving which endpoint is the strike versus knock-out. This is a
wording change only. Leg order and ratios are never sorted. Monetary formatting
retains full units and currency; negative zero is displayed as zero. Missing values
remain unavailable rather than being converted to zero.

Focused tests cover both directions, currency/scale formatting, premium signs,
zero allocation, range endpoint roles, configured language, cached comparisons and
sizing follow-ups, and rejection of old references after a market/weight change.
Existing financial, risk-policy, sizing, personal-weight and workspace tests cover
the other cross-cutting invariants. These do not certify the deferred live-data
refresh or linked Trade View preference-parity changes.

Live conversational acceptance remains a separate check. The existing live smoke
test skips when ANTHROPIC_API_KEY is absent. Provider-free scripted tests verify
orchestration and supplied facts, not real-model routing or prose quality.

Manual acceptance sequence for both higher and lower views:
1. Request a three-month USDBRL recommendation with a six-percent move. Expect
   the engine table and a short explanation, not five essays or internal scores.
2. Ask “Compare 1 and 3”, then “Why this size for 2?” and “Risk on all five”.
   Expect the same retained trades, no repricing, and recorded sizing/risk facts.
3. Ask for a zero-cost or credit construction. Target return on premium must be
   not applicable; credit must not be called premium paid or imply no risk.
4. Change the view or apply settings, then refer explicitly to an older shortlist.
   Expect clarification rather than silently reassigning old rank numbers.
5. Try Kelly without a supplied distribution. Expect the fixed-loss fallback
   disclosure, not invented Kelly figures. An unavailable figure is not zero.

No pricing, ranking, sizing, market-refresh or eligibility policy changes are
introduced by this step. Package version: 0.2.19.
