Current goal

- Switch comparator inputs and ranking to PM-weighted currency P&L rather than percent P&L.
- Ensure comparisons are by concrete structure variant, including strike and barrier details.
- Leave the Structure Evaluation UI behavior unchanged except for the admin explanation-pack preview using the same comparator basis.

Files changed

- `knowledge_engine/comparator.py`
- `conversation/explanation_context.py`
- `interface/app.py`
- `tests/test_comparator.py`
- `tests/test_explanation_context.py`

Decisions made

- Comparator ranking now uses `score_ccy` when available.
- Variant-level ranking is the primary comparator basis when `variant_evaluations_by_structure` is available.
- Chosen display text in the explanation pack now includes the concrete variant label with strikes and KO barrier when relevant.
- Structure-level fallback behavior is still retained when variant evaluations are not supplied.
- Structure Evaluation UI was not changed for end users; only the admin explanation preview was wired to pass variant evaluations through.

Bugs remaining / known caveats

- No failing focused comparator tests remain.
- Comparator code now mixes structure-level pairwise comparisons and variant-level comparisons in the explanation pack; that is intentional for now, but could be simplified later if the product direction becomes fully variant-centric.
- The comparator implementation lives on `claude/charming-leavitt-47b984`, not on `worktree-claude`.

Exact commands run

- `git status --short --branch`
- `git log --oneline --decorate -5`
- `git log --all --oneline --decorate --grep=comparator --grep=explanation --grep='pack preview'`
- `git diff --stat`
- `rg -n "comparator|comparison pack|pack preview|pm weighted|weighted pnl|score_pct|score_ccy|variant_label_with_strikes|specific variant" -S`
- `sed -n '1,260p' knowledge_engine/comparator.py`
- `sed -n '260,620p' knowledge_engine/comparator.py`
- `sed -n '1,260p' tests/test_comparator.py`
- `sed -n '260,520p' tests/test_comparator.py`
- `sed -n '1,220p' tests/test_explanation_context.py`
- `sed -n '1,220p' conversation/explanation_context.py`
- `sed -n '1045,1235p' interface/app.py`
- `python3 -m pytest tests/test_comparator.py tests/test_explanation_context.py`
- `'/Users/ash/Documents/Coding work/MacroTool/.venv/bin/python' -m pytest tests/test_comparator.py tests/test_explanation_context.py`
- `'/Users/ash/Documents/Coding work/MacroTool/.venv/bin/python' -m py_compile knowledge_engine/comparator.py conversation/explanation_context.py interface/app.py`

Next steps

- If desired, merge or cherry-pick this branch into the branch that will actually be used for the next comparator integration step.
- If follow-up prompt wiring is next, use the variant-centric fields already added to the explanation pack rather than rebuilding ranking logic.
- If PM-facing comparator output should become fully variant-centric, consider replacing the remaining structure-level `comparisons` block with variant-to-variant comparisons.

Latest checkpoint

- Follow-up prompt wiring is now implemented on `claude/charming-leavitt-47b984` but not yet merged elsewhere.
- `conversation/flow.py` now builds a rendered recommendation explanation pack after recommend-mode engine runs, best-effort only, and stores it on `flow.explanation_pack_context` for later use.
- `conversation/context_builder.py` now injects that explanation pack into the DONE/follow-up prompt only when present, and instructs the model to use it first for "why this / why not that / why this variant" questions while keeping the answer qualitative.
- `tests/test_followup_prompt.py` verifies the follow-up prompt includes the explanation-pack instruction and block only when an explanation context is supplied.
- Focused regression check passed:
  - `'/Users/ash/Documents/Coding work/MacroTool/.venv/bin/python' -m pytest tests/test_followup_prompt.py tests/test_explanation_context.py tests/test_comparator.py`

Current product impact

- No change to the visible structured Trade View recommendation path is expected from this checkpoint alone.
- The explanation pack preview should remain unchanged in intent; this step mainly makes the same pack available to any follow-up conversation path that uses `ConversationFlow`.
- The current Streamlit Trade View intake remains deterministic and does not call `flow.advance(...)`, so this checkpoint should not add active Anthropic API usage on the normal structured submission path.

Architecture note

- The conversation/LLM path is still the intended target architecture and should not be treated as dead code.
- The current Trade View UI is intentionally keeping that path silent while we test the deterministic engine pipeline and comparator plumbing.
- Doc and copy clean-up should describe this as "structured silent mode" or equivalent, rather than implying the LLM path has been removed.

Latest UI checkpoint

- The latest pushed branch state is `7305160` on `claude/charming-leavitt-47b984`.
- The admin `Explanation pack preview` is now split into separate sections under the preview expander:
  - overview
  - variant comparisons
  - structure comparisons
- `conversation/explanation_context.py` now exposes separate render helpers for the overview, variant-comparison section, and structure-comparison section, in addition to the full combined renderer.
- `interface/app.py` now uses those separate render helpers so the preview is easier to inspect and reason about during tuning.
- The same commit also aligned code comments, UI copy, and project docs around the intended architecture wording:
  - the conversation/LLM path remains the target architecture
  - the current Trade View page is running in a structured silent mode while deterministic pipes are being tested

Recommended next step

- If the next comparator iteration stays variant-first, enrich the variant-comparison section so it can eventually carry the same level of detail as the current structure-comparison section, then route or prioritize the relevant section based on the PM's follow-up question.
