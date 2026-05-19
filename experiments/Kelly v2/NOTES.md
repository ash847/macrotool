# Kelly v2 — Build Notes

Running log of decisions, surprises, and revisit candidates.

## Environment

- Shared `.venv` lives at the parent MacroTool repo root, not in this worktree. Run Python via the absolute path `/Users/ash/Documents/Coding work/MacroTool/.venv/bin/python`. Streamlit and pytest binaries similarly absolute. Not a problem, just noting.
- Folder name `Kelly v2` contains a space. Pytest and Streamlit handle it fine with quoted paths.

## Design choices made during build

### Step 1 — elicitation.py Option 1

- **Bin model:** bin *edges* (n_bins + 1 of them) over [p_outer_low, p_outer_high], `probs[i] = CDF(edge[i+1]) − CDF(edge[i])`, `bins[i]` = bin centre. Cleaner than centre-based finite differences and means renormalisation only needs to absorb the truncated tail mass (~`quantiles[0] + (1 − quantiles[-1])`), not floating-point noise.
- **Renormalisation:** clip raw probs to ≥ 0 (kills tiny PCHIP overshoot), then divide by sum. After renormalisation `probs` sums to exactly 1 in float.
- **Minimum N:** 3 anchors. Fewer than 3 doesn't make sense for a non-trivial CDF.
- **No tail extension:** grid truncates to [prices[0], prices[-1]]; mass outside the outer anchors is dropped. Matches PLAN.md decision. **Revisit candidate** — log here if real usage suggests we need a parametric tail.

## Revisit candidates

- Grid extent (truncate-with-warning vs parametric tail) — see PLAN.md decisions section.
