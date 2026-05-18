# Kelly v2 — Working Instructions

Standing instructions for any Claude Code session working in this folder. Read `PLAN.md` for the build spec.

## Scope and blast radius

- **Stay inside `experiments/Kelly v2/`.** Do not modify main app code, tests outside this folder, or root config files (`pyproject.toml`, `uv.lock`, `.streamlit/`, etc.).
- Do not alter or access any part of the rest of the MacroTool project. Treat the parent repo as read-only background — fine to glance at for reference, never to edit.
- If a build step seems to require touching anything outside this folder, **stop and ask** before proceeding.

## Dependencies

- If extra libraries or tools are needed, install them without asking. Use `uv add <pkg>` from the worktree root.
- After adding a dep, do not commit `pyproject.toml` / `uv.lock` changes as part of a Kelly v2 step. Mention them in the build-step commit message but leave them staged separately so the user can review. (Or commit them in a dedicated commit titled `deps: add <pkg> for Kelly v2`.)

## Commit policy

- **Commit after each build-order step** (see `PLAN.md` — there are 10 steps). One commit per step, clear message describing what landed.
- Commit message format: `Kelly v2: <step description>` — e.g. `Kelly v2: elicitation.py Option 1 with parametrised tests`.
- **Push at the end** of the build session, not after every commit.
- Co-author trailer: `Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>` (or whatever model is running).

## Testing

- Run unit tests after each step: `.venv/bin/python -m pytest experiments/"Kelly v2"/tests/ -v`
- Self-test the Streamlit UI using the Claude Preview MCP tools. The user will not test until the UI is built and ready.
- Self-verifiable: app boots, inputs render, constraints enforced, edge readouts update, warnings trigger at boundaries, no console errors.
- Not self-verifiable (defer to user): whether the elicitation UX *feels* right.
- If a test fails on something unrelated to Kelly v2 (e.g. a pre-existing main-app issue), skip it and note in `NOTES.md`. Do not fix.

## Running Streamlit

```bash
.venv/bin/streamlit run "experiments/Kelly v2/app.py"
```

Run in background via Bash `run_in_background: true`. Default port 8501. If port is taken, use `--server.port 8502` etc.

## Decisions and notes

- Maintain a `NOTES.md` in this folder. Log every non-trivial decision made during the build — design choices, surprises, workarounds, things flagged for revisit.
- The grid-extent truncate-with-warning decision is a **revisit candidate** (see PLAN.md). If anything during the build suggests it's wrong, log it loudly in NOTES.md.

## Token / API limits

- **If you run out of tokens or hit an API rate/spend limit, do NOT switch to a different API or model to keep going.** Stop, wait for the reset, and resume from where you left off when the session restarts.
- On resume, re-read `PLAN.md`, `CLAUDE.md`, `NOTES.md`, and the last commit message to re-establish context. Continue from the next uncompleted build step.
- Do not silently downgrade quality, skip tests, or abandon the spec to stretch remaining tokens. If you can't finish the current step cleanly within remaining budget, commit progress with a `WIP:` prefix, log status in `NOTES.md`, and stop.

## Default for ambiguity

- If something in `PLAN.md` is ambiguous and the user is not available, make the judgment call, log it in `NOTES.md` under "decisions made without sign-off", and continue. Do not block.
- If a judgment call would change the scope or DoD materially (e.g. dropping a sanity check, changing a locked-in decision), stop and wait instead.

## Out of scope for v2

- Kelly fraction itself (deferred)
- Multi-leg structures (vanilla only)
- Integration into the MacroTool main app
- Real LLM calls (this prototype is pure-numerical)
- Persistence, auth, multi-user
