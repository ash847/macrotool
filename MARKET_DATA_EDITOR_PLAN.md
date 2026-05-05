# Market Data Editor — Build Plan

Local-session market data overrides for the Streamlit app. PM can edit forwards and vol surfaces; edits live in `st.session_state` only — not persisted, not shared, not in Supabase.

## Scope

**Editable:** forwards, ATM vols, risk reversals, butterflies — per supported pair.
**Read-only:** DF curves, conventions, spot, instrument type.
**Validation:** Pydantic only. No no-arb checks on the smile (PM is trusted).
**Persistence:** none. Overrides clear on session end / hard refresh / app restart.
**Conversation interaction:** overrides apply to **new conversations only**. An active conversation keeps the snapshot it started with. UI must communicate this clearly.

## Architecture

### 1. Override storage — sparse dict in session state

```python
st.session_state["market_overrides"] = {
    "USDBRL": {
        "forwards": {"1M": 5.12, "3M": 5.18, ...},
        "atm_vols": {"1M": 14.2, ...},
        "risk_reversals": {...},
        "butterflies": {...},
    },
    # other pairs only present if edited
}
```

Sparse — only edited fields appear. Makes "reset this field" and "modified" indicators trivial. Easier to reason about than holding a full edited `MarketSnapshot` copy.

### 2. Snapshot construction — base + overrides → new MarketSnapshot

New module: `data/snapshot_overrides.py`

```python
def apply_overrides(base: MarketSnapshot, overrides: dict) -> MarketSnapshot:
    """Return a NEW MarketSnapshot with overrides merged onto base.
    Always constructs a fresh object — never mutates base."""
```

- Deep-copy base, walk overrides, set fields on the copy.
- Re-validate via Pydantic (`MarketSnapshot.model_validate(...)`) so any bad edit surfaces before the engine sees it.
- Pure function. No session-state knowledge. Trivially testable.

### 3. Wiring — single inject point in `interface/app.py`

Today: `interface/app.py:410` calls `load_snapshot()` and passes the result to `ConversationFlow(snapshot=...)`.

Change: at the moment a new conversation is created, build the effective snapshot:

```python
base = load_snapshot()
overrides = st.session_state.get("market_overrides", {})
effective = apply_overrides(base, overrides) if overrides else base
flow = ConversationFlow(snapshot=effective)
```

`ConversationFlow` already accepts `snapshot` (`conversation/flow.py:84`) — no change needed there.

**Important:** because overrides only take effect on flow construction, edits made *during* an active conversation do not affect that conversation. This is the simplifying assumption — accept it for v1.

### 4. UI — new "Market Data" page or sidebar section

Streamlit page (`interface/pages/market_data.py` or a section in `app.py`):

- Pair selector (USDBRL / USDTRY / EURPLN / GBPUSD).
- For the selected pair, show four editable tables:
  - Forwards (tenor → outright)
  - ATM vols (tenor → vol)
  - Risk reversals (tenor × delta → bp)
  - Butterflies (tenor × delta → bp)
- Read-only display: DF curve, spot, conventions (greyed out, with explanation: "Locked — DF curves are the basis for rate derivation; editing them would create inconsistencies with forwards.").
- "Modified" badge on any field where override differs from base.
- Per-field reset (×) and global "Reset all to base" button.
- Banner if a conversation is active: *"Edits apply to your next conversation. Click 'New conversation' to use them."*

**Editor mechanic:** start with `st.data_editor` for each table — it gives free dirty-tracking and validation. Catch the diff vs. base, store sparse in session state.

### 5. Reset surface

- **Per-field reset:** click × on a modified row → remove that key from overrides.
- **Per-pair reset:** button per pair → delete that pair from overrides.
- **Global reset:** "Reset all market data to base" → `del st.session_state["market_overrides"]`.
- **Implicit reset:** session end / hard refresh → Streamlit clears session state automatically.

## Implementation order

1. **`data/snapshot_overrides.py`** with `apply_overrides()` + unit tests.
   - Tests: empty overrides returns equal-content copy; partial forward override touches only that tenor; invalid override raises `ValidationError`; base snapshot is not mutated.
2. **Wire into `interface/app.py`** at flow construction. Verify with no overrides set → behavior unchanged (regression check via existing demo flow).
3. **Build the Market Data UI** — forwards only, single pair (USDBRL) first. Prove the edit → session state → next conversation flow end-to-end.
4. **Extend to vols** (ATM, RR, BF) once forwards path is solid.
5. **Extend to all four pairs.**
6. **Polish:** modified badges, reset buttons, "edits apply to next conversation" banner.

## Out of scope (explicit non-goals for v1)

- Persistence across sessions.
- Sharing overrides between users.
- Hot-applying edits to a live conversation.
- No-arb / smile sanity checks.
- Editing DF curves, spot, or conventions.
- Audit history / undo stack beyond per-field reset.
- Import/export of override sets.

## Risks / things to watch

- **Pydantic re-validation cost** — `MarketSnapshot` is small; should be negligible. Measure if it shows up.
- **`st.data_editor` quirks with nested dicts** — may need to flatten to a DataFrame for the editor and rebuild the dict on save.
- **Tenor/delta key consistency** — overrides must use exactly the same string keys as base (e.g. `"1M"` not `"1m"`). The merge step should fail loudly on unknown keys rather than silently ignoring them.
- **Rate context derivation** — `rate_context_for_snapshot` derives `r_d` via CIP from forwards. Editing forwards therefore *implicitly* edits the implied quote rate. This is correct and intended, but worth surfacing in the UI ("Editing forwards changes the implied carry").

## Test plan

- Unit: `apply_overrides` correctness + immutability of base.
- Unit: round-trip — base → overrides applied → key fields match overridden values, untouched fields match base.
- Manual: edit a 1M forward in USDBRL, start a new conversation, confirm validation step reflects the new carry/with_carry.
- Manual: edit during active conversation → confirm banner shows, current conversation's analysis unchanged, next conversation picks up the edit.
- Manual: hard refresh → overrides gone, base restored.
