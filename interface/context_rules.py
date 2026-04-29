"""
Context Rules page — view and edit scenario_weights.json contexts.

Three tabs:
  1. Context weights    — editable matrix of contexts × families.
     Each cell shows the relative weight multiplier (1.00 = no change,
     >1 = bumped up, <1 = reduced). Changes are saved to Supabase and
     every version is stored. The weighter picks up the latest version
     on the next trade query.

  2. Choosing a context — read-only view of the conditions that cause
     each context to fire, formatted in plain English.

  3. Priority & conditions — edit the context evaluation order and the
     MarketState conditions attached to each context.  First-match
     selection means order is semantically important.  Live preview
     shows which context fires for any set of market-state inputs.
"""

from __future__ import annotations

import copy
import uuid

import pandas as pd
import streamlit as st

from analytics.scenario_generator import FAMILIES
from knowledge_engine.scenario_weighter import (
    clear_scenario_weights_cache,
    load_scenario_weights_config,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Ordered for intuitive UI display (not alphabetical).
_COND_FIELDS = [
    "carry_regime",
    "with_carry",
    "T",
    "vol",
    "target_z_abs",
    "atmfsratio",
]
_COND_OPS = ["==", "!=", ">", ">=", "<", "<="]

# ---------------------------------------------------------------------------
# Shared helpers (used by Tab 2 and Tab 3)
# ---------------------------------------------------------------------------

_FIELD_LABELS = {
    "target_z_abs": "Target distance (σ)",
    "carry_regime": "Carry regime",
    "with_carry":   "Direction vs carry",
    "T":            "Tenor (years)",
    "vol":          "ATM vol",
    "atmfsratio":   "ATM/FS ratio",
}

_VALUE_LABELS: dict = {
    0:     "0 (noisy)",
    1:     "1 (potential)",
    2:     "2 (high)",
    True:  "with carry",
    False: "counter carry",
}


def _fmt_value(field: str, value) -> str:
    if field == "T":
        months = value * 12
        return f"{months:.0f}m ({value}y)"
    if field == "vol":
        return f"{value:.0%}"
    if isinstance(value, bool):
        return _VALUE_LABELS.get(value, str(value))
    if isinstance(value, int):
        return _VALUE_LABELS.get(value, str(value))
    return str(value)


def _fmt_condition(cond: dict) -> str:
    field = _FIELD_LABELS.get(cond["field"], cond["field"])
    op = cond["op"]
    val = _fmt_value(cond["field"], cond["value"])
    return f"{field} {op} {val}"


def _fmt_conditions(when: list[dict]) -> str:
    if not when:
        return "Always"
    return "  AND  ".join(_fmt_condition(c) for c in when)


# ---------------------------------------------------------------------------
# Tab 1 helpers — weight matrix
# ---------------------------------------------------------------------------

def _build_matrix(cfg: dict) -> pd.DataFrame:
    """Build a context × family multiplier matrix (1.0 = no change)."""
    rows = []
    for ctx in cfg["contexts"]:
        adj = ctx["adjustments"]
        row = {"Context": ctx["id"].replace("_", " ")}
        for fam in FAMILIES:
            delta = adj.get(fam, 0.0)
            row[fam.replace("_", " ").title()] = round(1.0 + delta, 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Context")


def _rebuild_config(cfg: dict, edited_df: pd.DataFrame) -> dict:
    """Reconstruct the full config dict from an edited weight matrix."""
    new_cfg = copy.deepcopy(cfg)
    for ctx in new_cfg["contexts"]:
        ctx_label = ctx["id"].replace("_", " ")
        new_adj: dict[str, float] = {}
        for fam in FAMILIES:
            fam_col = fam.replace("_", " ").title()
            val = float(edited_df.loc[ctx_label, fam_col])
            delta = round(val - 1.0, 4)
            if abs(delta) > 1e-6:
                new_adj[fam] = delta
        ctx["adjustments"] = new_adj
    return new_cfg


# ---------------------------------------------------------------------------
# Tab 1 — Context weights (editable)
# ---------------------------------------------------------------------------

def _render_context_weights(cfg: dict) -> None:
    st.caption(
        "Each cell is the relative weight multiplier applied to a scenario family "
        "when this context is active.  "
        "**1.00** = no change (baseline).  **> 1** = bumped up.  **< 1** = reduced.  "
        "Weights are re-normalised after all contexts fire so the full vector sums to 1.  "
        "Edit any cell and press **Save changes** — every version is stored."
    )

    fam_cols = [f.replace("_", " ").title() for f in FAMILIES]
    df = _build_matrix(cfg)

    # Editable matrix
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        key="ctx_weights_editor",
        column_config={
            col: st.column_config.NumberColumn(
                label=col,
                min_value=0.0,
                max_value=3.0,
                step=0.01,
                format="%.2f",
            )
            for col in fam_cols
        },
    )

    st.caption(
        f"Baseline: **{cfg['baseline']:.3f}** (1/{len(FAMILIES)}).  "
        f"Floor per family after normalisation: **{cfg['floor']:.2f}**."
    )

    # Save / revert controls
    col_save, col_revert, _ = st.columns([1, 1, 4])
    with col_save:
        if st.button("Save changes", type="primary", use_container_width=True):
            try:
                from interface.supabase_logger import save_config as _save
                new_cfg = _rebuild_config(cfg, edited_df)
                ok = _save("scenario_weights", new_cfg)
                if ok:
                    clear_scenario_weights_cache()
                    st.success("Saved. New weights apply on the next trade query.")
                else:
                    st.error("Save failed — Supabase not configured or unreachable.")
            except Exception as e:
                st.error(f"Save error: {e}")

    with col_revert:
        if st.button("Revert to defaults", use_container_width=True):
            clear_scenario_weights_cache()
            st.rerun()



# ---------------------------------------------------------------------------
# Tab 2 — Choosing a context (read-only)
# ---------------------------------------------------------------------------

def _render_choosing_a_context(cfg: dict) -> None:
    st.caption(
        "Contexts are evaluated **top-to-bottom**. The first one whose conditions are all "
        "satisfied is selected — at most one fires per trade right now. "
        "Order matters: more specific conditions appear higher in the list and take priority. "
        "A second context from user preferences (Tier 2) will be added later and blended with this one."
    )

    rows = []
    for ctx in cfg["contexts"]:
        when_str = _fmt_conditions(ctx.get("when", []))
        adj = ctx["adjustments"]
        families_str = "  /  ".join(
            f"{f.replace('_', ' ').title()} {d:+.2f}"
            for f, d in adj.items()
        )
        rows.append({
            "Context":    ctx["id"].replace("_", " "),
            "Fires when": when_str,
            "Adjusts":    families_str,
            "Reasoning":  ctx.get("comment", ""),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Context":    st.column_config.TextColumn(width="small"),
            "Fires when": st.column_config.TextColumn(width="medium"),
            "Adjusts":    st.column_config.TextColumn(width="medium"),
            "Reasoning":  st.column_config.TextColumn(width="large"),
        },
    )


# ---------------------------------------------------------------------------
# Tab 3 helpers — priority & conditions editor
# ---------------------------------------------------------------------------

def _init_priority_state(cfg: dict) -> list[dict]:
    """Deep-copy contexts from cfg and assign stable UIDs for widget keying."""
    result = []
    for ctx in cfg["contexts"]:
        c = copy.deepcopy(ctx)
        c["_uid"] = str(uuid.uuid4())
        result.append(c)
    return result


def _parse_condition_value(s: str):
    """
    Parse a string from the conditions editor into a typed Python value.
    Raises ValueError if not parseable.
    """
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    return float(s)


def _conditions_to_df(when: list[dict]) -> pd.DataFrame:
    """Convert a `when` condition list to a DataFrame for st.data_editor."""
    rows = []
    for c in when:
        raw = c["value"]
        val_str = "true" if raw is True else "false" if raw is False else str(raw)
        rows.append({"field": c["field"], "op": c["op"], "value": val_str})
    return pd.DataFrame(rows, columns=["field", "op", "value"])


def _df_to_conditions(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """
    Convert a data_editor DataFrame back to a `when` condition list.
    Returns (conditions, parse_errors).  Rows with all-empty cells are skipped.
    """
    conds: list[dict] = []
    errors: list[str] = []
    for idx, row in df.iterrows():
        field = str(row.get("field", "") or "").strip()
        op    = str(row.get("op",    "") or "").strip()
        val_s = str(row.get("value", "") or "").strip()

        if not field and not op and not val_s:
            continue  # blank row — skip silently

        if not field or field not in _COND_FIELDS:
            errors.append(f"Row {idx + 1}: unknown field '{field}'")
            continue
        if not op or op not in _COND_OPS:
            errors.append(f"Row {idx + 1}: unknown op '{op}'")
            continue
        if not val_s:
            errors.append(f"Row {idx + 1}: missing value")
            continue
        try:
            value = _parse_condition_value(val_s)
        except (ValueError, TypeError):
            errors.append(f"Row {idx + 1}: cannot parse value '{val_s}'")
            continue
        conds.append({"field": field, "op": op, "value": value})
    return conds, errors


def _check_shadowing(contexts: list[dict]) -> list[str]:
    """
    Detect obvious shadowing: context B is shadowed by an earlier context A
    when all of A's conditions are an exact subset of B's — meaning whenever
    B would fire, A fires first.  Returns warning strings (empty = none found).
    This is a heuristic; it catches copy-paste errors but not inequality bounds.
    """
    warnings: list[str] = []
    for j, ctx_b in enumerate(contexts):
        when_b = ctx_b.get("when", [])
        for i in range(j):
            ctx_a = contexts[i]
            when_a = ctx_a.get("when", [])

            if not when_a:
                # A has no conditions → always fires first
                warnings.append(
                    f"**{ctx_b['id']}** (#{j + 1}) can never fire — "
                    f"**{ctx_a['id']}** (#{i + 1}) has no conditions and always fires first."
                )
                break  # only one warning per shadowed context

            def _eq(c1: dict, c2: dict) -> bool:
                return (c1["field"] == c2["field"]
                        and c1["op"] == c2["op"]
                        and c1["value"] == c2["value"])

            a_subset_of_b = (
                len(when_a) < len(when_b)
                and all(any(_eq(ca, cb) for cb in when_b) for ca in when_a)
            )
            if a_subset_of_b:
                warnings.append(
                    f"**{ctx_b['id']}** (#{j + 1}) may be shadowed by "
                    f"**{ctx_a['id']}** (#{i + 1}) — "
                    f"all of {ctx_a['id']}'s conditions are a subset of "
                    f"{ctx_b['id']}'s, so {ctx_a['id']} always fires first."
                )
    return warnings


def _simulate_context_fire(contexts: list[dict], field_values: dict) -> str | None:
    """
    Simulate first-match context selection from a dict of field values.
    Returns the winning context id, or None if no context matches.
    Used for the live preview — does not go through the full MarketState pipeline.
    """
    _ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">":  lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<":  lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }
    _getters = {
        "target_z_abs": lambda fv: fv.get("target_z_abs"),
        "carry_regime": lambda fv: fv.get("carry_regime"),
        "with_carry":   lambda fv: fv.get("with_carry"),
        "T":            lambda fv: fv.get("T"),
        "vol":          lambda fv: fv.get("vol"),
        "atmfsratio":   lambda fv: fv.get("atmfsratio"),
    }

    for ctx in contexts:
        match = True
        for cond in ctx.get("when", []):
            getter  = _getters.get(cond["field"])
            op_fn   = _ops.get(cond["op"])
            if getter is None or op_fn is None:
                match = False
                break
            actual = getter(field_values)
            if actual is None:
                match = False
                break
            try:
                if not op_fn(actual, cond["value"]):
                    match = False
                    break
            except Exception:
                match = False
                break
        if match:
            return ctx["id"]
    return None


def _validate_contexts(contexts: list[dict]) -> list[str]:
    """Return a list of error strings. Empty list means valid."""
    errors: list[str] = []
    ids = [str(ctx.get("id", "")).strip() for ctx in contexts]

    for i, ctx_id in enumerate(ids):
        if not ctx_id:
            errors.append(f"Context #{i + 1} has an empty ID.")

    seen: set[str] = set()
    for ctx_id in ids:
        if ctx_id and ctx_id in seen:
            errors.append(f"Duplicate context ID: '{ctx_id}'.")
        seen.add(ctx_id)

    return errors


# ---------------------------------------------------------------------------
# Tab 3 — Priority & conditions (editor)
# ---------------------------------------------------------------------------

_PRIORITY_STATE_KEY = "ctx_priority_edit"


def _render_priority_conditions(cfg: dict) -> None:
    if _PRIORITY_STATE_KEY not in st.session_state:
        st.session_state[_PRIORITY_STATE_KEY] = _init_priority_state(cfg)

    contexts: list[dict] = st.session_state[_PRIORITY_STATE_KEY]

    st.caption(
        "Contexts are evaluated **top-to-bottom** — the first one whose conditions "
        "are all satisfied fires and applies its weight adjustments. "
        "Use **▲ / ▼** to change priority order. "
        "Edit conditions inline — add or delete rows using the table controls. "
        "Press **Save** to persist changes to Supabase. "
        "Changes here do not affect the weight values edited in the **Context weights** tab."
    )

    # -----------------------------------------------------------------------
    # Live preview
    # -----------------------------------------------------------------------
    st.subheader("Live preview")
    st.caption(
        "Set market-state values to see which context fires under first-match selection."
    )

    p = st.columns(6)
    with p[0]:
        prev_carry = st.selectbox(
            "Carry regime", [0, 1, 2], index=1, key="pv_carry",
            help="0 = noisy  /  1 = potential  /  2 = high"
        )
    with p[1]:
        prev_wc = st.selectbox(
            "With carry", [True, False], index=0,
            format_func=lambda x: "with carry" if x else "counter carry",
            key="pv_wc",
        )
    with p[2]:
        prev_T = st.number_input(
            "Tenor (yrs)", min_value=0.01, max_value=3.0,
            value=0.25, step=0.01, format="%.2f", key="pv_T",
        )
    with p[3]:
        prev_vol = st.number_input(
            "ATM vol", min_value=0.01, max_value=1.0,
            value=0.12, step=0.01, format="%.2f", key="pv_vol",
        )
    with p[4]:
        prev_tz = st.number_input(
            "Target |σ|", min_value=0.0, max_value=5.0,
            value=1.0, step=0.1, format="%.2f", key="pv_tz",
        )
    with p[5]:
        prev_atmfs = st.number_input(
            "ATM/FS ratio", min_value=0.0, max_value=10.0,
            value=1.0, step=0.1, format="%.2f", key="pv_atmfs",
        )

    field_values = {
        "carry_regime":  prev_carry,
        "with_carry":    prev_wc,
        "T":             prev_T,
        "vol":           prev_vol,
        "target_z_abs":  prev_tz,
        "atmfsratio":    prev_atmfs,
    }
    fired = _simulate_context_fire(contexts, field_values)
    if fired:
        st.success(f"Fires: **{fired}**")
    else:
        st.info("No context matches → **baseline (equal weights)**")

    st.divider()

    # -----------------------------------------------------------------------
    # Shadow-check warnings
    # -----------------------------------------------------------------------
    for warn in _check_shadowing(contexts):
        st.warning(warn)

    # -----------------------------------------------------------------------
    # Per-context cards
    # -----------------------------------------------------------------------
    for i, ctx in enumerate(contexts):
        uid = ctx.get("_uid", str(i))
        n_conds = len(ctx.get("when", []))
        cond_lbl = f"{n_conds} condition{'s' if n_conds != 1 else ''}"
        card_label = f"**#{i + 1}** — {ctx.get('id', '(unnamed)')}  ·  {cond_lbl}"

        with st.expander(card_label, expanded=False):
            # Reorder / delete
            b = st.columns([1, 1, 1, 5])
            with b[0]:
                if st.button("▲ Up", key=f"up_{uid}", disabled=(i == 0),
                             use_container_width=True):
                    contexts[i - 1], contexts[i] = contexts[i], contexts[i - 1]
                    st.rerun()
            with b[1]:
                if st.button("▼ Down", key=f"dn_{uid}",
                             disabled=(i == len(contexts) - 1),
                             use_container_width=True):
                    contexts[i], contexts[i + 1] = contexts[i + 1], contexts[i]
                    st.rerun()
            with b[2]:
                if st.button("Delete", key=f"del_{uid}", use_container_width=True):
                    contexts.pop(i)
                    st.rerun()

            # ID and comment
            id_col, cmt_col = st.columns([1, 2])
            with id_col:
                new_id = st.text_input(
                    "Context ID", value=ctx.get("id", ""), key=f"cid_{uid}",
                    help="Stable identifier used in the UI and version history.",
                )
                ctx["id"] = new_id
            with cmt_col:
                new_comment = st.text_area(
                    "Reasoning / comment",
                    value=ctx.get("comment", ""),
                    key=f"cmt_{uid}",
                    height=110,
                    help="Plain-English description of the trade archetype and why "
                         "the weights are set as they are.",
                )
                ctx["comment"] = new_comment

            # Conditions editor
            st.write("**Conditions** — ALL must be true for this context to fire:")
            cond_df = _conditions_to_df(ctx.get("when", []))
            edited_df = st.data_editor(
                cond_df,
                key=f"cond_{uid}",
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "field": st.column_config.SelectboxColumn(
                        "Field",
                        options=_COND_FIELDS,
                        required=True,
                        help="MarketState attribute to test",
                    ),
                    "op": st.column_config.SelectboxColumn(
                        "Op",
                        options=_COND_OPS,
                        required=True,
                    ),
                    "value": st.column_config.TextColumn(
                        "Value",
                        help=(
                            "Booleans: true / false.  "
                            "carry_regime: 0 / 1 / 2.  "
                            "vol / T / target_z_abs: decimal (e.g. 0.20, 0.083, 1.5)."
                        ),
                    ),
                },
            )
            new_conds, parse_errs = _df_to_conditions(edited_df)
            for err in parse_errs:
                st.caption(f"⚠️ {err}")
            ctx["when"] = new_conds

    st.divider()

    # Add context
    if st.button("➕ Add context", key="add_ctx_btn"):
        contexts.append({
            "_uid":        str(uuid.uuid4()),
            "id":          f"new_context_{len(contexts) + 1}",
            "comment":     "",
            "when":        [],
            "adjustments": {},
        })
        st.rerun()

    st.divider()

    # Save / Revert
    save_col, revert_col, _ = st.columns([1, 1, 4])
    with save_col:
        if st.button("Save changes", type="primary", key="save_prio",
                     use_container_width=True):
            errors = _validate_contexts(contexts)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                try:
                    from interface.supabase_logger import save_config as _save_cfg

                    # Re-fetch latest config so we don't overwrite adjustments
                    # saved in Tab 1 (Context weights) during this session.
                    clear_scenario_weights_cache()
                    latest_cfg = load_scenario_weights_config()
                    latest_adj: dict[str, dict] = {
                        c["id"]: c.get("adjustments", {})
                        for c in latest_cfg.get("contexts", [])
                    }

                    new_cfg = copy.deepcopy(latest_cfg)
                    new_cfg["contexts"] = []
                    for ctx in contexts:
                        clean = {k: v for k, v in ctx.items()
                                 if not k.startswith("_")}
                        # Preserve adjustments from Supabase for matching IDs;
                        # new contexts keep their (empty) adjustments.
                        clean["adjustments"] = latest_adj.get(
                            clean["id"], clean.get("adjustments", {})
                        )
                        new_cfg["contexts"].append(clean)

                    ok = _save_cfg("scenario_weights", new_cfg)
                    if ok:
                        clear_scenario_weights_cache()
                        st.success(
                            "Saved. Updated context rules apply on the next trade query."
                        )
                    else:
                        st.error(
                            "Save failed — Supabase not configured or unreachable."
                        )
                except Exception as e:
                    st.error(f"Save error: {e}")

    with revert_col:
        if st.button("Revert", key="revert_prio", use_container_width=True):
            if _PRIORITY_STATE_KEY in st.session_state:
                del st.session_state[_PRIORITY_STATE_KEY]
            clear_scenario_weights_cache()
            st.rerun()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("Context Rules")
    st.caption(
        "Scenario family weights derived from market state.  "
        "Edits are saved to Supabase and every version is retained.  "
        "The local `scenario_weights.json` is the factory default."
    )

    cfg = load_scenario_weights_config()

    tab_weights, tab_conditions, tab_priority = st.tabs(
        ["Context weights", "Context selection (read)", "Context selection (write)"]
    )

    with tab_weights:
        _render_context_weights(cfg)

    with tab_conditions:
        _render_choosing_a_context(cfg)

    with tab_priority:
        _render_priority_conditions(cfg)
