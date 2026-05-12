"""
Scenario Weightings page — edit explicit scenario-grid multipliers.
"""

from __future__ import annotations

import copy
import uuid

import pandas as pd
import streamlit as st

from analytics.scenario_generator import GRID_COLS, GRID_ROWS, VALID_GRID_CELLS, cell_id
from interface.security import assert_admin, current_user_email, is_admin_user
from knowledge_engine.scenario_weighter import (
    _FIELD_GETTERS,
    _OPS,
    clear_scenario_weights_cache,
    get_scenario_weights_source,
    load_scenario_weights_config,
)

_COND_FIELDS = [
    "carry_regime",
    "with_carry",
    "T",
    "vol",
    "target_z_abs",
    "atmfsratio",
    "primary_objective",
    "trade_management",
]
_COND_OPS = ["==", "!=", ">", ">=", "<", "<=", "in"]
_PRIM_OPTS = [
    "Balanced",
    "Keep cost low",
    "Hold up if the path is slow/noisy",
    "Keep risk clean",
]
_MGMT_OPTS = [
    "Standard hold",
    "May monetise early",
    "Need defendable mark-to-market",
]
_PRIORITY_STATE_KEY = "ctx_priority_edit"

_FIELD_LABELS = {
    "target_z_abs": "Target distance (σ)",
    "carry_regime": "Carry regime",
    "with_carry": "Direction vs carry",
    "T": "Tenor (years)",
    "vol": "ATM vol",
    "atmfsratio": "ATM/FS ratio",
    "primary_objective": "Primary objective",
    "trade_management": "Trade management",
}


def _valid_cell_ids() -> list[str]:
    return [cell_id(r, c) for r in GRID_ROWS for c in VALID_GRID_CELLS[r]]


def _ctx_label(ctx: dict) -> str:
    return ctx["id"].replace("_", " ")


def _fmt_conditions(when: list[dict]) -> str:
    if not when:
        return "Always"
    return "  AND  ".join(
        f"{_FIELD_LABELS.get(c['field'], c['field'])} {c['op']} {c['value']}"
        for c in when
    )


def _grid_df(ctx: dict, baseline: float) -> pd.DataFrame:
    multipliers = ctx.get("multipliers", {})
    rows = []
    for row in GRID_ROWS:
        item = {"Row": row}
        for col in GRID_COLS:
            cid = cell_id(row, col)
            item[col] = multipliers.get(cid, baseline) if col in VALID_GRID_CELLS[row] else None
        rows.append(item)
    return pd.DataFrame(rows).set_index("Row")


def _compact_multipliers(multipliers: dict[str, float], baseline: float) -> dict[str, float]:
    return {
        cid: float(value)
        for cid, value in multipliers.items()
        if abs(float(value) - float(baseline)) > 1e-9
    }


def _render_grid_editor(ctx: dict, baseline: float, min_multiplier: float) -> None:
    st.caption(
        f"Baseline multiplier **{baseline:.2f}**. Valid cells are editable with minimum "
        f"**{min_multiplier:.1f}**. Invalid cells are fixed as `-`. "
        "Leaving a cell at the baseline means that scenario keeps its default importance."
    )
    multipliers = copy.deepcopy(ctx.get("multipliers", {}))
    header_cols = st.columns([1] + [1] * len(GRID_COLS))
    header_cols[0].markdown("**Row**")
    for i, col in enumerate(GRID_COLS, start=1):
        header_cols[i].markdown(f"**{col}**")

    for row in GRID_ROWS:
        cols = st.columns([1] + [1] * len(GRID_COLS))
        cols[0].markdown(f"`{row}`")
        for i, col in enumerate(GRID_COLS, start=1):
            cid = cell_id(row, col)
            if col not in VALID_GRID_CELLS[row]:
                cols[i].markdown("-")
                continue
            val = float(multipliers.get(cid, baseline))
            multipliers[cid] = cols[i].number_input(
                f"{row}-{col}",
                min_value=float(min_multiplier),
                value=val,
                step=0.1,
                format="%.1f",
                key=f"grid_{ctx['id']}_{cid}",
                label_visibility="collapsed",
            )
    ctx["multipliers"] = _compact_multipliers(multipliers, baseline)


def _render_context_weights(cfg: dict) -> None:
    contexts = cfg["weightings"]
    labels = [_ctx_label(c) for c in contexts]
    selected = st.selectbox("Weighting", labels, key="ctx_grid_select")
    ctx = next(c for c in contexts if _ctx_label(c) == selected)
    overrides = {
        cid: val for cid, val in ctx.get("multipliers", {}).items()
        if abs(float(val) - float(cfg["baseline"])) > 1e-9
    }
    st.markdown(f"**Reasoning:** {ctx.get('comment', '')}")
    st.markdown(f"**Fires when:** {_fmt_conditions(ctx.get('when', []))}")
    st.caption(f"Explicit overrides in this weighting: **{len(overrides)}**")
    _render_grid_editor(ctx, cfg["baseline"], cfg["min_multiplier"])

    if st.button("Save multipliers", type="primary", use_container_width=True):
        try:
            from interface.supabase_logger import save_config as _save
            ok = _save(
                "scenario_definitions",
                cfg,
                _admin=is_admin_user(),
                user_email=current_user_email(),
            )
            if ok:
                clear_scenario_weights_cache()
                st.success("Saved. New scenario-grid multipliers apply on the next trade query.")
            else:
                st.error("Save failed — Supabase not configured or unreachable.")
        except Exception as e:
            st.error(f"Save error: {e}")


def _render_choosing_a_context(cfg: dict) -> None:
    rows = []
    for ctx in cfg["weightings"]:
        overrides = sum(
            1
            for cid, val in ctx.get("multipliers", {}).items()
            if cid in _valid_cell_ids() and abs(float(val) - float(cfg["baseline"])) > 1e-9
        )
        rows.append({
            "Weighting": _ctx_label(ctx),
            "Fires when": _fmt_conditions(ctx.get("when", [])),
            "Overrides": overrides,
            "Reasoning": ctx.get("comment", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _init_priority_state(cfg: dict) -> list[dict]:
    result = []
    for ctx in cfg["weightings"]:
        c = copy.deepcopy(ctx)
        c["_uid"] = str(uuid.uuid4())
        c["_original_id"] = ctx.get("id", "")
        result.append(c)
    return result


def _merge_contexts_with_latest_multipliers(
    contexts: list[dict],
    latest_cfg: dict,
) -> list[dict]:
    latest_mult = {
        c["id"]: copy.deepcopy(c.get("multipliers", {}))
        for c in latest_cfg.get("weightings", [])
    }
    merged: list[dict] = []
    for ctx in contexts:
        source_id = str(ctx.get("_original_id") or ctx.get("id") or "")
        clean = {k: copy.deepcopy(v) for k, v in ctx.items() if not k.startswith("_")}
        clean["multipliers"] = latest_mult.get(source_id, clean.get("multipliers", {}))
        merged.append(clean)
    return merged


def _parse_condition_value(s: str):
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _format_value_for_editor(value) -> str:
    if isinstance(value, list):
        return ", ".join(_format_value_for_editor(v) for v in value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _conditions_to_df(when: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {"field": c["field"], "op": c["op"], "value": _format_value_for_editor(c["value"])}
        for c in when
    ], columns=["field", "op", "value"])


def _df_to_conditions(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    conds: list[dict] = []
    errors: list[str] = []
    for idx, row in df.iterrows():
        field = str(row.get("field", "") or "").strip()
        op = str(row.get("op", "") or "").strip()
        val_s = str(row.get("value", "") or "").strip()
        if not field and not op and not val_s:
            continue
        if field not in _COND_FIELDS:
            errors.append(f"Row {idx + 1}: unknown field '{field}'")
            continue
        if op not in _COND_OPS:
            errors.append(f"Row {idx + 1}: unknown op '{op}'")
            continue
        if not val_s:
            errors.append(f"Row {idx + 1}: missing value")
            continue
        value = [_parse_condition_value(p.strip()) for p in val_s.split(",")] if op == "in" else _parse_condition_value(val_s)
        conds.append({"field": field, "op": op, "value": value})
    return conds, errors


def _simulate_context_fire(contexts: list[dict], field_values: dict) -> str | None:
    for ctx in contexts:
        match = True
        for cond in ctx.get("when", []):
            actual = _FIELD_GETTERS[cond["field"]](field_values["ms"], field_values["prefs"])
            if actual is None or not _OPS[cond["op"]](actual, cond["value"]):
                match = False
                break
        if match:
            return ctx["id"]
    return None


def _validate_contexts(contexts: list[dict]) -> list[str]:
    errors: list[str] = []
    ids = [str(ctx.get("id", "")).strip() for ctx in contexts]
    for i, ctx_id in enumerate(ids):
        if not ctx_id:
            errors.append(f"Weighting #{i + 1} has an empty ID.")
    seen: set[str] = set()
    for ctx_id in ids:
        if ctx_id and ctx_id in seen:
            errors.append(f"Duplicate weighting ID: '{ctx_id}'.")
        seen.add(ctx_id)
    return errors


def _render_priority_conditions(cfg: dict) -> None:
    if _PRIORITY_STATE_KEY not in st.session_state:
        st.session_state[_PRIORITY_STATE_KEY] = _init_priority_state(cfg)
    contexts: list[dict] = st.session_state[_PRIORITY_STATE_KEY]

    st.subheader("Live preview")
    p1 = st.columns(6)
    with p1[0]:
        prev_carry = st.selectbox("Carry regime", [0, 1, 2], index=1, key="pv_carry")
    with p1[1]:
        prev_wc = st.selectbox("With carry", [True, False], index=0, format_func=lambda x: "with carry" if x else "counter carry", key="pv_wc")
    with p1[2]:
        prev_T = st.number_input("Tenor (yrs)", min_value=0.01, max_value=3.0, value=0.25, step=0.01, format="%.2f", key="pv_T")
    with p1[3]:
        prev_vol = st.number_input("ATM vol", min_value=0.01, max_value=1.0, value=0.12, step=0.01, format="%.2f", key="pv_vol")
    with p1[4]:
        prev_tz = st.number_input("Target |σ|", min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.2f", key="pv_tz")
    with p1[5]:
        prev_atmfs = st.number_input("ATM/FS ratio", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f", key="pv_atmfs")
    p2 = st.columns(2)
    with p2[0]:
        prev_prim = st.selectbox("Primary objective", _PRIM_OPTS, index=0, key="pv_prim")
    with p2[1]:
        prev_mgmt = st.selectbox("Trade management", _MGMT_OPTS, index=0, key="pv_mgmt")

    class _PreviewMS:
        carry_regime = prev_carry
        with_carry = prev_wc
        T = prev_T
        vol = prev_vol
        target_z = prev_tz
        atmfsratio = prev_atmfs

    fired = _simulate_context_fire(contexts, {
        "ms": _PreviewMS,
        "prefs": {"primary_objective": prev_prim, "trade_management": prev_mgmt},
    })
    st.success(f"Fires: **{fired}**") if fired else st.info("No weighting matches → baseline grid")

    st.divider()
    for i, ctx in enumerate(contexts):
        uid = ctx.get("_uid", str(i))
        with st.expander(f"**#{i + 1}** — {ctx.get('id', '(unnamed)')}"):
            b = st.columns([1, 1, 1, 5])
            if b[0].button("▲ Up", key=f"up_{uid}", disabled=(i == 0), use_container_width=True):
                contexts[i - 1], contexts[i] = contexts[i], contexts[i - 1]
                st.rerun()
            if b[1].button("▼ Down", key=f"dn_{uid}", disabled=(i == len(contexts) - 1), use_container_width=True):
                contexts[i], contexts[i + 1] = contexts[i + 1], contexts[i]
                st.rerun()
            if b[2].button("Delete", key=f"del_{uid}", use_container_width=True):
                contexts.pop(i)
                st.rerun()
            id_col, cmt_col = st.columns([1, 2])
            ctx["id"] = id_col.text_input("Weighting ID", value=ctx.get("id", ""), key=f"cid_{uid}")
            ctx["comment"] = cmt_col.text_area("Reasoning / comment", value=ctx.get("comment", ""), key=f"cmt_{uid}", height=110)
            cond_df = _conditions_to_df(ctx.get("when", []))
            edited_df = st.data_editor(
                cond_df, key=f"cond_{uid}", use_container_width=True, num_rows="dynamic",
                column_config={
                    "field": st.column_config.SelectboxColumn("Field", options=_COND_FIELDS, required=True),
                    "op": st.column_config.SelectboxColumn("Op", options=_COND_OPS, required=True),
                    "value": st.column_config.TextColumn("Value"),
                },
            )
            new_conds, parse_errs = _df_to_conditions(edited_df)
            for err in parse_errs:
                st.caption(f"⚠️ {err}")
            ctx["when"] = new_conds

    if st.button("➕ Add weighting", key="add_ctx_btn"):
        contexts.append({
            "_uid": str(uuid.uuid4()),
            "_original_id": "",
            "id": f"new_weighting_{len(contexts) + 1}",
            "comment": "",
            "when": [],
            "multipliers": {},
        })
        st.rerun()

    save_col, revert_col, _ = st.columns([1, 1, 4])
    if save_col.button("Save changes", type="primary", key="save_prio", use_container_width=True):
        errors = _validate_contexts(contexts)
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                from interface.supabase_logger import save_config as _save_cfg
                clear_scenario_weights_cache()
                latest_cfg = load_scenario_weights_config()
                new_cfg = copy.deepcopy(latest_cfg)
                new_cfg["weightings"] = _merge_contexts_with_latest_multipliers(contexts, latest_cfg)
                ok = _save_cfg(
                    "scenario_definitions",
                    new_cfg,
                    _admin=is_admin_user(),
                    user_email=current_user_email(),
                )
                if ok:
                    clear_scenario_weights_cache()
                    st.success("Saved. Updated scenario weighting rules apply on the next trade query.")
                else:
                    st.error("Save failed — Supabase not configured or unreachable.")
            except Exception as e:
                st.error(f"Save error: {e}")
    if revert_col.button("Revert", key="revert_prio", use_container_width=True):
        st.session_state.pop(_PRIORITY_STATE_KEY, None)
        clear_scenario_weights_cache()
        st.rerun()


def render() -> None:
    assert_admin()
    st.header("Scenario Weightings")
    st.caption(
        "Explicit scenario-grid multipliers derived from market state. "
        "Edits are saved to Supabase and every version is retained. "
        "A multiplier of 1.0 means baseline importance for that scenario cell."
    )
    cfg = load_scenario_weights_config()
    st.caption(f"Loaded from: `{get_scenario_weights_source()}`")
    tab_weights, tab_conditions, tab_priority = st.tabs(
        ["Scenario grid", "Weighting selection (read)", "Weighting selection (write)"]
    )
    with tab_weights:
        _render_context_weights(cfg)
    with tab_conditions:
        _render_choosing_a_context(cfg)
    with tab_priority:
        _render_priority_conditions(cfg)
