"""
Context Rules page — view and edit scenario_weights.json contexts.

Two tabs:
  1. Context weights  — editable matrix of contexts × families.
     Each cell shows the relative weight multiplier (1.00 = no change,
     >1 = bumped up, <1 = reduced). Changes are saved to Supabase and
     every version is stored. The weighter picks up the latest version
     on the next trade query.

  2. Choosing a context — read-only view of the conditions that cause
     each context to fire, formatted in plain English.
"""

from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from analytics.scenario_generator import FAMILIES
from knowledge_engine.scenario_weighter import (
    clear_scenario_weights_cache,
    load_scenario_weights_config,
)


# ---------------------------------------------------------------------------
# Helpers
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
    """Reconstruct the full config dict from an edited matrix."""
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


def _color(val: float) -> str:
    if val > 1.0:
        intensity = min(int((val - 1.0) * 1000), 120)
        return f"background-color: rgba(0, 180, 80, {intensity / 255:.2f})"
    if val < 1.0:
        intensity = min(int((1.0 - val) * 1000), 120)
        return f"background-color: rgba(220, 50, 50, {intensity / 255:.2f})"
    return ""


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

    # Colour preview of the edited state
    styled = edited_df.style.map(_color).format("{:.2f}")
    st.caption("Preview (colour only — not editable):")
    st.dataframe(styled, use_container_width=True)

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

    # Version history
    st.divider()
    with st.expander("Version history", expanded=False):
        try:
            from interface.supabase_logger import fetch_config_history as _hist
            history = _hist("scenario_weights")
        except Exception:
            history = []

        if not history:
            st.caption("No saved versions yet (or Supabase not configured).")
        else:
            st.caption(f"{len(history)} version(s) stored, newest first.")
            for i, entry in enumerate(history):
                saved_at = entry.get("saved_at", "unknown time")
                v_cfg = entry.get("value", {})
                # Summarise non-default adjustments for this version
                non_default = []
                for ctx in v_cfg.get("contexts", []):
                    adj = ctx.get("adjustments", {})
                    if adj:
                        summary = ", ".join(
                            f"{f.replace('_',' ').title()} {d:+.2f}"
                            for f, d in adj.items()
                        )
                        non_default.append(f"{ctx['id'].replace('_',' ')}: {summary}")
                label = f"v{len(history) - i} — {saved_at}"
                with st.expander(label, expanded=False):
                    if non_default:
                        for line in non_default:
                            st.caption(line)
                    else:
                        st.caption("All families at baseline (no adjustments).")


# ---------------------------------------------------------------------------
# Tab 2 — Choosing a context (read-only)
# ---------------------------------------------------------------------------

def _render_choosing_a_context(cfg: dict) -> None:
    st.caption(
        "Each context fires when **all** of its conditions are satisfied by the current "
        "MarketState (which is already view-conditioned — it reflects both the market "
        "snapshot and the trade direction/target/tenor you supplied). "
        "Multiple contexts can fire simultaneously; their adjustments stack additively."
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

    tab_weights, tab_conditions = st.tabs(["Context weights", "Choosing a context"])

    with tab_weights:
        _render_context_weights(cfg)

    with tab_conditions:
        _render_choosing_a_context(cfg)
