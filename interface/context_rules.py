"""
Context Rules page — read-only view of scenario_weights.json.

Two tabs:
  1. Context weights  — matrix of contexts × families, showing the relative
     weight multiplier each context applies (1.00 = no change, >1 = bumped
     up, <1 = reduced). Same at-a-glance format as the affinity scores panel.

  2. Choosing a context — for each context, the conditions that must hold for
     it to fire, formatted in plain English with the full reasoning comment.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.scenario_generator import FAMILIES
from knowledge_engine.scenario_weighter import load_scenario_weights_config


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

_VALUE_LABELS = {
    # carry_regime ints
    0: "0 (noisy)",
    1: "1 (potential)",
    2: "2 (high)",
    # with_carry bools
    True:  "with carry",
    False: "counter carry",
}


def _fmt_value(field: str, value) -> str:
    if field == "T":
        # Express as months for readability
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
# Tab 1 — Context weights matrix
# ---------------------------------------------------------------------------

def _render_context_weights(cfg: dict) -> None:
    st.caption(
        "Each cell shows the relative weight multiplier applied to a scenario family "
        "when this context is active.  "
        "**1.00** = no change (baseline).  "
        "**> 1** = bumped up (this family matters more in this context).  "
        "**< 1** = reduced (less emphasis).  "
        "All weights are re-normalised after adjustment so the full vector sums to 1."
    )

    contexts = cfg["contexts"]

    # Build the matrix: rows = contexts, columns = families
    fam_labels = [f.replace("_", " ").title() for f in FAMILIES]
    rows = []
    for ctx in contexts:
        adj = ctx["adjustments"]
        row = {"Context": ctx["id"].replace("_", " ")}
        for fam in FAMILIES:
            delta = adj.get(fam, 0.0)
            row[fam.replace("_", " ").title()] = round(1.0 + delta, 3)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Context")

    def _color(val: float) -> str:
        if val > 1.0:
            # Green — intensity proportional to delta
            intensity = min(int((val - 1.0) * 1000), 120)
            return f"background-color: rgba(0, 180, 80, {intensity / 255:.2f})"
        if val < 1.0:
            intensity = min(int((1.0 - val) * 1000), 120)
            return f"background-color: rgba(220, 50, 50, {intensity / 255:.2f})"
        return ""

    styled = (
        df.style
        .map(_color)
        .format("{:.2f}")
    )
    st.dataframe(styled, use_container_width=True)

    # Legend row
    st.caption(
        f"Baseline weight before any context fires: "
        f"**{cfg['baseline']:.3f}** (1/{len(FAMILIES)}).  "
        f"Floor per family after normalisation: **{cfg['floor']:.2f}**."
    )


# ---------------------------------------------------------------------------
# Tab 2 — Choosing a context
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
        "Read-only view of `knowledge/defaults/scenario_weights.json`.  "
        "To change weights, edit that file — no code changes needed."
    )

    cfg = load_scenario_weights_config()

    tab_weights, tab_conditions = st.tabs(["Context weights", "Choosing a context"])

    with tab_weights:
        _render_context_weights(cfg)

    with tab_conditions:
        _render_choosing_a_context(cfg)
