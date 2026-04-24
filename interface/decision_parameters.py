"""
Decision Parameters Editor — expert workshop UI.

Run standalone:
    streamlit run interface/decision_parameters.py --server.port 8503

Loads affinity_scores.json, presents editable grids (one tab per scoring
dimension), colour-coded by score value. Exports updated JSON for copy-paste
back into the file.
"""

import copy
import json
import pathlib

import numpy as np
import pandas as pd
import streamlit as st

_SCORES_PATH = pathlib.Path(__file__).parent.parent / "knowledge" / "defaults" / "affinity_scores.json"

_DIMS = ["target_z_abs", "carry_regime", "atmfsratio", "carry_alignment"]

_DIM_LABELS = {
    "target_z_abs":    "Target Z (distance from forward)",
    "carry_regime":    "Carry Regime",
    "atmfsratio":      "ATM/F-S Ratio",
    "carry_alignment": "Carry Alignment",
}

_BUCKET_ORDER = {
    "target_z_abs":    ["no_target", "near", "moderate", "extended", "far"],
    "carry_regime":    ["0", "1", "2"],
    "atmfsratio":      ["low", "medium", "high"],
    "carry_alignment": ["with_low", "with_medium", "with_high",
                        "counter_low", "counter_medium", "counter_high"],
}

_STRUCT_ORDER = ["vanilla", "risk_reversal", "1x1_spread", "seagull", "1x2_spread",
                 "rko", "european_digital", "european_digital_rko"]

_SCORE_MIN, _SCORE_MAX = -3, 3


def _load() -> dict:
    try:
        from interface.supabase_logger import fetch_config
        data = fetch_config("affinity_scores")
        if data:
            return data
    except Exception:
        pass
    with open(_SCORES_PATH) as f:
        return json.load(f)


def _color_score(val):
    """Background colour for score cells."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        intensity = min(int(v / _SCORE_MAX * 180), 180)
        return f"background-color: rgb({255-intensity}, 255, {255-intensity})"
    elif v < 0:
        intensity = min(int(abs(v) / abs(_SCORE_MIN) * 180), 180)
        return f"background-color: rgb(255, {255-intensity}, {255-intensity})"
    return "background-color: #f5f5f5"


def _build_df(scores_cfg: dict, dim: str) -> pd.DataFrame:
    buckets = _BUCKET_ORDER[dim]
    structs = [s for s in _STRUCT_ORDER if s in scores_cfg["structures"]]
    data = {}
    for bucket in buckets:
        data[bucket] = [
            scores_cfg["structures"][s].get(dim, {}).get(bucket, 0)
            for s in structs
        ]
    return pd.DataFrame(data, index=structs)


def _write_df_back(scores_cfg: dict, dim: str, df: pd.DataFrame) -> None:
    for struct in df.index:
        for bucket in df.columns:
            val = df.loc[struct, bucket]
            scores_cfg["structures"][struct].setdefault(dim, {})[bucket] = float(val)


def _render_gates(scores_cfg: dict) -> dict:
    st.subheader("Gates (hard filters)")
    st.caption("A structure is excluded entirely if the market state fails its gate. Leave blank = no gate.")

    structs = [s for s in _STRUCT_ORDER if s in scores_cfg["structures"]]
    gate_keys = ["target_z_abs_min", "target_z_abs_max"]

    rows = []
    for s in structs:
        gates = scores_cfg["structures"][s].get("gates", {})
        rows.append({
            "structure": s,
            "target_z_abs_min": gates.get("target_z_abs_min", None),
            "target_z_abs_max": gates.get("target_z_abs_max", None),
        })

    df = pd.DataFrame(rows).set_index("structure")
    edited = st.data_editor(
        df,
        use_container_width=True,
        column_config={
            "target_z_abs_min": st.column_config.NumberColumn("Min |target_z| (σ)", min_value=0.0, max_value=5.0, step=0.25, format="%.2f"),
            "target_z_abs_max": st.column_config.NumberColumn("Max |target_z| (σ)", min_value=0.0, max_value=5.0, step=0.25, format="%.2f"),
        },
        key="gates_editor",
    )
    return edited


def _render_thresholds(scores_cfg: dict) -> None:
    st.subheader("Bucket thresholds")
    st.caption("Cut-points that define which bucket a market state falls into.")

    col1, col2, col3 = st.columns(3)
    tz = scores_cfg["thresholds"]["target_z_abs"]
    cr = scores_cfg["thresholds"]["carry_regime"]
    atm = scores_cfg["thresholds"]["atmfsratio"]

    with col1:
        st.markdown("**Target Z buckets** (σ from forward)")
        tz0 = st.number_input("near / moderate boundary", value=float(tz[0]), step=0.1, format="%.2f", key="tz0")
        tz1 = st.number_input("moderate / extended boundary", value=float(tz[1]), step=0.1, format="%.2f", key="tz1")
        tz2 = st.number_input("extended / far boundary", value=float(tz[2]), step=0.1, format="%.2f", key="tz2")
        scores_cfg["thresholds"]["target_z_abs"] = [tz0, tz1, tz2]

    with col2:
        st.markdown("**Carry regime buckets** (|c| = normalised carry)")
        cr0 = st.number_input("noisy / potential boundary", value=float(cr[0]), step=0.05, format="%.2f", key="cr0")
        cr1 = st.number_input("potential / high boundary", value=float(cr[1]), step=0.05, format="%.2f", key="cr1")
        scores_cfg["thresholds"]["carry_regime"] = [cr0, cr1]

    with col3:
        st.markdown("**ATM/F-S ratio buckets**")
        atm0 = st.number_input("low / medium boundary", value=float(atm[0]), step=0.1, format="%.2f", key="atm0")
        atm1 = st.number_input("medium / high boundary", value=float(atm[1]), step=0.1, format="%.2f", key="atm1")
        scores_cfg["thresholds"]["atmfsratio"] = [atm0, atm1]


def _render_dim(scores_cfg: dict, dim: str) -> None:
    st.markdown(f"**Scores: {_DIM_LABELS[dim]}**")
    st.caption(f"Rows = structures · Columns = market condition buckets · Scale: {_SCORE_MIN} (penalise) → 0 (neutral) → {_SCORE_MAX} (strongly prefer)")

    df = _build_df(scores_cfg, dim)

    col_cfg = {
        b: st.column_config.NumberColumn(b, min_value=_SCORE_MIN, max_value=_SCORE_MAX, step=0.5, format="%.1f")
        for b in df.columns
    }

    edited = st.data_editor(
        df,
        use_container_width=True,
        column_config=col_cfg,
        key=f"editor_{dim}",
    )

    # Write edits back into working copy
    _write_df_back(scores_cfg, dim, edited)

    # Colour preview (read-only styled view below the editor)
    st.caption("Colour preview")
    st.dataframe(
        edited.style.map(_color_score),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Main render (called from app.py)
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Decision Parameters Editor")
    st.caption("Scores are loaded from Supabase (falls back to local file). Save pushes to Supabase and takes effect on the next query.")

    if "scores_cfg" not in st.session_state:
        st.session_state.scores_cfg = _load()

    working = copy.deepcopy(st.session_state.scores_cfg)

    tab_tz, tab_cr, tab_atm, tab_ca, tab_gates, tab_thresh = st.tabs([
        "Target Z",
        "Carry Regime",
        "ATM/F-S Ratio",
        "Carry Alignment",
        "Gates",
        "Thresholds",
    ])

    with tab_tz:
        _render_dim(working, "target_z_abs")

    with tab_cr:
        _render_dim(working, "carry_regime")

    with tab_atm:
        _render_dim(working, "atmfsratio")

    with tab_ca:
        _render_dim(working, "carry_alignment")

    with tab_gates:
        gates_df = _render_gates(working)
        for struct in gates_df.index:
            gates = {}
            for k in ["target_z_abs_min", "target_z_abs_max"]:
                v = gates_df.loc[struct, k]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    gates[k] = float(v)
            working["structures"][struct]["gates"] = gates

    with tab_thresh:
        _render_thresholds(working)

    st.divider()
    col_l, col_r = st.columns([3, 1])

    original = _load()
    out = {k: v for k, v in original.items() if k.startswith("_")}
    out.update({k: v for k, v in working.items() if not k.startswith("_")})

    with col_r:
        if st.button("Save", type="primary", use_container_width=True):
            st.session_state.scores_cfg = copy.deepcopy(working)
            from knowledge_engine.loader import clear_affinity_scores_cache
            from interface.supabase_logger import save_config, init_status
            sb_ok, _ = init_status()
            if sb_ok and save_config("affinity_scores", out):
                clear_affinity_scores_cache()
                st.success("Saved to Supabase — next query will use updated scores.")
            else:
                try:
                    with open(_SCORES_PATH, "w") as f:
                        json.dump(out, f, indent=2)
                        f.write("\n")
                    clear_affinity_scores_cache()
                    st.warning("Supabase unavailable — saved to local file only.")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with col_l:
        st.code(json.dumps(out, indent=2), language="json")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__" or not hasattr(st, "_is_running_with_streamlit"):
    st.set_page_config(page_title="Decision Parameters Editor", layout="wide")
    render()
