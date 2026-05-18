"""
MacroTool — EM FX distribution view tool.

The current Trade View UI uses structured inputs and runs the deterministic
engine path directly. The conversational LLM flow remains in the codebase as
the target architecture, but it is intentionally silent on this UI path while
we test the pipes.

Run with:
    .venv/bin/streamlit run interface/app.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from conversation.flow import ConversationFlow, target_from_reference
from interface.charts import build_distribution_fan, build_maturity_histogram
from interface.security import current_user_email, is_admin_user, require_login
from interface.llm_config import (
    get_llm_provider,
    get_provider_api_key,
    get_provider_model,
    provider_label,
    get_gemini_vertex_credentials,
    gemini_status,
)
from interface.structure_eval import (
    LINEAR_NOTIONAL,
    fmt_ccy,
    fmt_ccy_label,
    variant_label_with_strikes,
    target_price,
    render_structure_variants,
    render_structure_evaluation,
)
from interface.advisor_chat import render_advisor_chat
from knowledge_engine.structure_scorer import get_scoring_detail
from knowledge_engine.models import TradeView
from analytics.distributions import interpolate_vol
from data.snapshot_loader import load_snapshot
from data.snapshot_overrides import apply_overrides
from interface.debug_log import (
    log_prompt, log_view_extracted,
    log_market_state, log_scorer_result, log_error,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MacroTool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Secrets → os.environ — must run before session state so ConversationFlow
# (which creates SessionTrace) sees the Langfuse keys
# ---------------------------------------------------------------------------

def _inject_secrets() -> None:
    _secret_keys = [
        "LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_KEY",
    ]
    try:
        for k in _secret_keys:
            if k in st.secrets and k not in os.environ:
                os.environ[k] = st.secrets[k]
    except Exception:
        pass

_inject_secrets()
require_login()
USER_EMAIL = current_user_email()
IS_ADMIN = is_admin_user()

from conversation import tracing as _tracing
_tracing._init_client()

from interface.supabase_logger import log_query as _log_query, log_feedback as _log_feedback, reinit as _sb_reinit, init_status as _sb_status
_sb_reinit()
from knowledge_engine.loader import load_structure_profiles as _lsp
_lsp.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_effective_snapshot():
    base = load_snapshot()
    overrides = st.session_state.get("market_overrides", {})
    return apply_overrides(base, overrides) if overrides else base


def _make_flow() -> ConversationFlow:
    provider = get_llm_provider()
    return ConversationFlow(
        api_key=get_provider_api_key(provider),
        snapshot=_get_effective_snapshot(),
        provider=provider,
        model=get_provider_model(provider),
        credentials=get_gemini_vertex_credentials() if provider == "gemini" else None,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "flow" not in st.session_state:
    st.session_state.flow = _make_flow()
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "page" not in st.session_state:
    st.session_state.page = "Trade View"
if "target_rr" not in st.session_state:
    st.session_state.target_rr = 3.0
if "clarification" not in st.session_state:
    st.session_state.clarification = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pref_primary_objective" not in st.session_state:
    st.session_state.pref_primary_objective = "Balanced"
if "pref_structure_constraint" not in st.session_state:
    st.session_state.pref_structure_constraint = "No restriction"
if "pref_trade_management" not in st.session_state:
    st.session_state.pref_trade_management = "Standard hold"
if "market_edit_mode" not in st.session_state:
    st.session_state.market_edit_mode = {}

flow: ConversationFlow = st.session_state.flow

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("MacroTool")
    try:
        from importlib.metadata import version as _pkg_version
        st.caption(f"EM FX trade structuring · v{_pkg_version('macrotool')}")
    except Exception:
        st.caption("EM FX trade structuring")
    st.caption(f"Signed in as {USER_EMAIL}")
    st.button("Sign out", on_click=st.logout, use_container_width=True)
    st.divider()

    nav_labels = ("Trade View", "Market Data", "Structure Selection", "Scenario Weightings", "Query log") if IS_ADMIN else ("Trade View",)
    for label in nav_labels:
        active = st.session_state.page == label
        if st.button(
            label,
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            if label == "Trade View":
                st.session_state.flow = _make_flow()
                st.session_state.submitted = False
                st.session_state.last_prompt = ""
                st.session_state.clarification = ""
                st.session_state.chat_history = []
            st.session_state.page = label
            st.rerun()

    st.divider()

    st.markdown("Risk / Reward target")
    with st.container(border=True):
        st.session_state.target_rr = st.slider(
            "Risk 1 to make",
            min_value=1.5,
            max_value=10.0,
            value=st.session_state.target_rr,
            step=0.5,
            format="%.1f×",
        )

    st.divider()

    active_provider = get_llm_provider()
    active_model = get_provider_model(active_provider)
    st.caption(f"LLM: {provider_label(active_provider)} · {active_model}")
    if active_provider == "gemini":
        gemini_ready, gemini_message = gemini_status()
        if gemini_ready:
            st.success(gemini_message)
            if st.button("Test LLM connection", use_container_width=True):
                with st.spinner("Calling LLM…"):
                    try:
                        _test_msgs = [{"role": "user", "content": "Reply with exactly: OK"}]
                        for _ in st.session_state.flow._client.stream(_test_msgs, system="You are a helpful assistant."):
                            pass
                        _test_resp = st.session_state.flow._client.last_response.strip()
                        st.success(f"LLM responded: {_test_resp!r}")
                    except Exception as _e:
                        st.error(f"LLM call failed: {_e}")
        else:
            st.error(gemini_message)
    elif get_provider_api_key(active_provider):
        st.success("API key ready")
    else:
        st.error(f"Server {provider_label(active_provider)} API key not configured.")

    sb_connected, sb_error = _sb_status()
    if sb_connected:
        st.success("Supabase connected")
    else:
        st.warning(f"Supabase: {sb_error}")


# ---------------------------------------------------------------------------
# Structured intake helpers
# ---------------------------------------------------------------------------

_HORIZON_OPTIONS: list[tuple[str, int]] = [
    (f"{month}M", round(month * 365 / 12)) for month in range(1, 13)
]
_DIRECTION_OPTIONS = {
    "Higher": "base_higher",
    "Lower": "base_lower",
}
_PRIMARY_OBJECTIVE_OPTIONS = [
    "Balanced",
    "Keep cost low",
    "Hold up if the path is slow/noisy",
    "Keep risk clean",
]
_STRUCTURE_CONSTRAINT_OPTIONS = [
    "No restriction",
    "Avoid capped structures",
    "Avoid complex structures",
    "Avoid tail-risky structures",
]
_TRADE_MANAGEMENT_OPTIONS = [
    "Standard hold",
    "May monetise early",
    "Need defendable mark-to-market",
]


def _build_prompt_summary(pair: str, direction: str, horizon_days: int, target: float) -> str:
    direction_label = "Long" if direction == "base_higher" else "Short"
    return f"{direction_label} {pair}, target {target:.4f}, {horizon_days}d"


def _submit_structured_view(pair: str, direction: str, horizon_days: int, target: float) -> str | None:
    direction_label = "base higher" if direction == "base_higher" else "base lower"
    prompt = f"pair={pair}; direction={direction_label}; target={target:.4f}; horizon_days={horizon_days}"
    log_prompt(prompt)

    ccy = flow._snapshot.get(pair)
    if ccy is None:
        return f"ERROR: Unsupported pair {pair}."
    from pricing.forwards import rate_context_for_snapshot
    horizon_years = horizon_days / 365.0
    rate_ctx = rate_context_for_snapshot(ccy, horizon_years)
    fwd = rate_ctx.forward

    if direction == "base_higher" and target <= fwd:
        return f"ERROR: For `Base higher`, target must be above the horizon forward ({fwd:.4f})."
    if direction == "base_lower" and target >= fwd:
        return f"ERROR: For `Base lower`, target must be below the horizon forward ({fwd:.4f})."

    magnitude_pct = abs(target / fwd - 1.0) * 100.0
    view = TradeView(
        pair=pair,
        direction=direction,
        direction_conviction="medium",
        horizon_days=horizon_days,
        magnitude_pct=magnitude_pct,
    )

    flow.view = view
    flow.ccy = ccy
    flow.structure_constraint = st.session_state.get(
        "pref_structure_constraint", "No restriction"
    )
    flow.primary_objective = st.session_state.get(
        "pref_primary_objective", "Balanced"
    )
    flow.trade_management = st.session_state.get(
        "pref_trade_management", "Standard hold"
    )
    try:
        flow._run_engines()
        log_view_extracted(view.__dict__)
        if flow.market_state:
            log_market_state(flow.market_state)
        if flow.selector_result:
            log_scorer_result(flow.selector_result)
    except Exception as e:
        log_error("_run_engines", e)
        raise

    if (
        flow.market_state
        and flow.market_state.target_z is not None
        and abs(flow.market_state.target_z) < 0.25
    ):
        flow.market_state = None
        flow.selector_result = None
        return ("ERROR: Target is less than 0.25σ from the forward — "
                "the move is not large enough to structure an option trade.")

    try:
        _log_query(
            prompt=prompt,
            pair=view.pair,
            direction=view.direction,
            magnitude_pct=view.magnitude_pct,
            horizon_days=view.horizon_days,
            target_z=flow.market_state.target_z if flow.market_state else None,
            carry_regime=flow.market_state.carry_regime if flow.market_state else None,
            top_structure=flow.selector_result.shortlist[0].structure_id if flow.selector_result and flow.selector_result.shortlist else None,
            llm_response="",
            user_email=USER_EMAIL,
        )
    except Exception as e:
        log_error("supabase_log_query", e)

    st.session_state.last_prompt = _build_prompt_summary(pair, direction, horizon_days, target)
    return None


# ---------------------------------------------------------------------------
# Market Data page helpers
# ---------------------------------------------------------------------------

_TENOR_ORDER = ["1W", "1M", "2M", "3M", "6M", "1Y"]
_DELTA_ORDER = ["10DP", "25DP", "ATM", "25DC", "10DC"]


def _render_query_log() -> None:
    from interface.supabase_logger import fetch_queries
    st.subheader("Query log")
    rows = fetch_queries(_admin=IS_ADMIN)
    if not rows:
        st.caption("No queries logged yet, or Supabase not connected.")
        return
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    df["direction"] = df["direction"].str.replace("_", " ")
    df["target_z"] = df["target_z"].apply(lambda x: f"{x:+.2f}σ" if x is not None else "—")
    df["carry_regime"] = df["carry_regime"].map({0: "0 noisy", 1: "1 potential", 2: "2 high"}).fillna("—")
    df = df.rename(columns={
        "created_at":    "Time",
        "user_email":    "User",
        "pair":          "Pair",
        "direction":     "Direction",
        "magnitude_pct": "Mag %",
        "horizon_days":  "Horizon",
        "target_z":      "Target z",
        "carry_regime":  "Carry regime",
        "top_structure": "Top structure",
        "prompt":        "Prompt",
    })
    display_cols = ["Time"]
    if "User" in df.columns:
        display_cols.append("User")
    display_cols.extend(["Pair", "Direction", "Mag %", "Horizon", "Target z", "Carry regime", "Top structure", "Prompt"])
    df = df[display_cols]
    st.dataframe(df, use_container_width=True, hide_index=True)


def _ordered_tenors(ccy) -> list[str]:
    return [t for t in _TENOR_ORDER if ccy.get_forward(t) is not None or ccy.get_atm_vol(t) is not None]


def _delta_pillars(ccy) -> list[str]:
    deltas = set()
    nodes = {(n.tenor, n.delta): n.vol for n in ccy.vol_surface}
    for delta in _DELTA_ORDER:
        if delta.endswith("DC"):
            pillar = delta[:-2]
            if any((tenor, f"{pillar}DC") in nodes and (tenor, f"{pillar}DP") in nodes for tenor in _ordered_tenors(ccy)):
                deltas.add(pillar)
    return sorted(deltas, key=lambda d: int(d))


def _surface_tables(ccy) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tenors = _ordered_tenors(ccy)
    pillars = _delta_pillars(ccy)

    atm_rows = [{"Tenor": tenor, "ATM": round(ccy.get_atm_vol(tenor) * 100, 2)} for tenor in tenors]
    rr_rows = []
    bf_rows = []
    for tenor in tenors:
        rr_row = {"Tenor": tenor}
        bf_row = {"Tenor": tenor}
        atm = ccy.get_atm_vol(tenor)
        for pillar in pillars:
            call = ccy.get_vol(tenor, f"{pillar}DC")
            put = ccy.get_vol(tenor, f"{pillar}DP")
            rr_row[f"{pillar}D RR"] = round((call - put) * 100, 2)
            bf_row[f"{pillar}D BF"] = round((0.5 * (call + put) - atm) * 100, 2)
        rr_rows.append(rr_row)
        bf_rows.append(bf_row)

    return pd.DataFrame(atm_rows), pd.DataFrame(rr_rows), pd.DataFrame(bf_rows)


def _forward_table(ccy) -> pd.DataFrame:
    rows = [
        {
            "Tenor": f.tenor,
            "Points": round(f.points, 6),
            "Outright": round(f.outright, 6),
        }
        for f in sorted(ccy.forwards, key=lambda x: _TENOR_ORDER.index(x.tenor))
    ]
    return pd.DataFrame(rows)


def _pair_modified_summary(pair_override: dict) -> str:
    parts = []
    if pair_override.get("forwards"):
        parts.append(f"forwards ({len(pair_override['forwards'])})")
    if pair_override.get("atm_vols"):
        parts.append(f"ATM vols ({len(pair_override['atm_vols'])})")
    if pair_override.get("risk_reversals"):
        parts.append(f"RR ({sum(len(v) for v in pair_override['risk_reversals'].values())})")
    if pair_override.get("butterflies"):
        parts.append(f"BF ({sum(len(v) for v in pair_override['butterflies'].values())})")
    return ", ".join(parts)


def _marked_value(value: float, modified: bool, fmt: str) -> str:
    rendered = format(value, fmt)
    return f"{rendered} *" if modified else rendered


def _display_forward_table(base_ccy, current_ccy, pair_override: dict) -> pd.DataFrame:
    modified = set(pair_override.get("forwards", {}))
    rows = []
    for base_fwd, current_fwd in zip(
        sorted(base_ccy.forwards, key=lambda x: _TENOR_ORDER.index(x.tenor)),
        sorted(current_ccy.forwards, key=lambda x: _TENOR_ORDER.index(x.tenor)),
    ):
        rows.append({
            "Tenor": current_fwd.tenor,
            "Points": _marked_value(current_fwd.points, current_fwd.tenor in modified, ".2f"),
            "Outright": _marked_value(current_fwd.outright, current_fwd.tenor in modified, ".4f"),
        })
    return pd.DataFrame(rows)


def _display_surface_tables(base_ccy, current_ccy, pair_override: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_atm, base_rr, base_bf = _surface_tables(base_ccy)
    cur_atm, cur_rr, cur_bf = _surface_tables(current_ccy)

    atm_modified = set(pair_override.get("atm_vols", {}))
    cur_atm["ATM"] = [
        _marked_value(val, tenor in atm_modified, ".2f")
        for tenor, val in zip(cur_atm["Tenor"], cur_atm["ATM"])
    ]

    rr_overrides = pair_override.get("risk_reversals", {})
    bf_overrides = pair_override.get("butterflies", {})

    for df, overrides in ((cur_rr, rr_overrides), (cur_bf, bf_overrides)):
        for col in df.columns:
            if col == "Tenor":
                continue
            pillar = re.match(r"(\d+)D", col).group(1)
            df[col] = [
                _marked_value(val, pillar in overrides.get(tenor, {}), ".2f")
                for tenor, val in zip(df["Tenor"], df[col])
            ]

    return cur_atm, cur_rr, cur_bf


def _collect_pair_overrides(base_ccy, edited_fwds: pd.DataFrame, edited_atm: pd.DataFrame, edited_rr: pd.DataFrame, edited_bf: pd.DataFrame) -> dict:
    overrides: dict[str, dict] = {}

    forward_overrides = {}
    base_fwd_map = {f.tenor: f.outright for f in base_ccy.forwards}
    for row in edited_fwds.to_dict("records"):
        tenor = row["Tenor"]
        outright = float(row["Outright"])
        if abs(outright - base_fwd_map[tenor]) > 1e-12:
            forward_overrides[tenor] = outright
    if forward_overrides:
        overrides["forwards"] = forward_overrides

    atm_overrides = {}
    base_atm_map = {tenor: base_ccy.get_atm_vol(tenor) for tenor in edited_atm["Tenor"]}
    for row in edited_atm.to_dict("records"):
        tenor = row["Tenor"]
        atm = float(row["ATM"]) / 100.0
        if abs(atm - base_atm_map[tenor]) > 1e-12:
            atm_overrides[tenor] = atm
    if atm_overrides:
        overrides["atm_vols"] = atm_overrides

    for section_name, edited_df in (("risk_reversals", edited_rr), ("butterflies", edited_bf)):
        section_overrides = {}
        for row in edited_df.to_dict("records"):
            tenor = row["Tenor"]
            tenor_overrides = {}
            for col, value in row.items():
                if col == "Tenor":
                    continue
                pillar = re.match(r"(\d+)D", col).group(1)
                call = base_ccy.get_vol(tenor, f"{pillar}DC")
                put = base_ccy.get_vol(tenor, f"{pillar}DP")
                atm = base_ccy.get_atm_vol(tenor)
                base_value = (call - put) * 100 if section_name == "risk_reversals" else (0.5 * (call + put) - atm) * 100
                if abs(float(value) - base_value) > 1e-12:
                    tenor_overrides[pillar] = float(value) / 100.0
            if tenor_overrides:
                section_overrides[tenor] = tenor_overrides
        if section_overrides:
            overrides[section_name] = section_overrides

    return overrides


def _render_market_data() -> None:
    base_snapshot = load_snapshot()
    overrides = st.session_state.get("market_overrides", {})
    snapshot = apply_overrides(base_snapshot, overrides) if overrides else base_snapshot
    st.subheader("Market Data")
    st.caption(
        f"Snapshot date: {snapshot.snapshot_date}  ·  {snapshot.data_note}"
    )
    if flow.view:
        st.warning("Edits apply to your next conversation. Click `↩ New view` to use them.")

    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("Reset all market data to base", use_container_width=True):
            st.session_state.pop("market_overrides", None)
            st.session_state.market_edit_mode = {}
            if not flow.view:
                st.session_state.flow = _make_flow()
            st.rerun()

    for pair, ccy in snapshot.currencies.items():
        base_ccy = base_snapshot.get(pair)
        pair_override = overrides.get(pair, {})
        is_editing = st.session_state.market_edit_mode.get(pair, False)
        header = f"{pair}  —  {ccy.instrument_type}  ·  spot {ccy.spot:.4f}"
        with st.expander(header, expanded=(pair == next(iter(snapshot.currencies)))):
            actions = st.columns([1, 1, 4])
            if pair_override:
                actions[2].caption(f"Modified locally: {_pair_modified_summary(pair_override)}")
            elif not is_editing:
                actions[2].caption("Using base market data.")

            if not is_editing:
                if actions[0].button("Edit", key=f"edit_{pair}", use_container_width=True):
                    st.session_state.market_edit_mode[pair] = True
                    st.rerun()
                if actions[1].button("Reset pair", key=f"reset_{pair}", use_container_width=True, disabled=not pair_override):
                    st.session_state.get("market_overrides", {}).pop(pair, None)
                    if not st.session_state.get("market_overrides"):
                        st.session_state.pop("market_overrides", None)
                    if not flow.view:
                        st.session_state.flow = _make_flow()
                    st.rerun()

                fwd_df = _display_forward_table(base_ccy, ccy, pair_override)
                atm_df, rr_df, bf_df = _display_surface_tables(base_ccy, ccy, pair_override)
                df_curve_map = {
                    "USD": getattr(ccy, "usd_df_curve", []),
                    "EUR": getattr(ccy, "eur_df_curve", []),
                    "GBP": getattr(ccy, "gbp_df_curve", []),
                }
                base_curve = df_curve_map.get(pair[:3], [])
                tenors = _ordered_tenors(ccy)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Forwards**")
                    st.dataframe(fwd_df, use_container_width=True, hide_index=True)
                    st.markdown("**ATM vols (vol points)**")
                    st.dataframe(atm_df, use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Risk reversals (vol points)**")
                    st.dataframe(rr_df, use_container_width=True, hide_index=True)
                    st.markdown("**Butterflies (vol points)**")
                    st.dataframe(bf_df, use_container_width=True, hide_index=True)

                if base_curve:
                    st.markdown("**Discount factors**")
                    st.caption("Locked — DF curves, spot, and instrument type are read-only in v1.")
                    df_map = {d.tenor: d.df for d in base_curve}
                    df_df = pd.DataFrame({f"{pair[:3]} DF": [df_map.get(t, "") for t in tenors]}, index=tenors)
                    df_df.index.name = "Tenor"
                    st.dataframe(df_df, use_container_width=True)
            else:
                st.caption("Editing locally only. Changes are validated and saved to this browser session; they apply on the next conversation.")
                edit_fwds = st.data_editor(
                    _forward_table(ccy),
                    key=f"fwd_editor_{pair}",
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Tenor", "Points"],
                )
                edit_atm, edit_rr, edit_bf = _surface_tables(ccy)
                edit_atm = st.data_editor(
                    edit_atm,
                    key=f"atm_editor_{pair}",
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Tenor"],
                )
                edit_rr = st.data_editor(
                    edit_rr,
                    key=f"rr_editor_{pair}",
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Tenor"],
                )
                edit_bf = st.data_editor(
                    edit_bf,
                    key=f"bf_editor_{pair}",
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Tenor"],
                )

                save_col, cancel_col, reset_col = st.columns([1, 1, 1])
                if save_col.button("Save locally", key=f"save_{pair}", use_container_width=True, type="primary"):
                    try:
                        pair_changes = _collect_pair_overrides(base_ccy, edit_fwds, edit_atm, edit_rr, edit_bf)
                        next_overrides = dict(st.session_state.get("market_overrides", {}))
                        if pair_changes:
                            next_overrides[pair] = pair_changes
                        else:
                            next_overrides.pop(pair, None)
                        if next_overrides:
                            apply_overrides(base_snapshot, next_overrides)
                            st.session_state["market_overrides"] = next_overrides
                        else:
                            st.session_state.pop("market_overrides", None)
                        st.session_state.market_edit_mode[pair] = False
                        if not flow.view:
                            st.session_state.flow = _make_flow()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not save local edits: {e}")
                if cancel_col.button("Cancel", key=f"cancel_{pair}", use_container_width=True):
                    st.session_state.market_edit_mode[pair] = False
                    st.rerun()
                if reset_col.button("Reset pair", key=f"edit_reset_{pair}", use_container_width=True, disabled=not pair_override):
                    st.session_state.get("market_overrides", {}).pop(pair, None)
                    if not st.session_state.get("market_overrides"):
                        st.session_state.pop("market_overrides", None)
                    st.session_state.market_edit_mode[pair] = False
                    if not flow.view:
                        st.session_state.flow = _make_flow()
                    st.rerun()


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if st.session_state.page != "Trade View" and not IS_ADMIN:
    st.session_state.page = "Trade View"
    st.rerun()

if st.session_state.page == "Market Data":
    _render_market_data()

elif st.session_state.page == "Query log":
    _render_query_log()

elif st.session_state.page == "Structure Selection":
    from interface.decision_parameters import render as _render_decision_params
    _render_decision_params()

elif st.session_state.page == "Scenario Weightings":
    from interface.context_rules import render as _render_context_rules
    _render_context_rules()

else:
    # ---- Trade View page ----

    _brief_path = Path(__file__).parent / "testing_brief.json"
    try:
        _brief = json.loads(_brief_path.read_text())
        with st.expander(f"Testing brief — {_brief.get('updated', '')}", expanded=not flow.view):
            st.markdown(f"**Focus:** {_brief['focus']}")
            col_try, col_skip = st.columns(2)
            with col_try:
                st.markdown("**Try these**")
                for item in _brief.get("try_these", []):
                    st.caption(f"• {item}")
            with col_skip:
                st.markdown("**Ignore for now**")
                for item in _brief.get("ignore_for_now", []):
                    st.caption(f"• {item}")
    except Exception:
        pass

    if flow.view and "last_prompt" in st.session_state and st.session_state.last_prompt:
        st.info(f"**View:** {st.session_state.last_prompt}")

    if flow.flat_distribution and flow.smile_distribution:
        _target = target_price(flow)

        col_fan, col_hist = st.columns(2)
        with col_fan:
            fig_fan = build_distribution_fan(flow.flat_distribution, flow.smile_distribution, _target)
            if fig_fan:
                st.plotly_chart(fig_fan, use_container_width=True)
        with col_hist:
            fig_hist = build_maturity_histogram(flow.flat_distribution, flow.smile_distribution, _target)
            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True)

        if _target:
            flat = flow.flat_distribution
            try:
                fwd = flat.terminal_median
                sigma_sqrtT = math.log(flat.terminal_plus1s / fwd)
                z = math.log(_target / fwd) / sigma_sqrtT
                move_from_fwd_pct = (_target / fwd - 1) * 100
                direction_word = "appreciation" if flow.view.direction == "base_higher" else "depreciation"
                st.markdown(
                    f"Target **{_target:.4f}** ({move_from_fwd_pct:+.1f}% from the {flow.view.horizon_days}d forward of **{fwd:.4f}**) "
                    f"represents **{z:+.1f}σ** — "
                    f"a {flow.view.magnitude_pct:.1f}% {direction_word} from spot."
                )
            except (ValueError, ZeroDivisionError):
                pass
    else:
        st.markdown("### Enter trade view")
        st.caption("Select the pair, direction, horizon, and target level.")

    # Structure recommendation
    if flow.market_state and flow.selector_result and flow.selector_result.shortlist:
        st.divider()

        ms = flow.market_state
        h = flow.view.horizon_days
        _is_call = flow.view.direction == "base_higher"
        _target = target_price(flow)
        st.subheader("Market state")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spot", f"{ms.spot:.4f}")
        c2.metric("Forward", f"{ms.fwd:.4f}")
        c3.metric("ATM Vol", f"{ms.vol:.1%}")
        c4.metric("Horizon", f"{h}d")

        c1, c2, c3, c4 = st.columns(4)
        regime_label = {0: "0 — noisy", 1: "1 — potential", 2: "2 — high carry"}
        c1.metric("Carry c", f"{ms.c:+.3f}")
        c2.metric("Carry regime", regime_label[ms.carry_regime])
        if ms.target_z is not None:
            c3.metric("Target z", f"{ms.target_z:+.2f}σ  ({ms.put_call})")
        else:
            c3.metric("Target z", "—")
        if ms.atmfsratio is not None:
            c4.metric("ATM fwd ratio", f"{ms.atmfsratio:.2f}x")
        else:
            c4.metric("ATM fwd ratio", "—")

        _pair = flow.view.pair
        _base, _quote = _pair[:3], _pair[3:]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"r {_base}", f"{ms.r_f:.2%}")
        c2.metric(f"r {_quote} (implied)", f"{ms.r_d:.2%}")
        try:
            v25dc = interpolate_vol(flow.ccy, h, "25DC")
            v25dp = interpolate_vol(flow.ccy, h, "25DP")
            rr  = v25dc - v25dp
            fly = 0.5 * (v25dc + v25dp) - ms.vol
            c3.metric("25d RR", f"{rr:+.2%}", help=f"25DC {v25dc:.2%} / ATM {ms.vol:.2%} / 25DP {v25dp:.2%}")
            c4.metric("25d Fly", f"{fly:+.2%}", help=f"0.5×(25DC+25DP) − ATM  |  synthetic data")
        except Exception:
            c3.metric("25d RR", "—")
            c4.metric("25d Fly", "—")

        _move_pct = _stop_pct = _stop_price = _loss_budget = None
        _base_ccy_top = flow.view.pair[:3]
        if _target is not None:
            _move_pct = abs(_target - ms.fwd) / ms.fwd
            _stop_pct = _move_pct / flow.target_rr
            _stop_price = ms.fwd * (1 - _stop_pct) if _is_call else ms.fwd * (1 + _stop_pct)
            _loss_budget = LINEAR_NOTIONAL * _stop_pct
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Move to target", f"{_move_pct:+.1%}", help="(target − fwd) / fwd")
            c2.metric(f"Implied stop ({flow.target_rr:.1f}× R:R)", f"{_stop_pct:.1%}", help="move_to_target / R:R — acceptable reversal from fwd before stopping out")
            c3.metric("Stop price", f"{_stop_price:.4f}", help="fwd level implying the stop loss")
            c4.metric(
                "Loss budget",
                fmt_ccy(_loss_budget, _base_ccy_top),
                help=f"Linear notional {fmt_ccy(LINEAR_NOTIONAL, _base_ccy_top)} × stop %. "
                     "Each structure variant is sized so its max loss equals this.",
            )

        st.subheader("Structure scores")
        _sc_pref = st.session_state.get("pref_structure_constraint", "No restriction")
        rows = get_scoring_detail(ms, structure_constraint=_sc_pref)
        _show_constraint = (_sc_pref != "No restriction")
        table_data = []
        for r in rows:
            dims = r["dimensions"]
            eligible = r["eligible"]
            def _s(dim):
                return dims[dim]["score"] if eligible else None
            row = {
                "Structure":      r["display_name"],
                "Target Z":       _s("target_z_abs"),
                "Carry regime":   _s("carry_regime"),
                "ATM/FS ratio":   _s("atmfsratio"),
                "Carry align":    _s("carry_alignment"),
                "Constraint":     _s("structure_constraint"),
                "Total":          r["total_score"] if eligible else None,
                "Overlay":        r["overlay_only"],
                "Eligible":       eligible,
            }
            table_data.append(row)

        score_df = pd.DataFrame(table_data)
        score_df = score_df.sort_values(
            ["Eligible", "Total"], ascending=[False, False]
        ).reset_index(drop=True)
        score_df.index = score_df.index + 1

        def _color(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "color: #aaa"
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v > 0:
                return "color: #1a7a1a; font-weight: bold"
            if v < 0:
                return "color: #b00000; font-weight: bold"
            return "color: #888"

        display_df = score_df.drop(columns=["Overlay", "Eligible"]).copy()
        display_df["Status"] = score_df.apply(
            lambda r: ("overlay" if r["Overlay"] else "") if r["Eligible"] else "gated", axis=1
        )
        _col_order = ["Structure", "Target Z", "Carry regime", "ATM/FS ratio",
                      "Carry align", "Constraint", "Total", "Status"]
        _score_cols = ["Target Z", "Carry regime", "ATM/FS ratio", "Carry align",
                       "Constraint", "Total"]

        display_df = display_df[_col_order]
        display_df[_score_cols] = display_df[_score_cols].astype(object)
        display_df.fillna("—", inplace=True)

        if _show_constraint:
            st.caption(f"Constraint applied: **{_sc_pref}**")

        styled = display_df.style.map(_color, subset=_score_cols)
        st.dataframe(styled, use_container_width=True)

        render_structure_variants(flow, _is_call, _target, _stop_price, _loss_budget)

    # Feedback form (only after a view is active)
    if flow.view:
        try:
            _brief = json.loads(_brief_path.read_text())
            questions = _brief.get("questions", [])
        except Exception:
            questions = []
        if questions:
            with st.expander("Feedback", expanded=False):
                st.caption("3 quick questions — helps calibrate the scorer")
                _fb_key = f"fb_{st.session_state.get('last_prompt','')[:40]}"
                if st.session_state.get(f"{_fb_key}_submitted"):
                    st.success("Thanks — feedback recorded.")
                else:
                    answers = []
                    for i, q in enumerate(questions):
                        val = st.radio(q, ["Yes", "No"], index=None, horizontal=True, key=f"{_fb_key}_q{i}")
                        answers.append(True if val == "Yes" else (False if val == "No" else None))
                    note = st.text_area("Anything else? (optional)", key=f"{_fb_key}_note", height=80)
                    if st.button("Submit feedback", key=f"{_fb_key}_submit"):
                        try:
                            _log_feedback(
                                prompt=st.session_state.get("last_prompt"),
                                pair=flow.view.pair if flow.view else None,
                                answers=answers,
                                questions=questions,
                                note=note or None,
                                user_email=USER_EMAIL,
                            )
                        except Exception:
                            pass
                        st.session_state[f"{_fb_key}_submitted"] = True
                        st.rerun()

    # Structure Evaluation
    if (
        flow.market_state
        and flow.selector_result
        and flow.selector_result.shortlist
        and target_price(flow) is not None
    ):
        render_structure_evaluation(flow, IS_ADMIN, target_price(flow))

    # Advisor chat
    render_advisor_chat(flow)

    # Clarification / error message
    if "clarification" in st.session_state and st.session_state.clarification:
        msg = st.session_state.clarification
        if msg.startswith("ERROR:"):
            st.error(msg[6:].strip())
        else:
            st.info(msg)
        st.session_state.clarification = ""

    if not flow.view:
        with st.form("trade_view_form", clear_on_submit=False):
            _pair_options = list(flow._snapshot.currencies.keys())
            _default_pair = "USDBRL" if "USDBRL" in _pair_options else _pair_options[0]
            _pair_ix = _pair_options.index(_default_pair)
            _dir_label_default = "Lower"
            _horizon_days_default = _HORIZON_OPTIONS[2][1]
            _horizon_labels = [label for label, _ in _HORIZON_OPTIONS]
            _horizon_values = [days for _, days in _HORIZON_OPTIONS]
            _h_ix = _horizon_values.index(_horizon_days_default)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                form_pair = st.selectbox("Pair", _pair_options, index=_pair_ix)
            with c2:
                form_direction_label = st.selectbox(
                    "Direction",
                    list(_DIRECTION_OPTIONS.keys()),
                    index=list(_DIRECTION_OPTIONS.keys()).index(_dir_label_default),
                )
            with c3:
                form_horizon_label = st.selectbox("Horizon", _horizon_labels, index=_h_ix)
            with c4:
                _fallback_target = 5.60
                form_target = st.number_input(
                    "Target",
                    min_value=0.0001,
                    value=float(_fallback_target),
                    step=0.0001,
                    format="%.4f",
                )

            st.markdown("**Trade preferences**")
            st.caption("These preferences are applied in the deterministic engine path. The conversational LLM path remains silent on this screen for now.")

            p1, p2, p3 = st.columns(3)
            with p1:
                form_primary_objective = st.selectbox(
                    "Primary objective",
                    _PRIMARY_OBJECTIVE_OPTIONS,
                    index=_PRIMARY_OBJECTIVE_OPTIONS.index(st.session_state.pref_primary_objective),
                )
            with p2:
                form_structure_constraint = st.selectbox(
                    "Structure constraint",
                    _STRUCTURE_CONSTRAINT_OPTIONS,
                    index=_STRUCTURE_CONSTRAINT_OPTIONS.index(st.session_state.pref_structure_constraint),
                )
            with p3:
                form_trade_management = st.selectbox(
                    "Trade management style",
                    _TRADE_MANAGEMENT_OPTIONS,
                    index=_TRADE_MANAGEMENT_OPTIONS.index(st.session_state.pref_trade_management),
                )

            submitted = st.form_submit_button("Run trade view", type="primary", use_container_width=True)

        if submitted:
            flow.target_rr = st.session_state.target_rr
            st.session_state.clarification = ""
            st.session_state.pref_primary_objective = form_primary_objective
            st.session_state.pref_structure_constraint = form_structure_constraint
            st.session_state.pref_trade_management = form_trade_management
            with st.spinner("Running trade view..."):
                clarification = _submit_structured_view(
                    pair=form_pair,
                    direction=_DIRECTION_OPTIONS[form_direction_label],
                    horizon_days=dict(_HORIZON_OPTIONS)[form_horizon_label],
                    target=form_target,
                )

            if clarification:
                st.session_state.clarification = clarification
            st.rerun()
