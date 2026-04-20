"""
MacroTool — EM FX distribution view tool.

Takes a natural language trade view, extracts pair/direction/magnitude/horizon,
renders a price distribution cone, and shows how big the target move is in σ terms.

Run with:
    .venv/bin/streamlit run interface/app.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from conversation.flow import ConversationFlow, _parse_view_tag
from conversation import context_builder
from interface.charts import build_distribution_fan, build_maturity_histogram
from knowledge_engine.structure_scorer import get_scoring_detail
from analytics.distributions import interpolate_vol
from data.snapshot_loader import load_snapshot
from interface.debug_log import (
    log_prompt, log_view_extracted, log_view_failed,
    log_market_state, log_scorer_result, log_error, read_recent,
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
        "ANTHROPIC_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    ]
    try:
        for k in _secret_keys:
            if k in st.secrets and k not in os.environ:
                os.environ[k] = st.secrets[k]
    except Exception:
        pass

_inject_secrets()

from conversation import tracing as _tracing
_tracing._init_client()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "flow" not in st.session_state:
    st.session_state.flow = ConversationFlow()
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "page" not in st.session_state:
    st.session_state.page = "Trade View"

flow: ConversationFlow = st.session_state.flow


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


_key = _get_api_key()
if _key:
    import anthropic as _anth
    flow._client._client = _anth.Anthropic(api_key=_key)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("MacroTool")
    st.caption("EM FX price distribution")
    st.divider()

    # Navigation
    for label in ("Trade View", "Market Data"):
        active = st.session_state.page == label
        if st.button(
            label,
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.page = label
            st.rerun()

    st.divider()

    if _get_api_key():
        st.success("API key ready")
    else:
        api_key_input = st.text_input(
            "Anthropic API key",
            type="password",
            placeholder="sk-ant-...",
        )
        if api_key_input:
            import anthropic as _anth2
            flow._client._client = _anth2.Anthropic(api_key=api_key_input)

    lf_connected, lf_error = _tracing.init_status()
    if lf_connected:
        st.success("Langfuse connected")
    elif lf_error:
        st.warning(f"Langfuse: {lf_error}")

    st.divider()

    if flow.view:
        st.subheader("View")
        st.write(f"**Pair:** {flow.view.pair}")
        st.write(f"**Direction:** {flow.view.direction.replace('_', ' ')}")
        st.write(f"**Horizon:** {flow.view.horizon_days}d")
        if flow.view.magnitude_pct and flow.market_state:
            sign = 1 if flow.view.direction == "base_higher" else -1
            _t = flow.ccy.spot * (1 + sign * flow.view.magnitude_pct / 100)
            move_from_fwd = (_t / flow.market_state.fwd - 1) * 100
            st.write(f"**Target move from fwd:** {move_from_fwd:+.1f}%")
        elif flow.view.magnitude_pct:
            st.write(f"**Target move:** {flow.view.magnitude_pct:.1f}%")

    st.divider()

    if st.button("↩ New view", use_container_width=True):
        st.session_state.flow = ConversationFlow()
        st.session_state.submitted = False
        st.session_state.last_prompt = ""
        st.rerun()

    with st.expander("Pair reference"):
        st.caption("**USDBRL** — NDF, topside (call) skew, elevated vol")
        st.caption("**USDTRY** — NDF, strong topside skew, normal vol")
        st.caption("**EURPLN** — Deliverable, symmetric skew, normal vol")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_price(flow: ConversationFlow) -> float | None:
    if not (flow.view and flow.view.magnitude_pct):
        return None
    sign = 1 if flow.view.direction == "base_higher" else -1
    return flow.ccy.spot * (1 + sign * flow.view.magnitude_pct / 100)


def _sigma_sentence(flow: ConversationFlow, target: float) -> str:
    flat = flow.flat_distribution
    if not flat:
        return ""
    try:
        fwd = flat.terminal_median
        sigma_sqrtT = math.log(flat.terminal_plus1s / fwd)
        z = math.log(target / fwd) / sigma_sqrtT
        move_from_fwd_pct = (target / fwd - 1) * 100
        direction_word = "appreciation" if flow.view.direction == "base_higher" else "depreciation"
        return (
            f"Target **{target:.4f}** ({move_from_fwd_pct:+.1f}% from the {flow.view.horizon_days}d forward of **{fwd:.4f}**) "
            f"represents **{z:+.1f}σ** — "
            f"a {flow.view.magnitude_pct:.1f}% {direction_word} from spot."
        )
    except (ValueError, ZeroDivisionError):
        return ""


# ---------------------------------------------------------------------------
# View extraction (one LLM call, no text displayed)
# ---------------------------------------------------------------------------

def _extract_view(prompt: str) -> str | None:
    log_prompt(prompt)
    system = context_builder.build_intake_prompt(flow._snapshot)
    full_text = ""
    for chunk in flow._client.stream([{"role": "user", "content": prompt}], system):
        full_text += chunk

    view = _parse_view_tag(full_text)
    if view:
        flow.view = view
        flow.ccy = flow._snapshot.get(view.pair)
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
        return None
    else:
        log_view_failed(full_text)
        return full_text


# ---------------------------------------------------------------------------
# Market Data page
# ---------------------------------------------------------------------------

_TENOR_ORDER = ["1W", "1M", "2M", "3M", "6M", "1Y"]
_DELTA_ORDER = ["10DP", "25DP", "ATM", "25DC", "10DC"]


def _render_market_data() -> None:
    snapshot = load_snapshot()
    st.subheader("Market Data")
    st.caption(
        f"Snapshot date: {snapshot.snapshot_date}  ·  {snapshot.data_note}"
    )

    for pair, ccy in snapshot.currencies.items():
        with st.expander(f"{pair}  —  {ccy.instrument_type}  ·  spot {ccy.spot:.4f}", expanded=True):

            col_fwd, col_vol = st.columns(2)

            # Forwards table
            with col_fwd:
                st.markdown("**Forwards**")
                fwd_rows = [
                    {"Tenor": f.tenor, "Points": f.points, "Outright": f"{f.outright:.4f}"}
                    for f in sorted(ccy.forwards, key=lambda x: _TENOR_ORDER.index(x.tenor))
                ]
                st.dataframe(pd.DataFrame(fwd_rows), use_container_width=True, hide_index=True)

            # Vol surface pivot: rows = tenors, cols = deltas
            with col_vol:
                st.markdown(
                    "**Vol surface** — calls/puts on base currency (ccy1)",
                    help="25DC = 25-delta call on the base currency; 25DP = 25-delta put on the base currency.",
                )
                nodes = {(n.tenor, n.delta): n.vol for n in ccy.vol_surface}
                tenors = [t for t in _TENOR_ORDER if any(k[0] == t for k in nodes)]
                deltas = [d for d in _DELTA_ORDER if any(k[1] == d for k in nodes)]
                vol_data = {
                    d: [f"{nodes.get((t, d), float('nan')):.1%}" for t in tenors]
                    for d in deltas
                }
                vol_df = pd.DataFrame(vol_data, index=tenors)
                vol_df.index.name = "Tenor"
                st.dataframe(vol_df, use_container_width=True)

            # Discount curves (if present)
            if ccy.usd_df_curve or ccy.eur_df_curve:
                st.markdown("**Discount factors**")
                df_cols = {}
                if ccy.usd_df_curve:
                    usd_map = {d.tenor: d.df for d in ccy.usd_df_curve}
                    df_cols["USD DF"] = [usd_map.get(t, "") for t in tenors]
                if ccy.eur_df_curve:
                    eur_map = {d.tenor: d.df for d in ccy.eur_df_curve}
                    df_cols["EUR DF"] = [eur_map.get(t, "") for t in tenors]
                df_df = pd.DataFrame(df_cols, index=tenors)
                df_df.index.name = "Tenor"
                st.dataframe(df_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if st.session_state.page == "Market Data":
    _render_market_data()

else:
    # ---- Trade View page ----

    # Echo the user's prompt back at the top once a view is active
    if flow.view and "last_prompt" in st.session_state and st.session_state.last_prompt:
        st.info(f"**View:** {st.session_state.last_prompt}")

    if flow.flat_distribution and flow.smile_distribution:
        target = _target_price(flow)

        col_fan, col_hist = st.columns(2)
        with col_fan:
            fig_fan = build_distribution_fan(flow.flat_distribution, flow.smile_distribution, target)
            if fig_fan:
                st.plotly_chart(fig_fan, use_container_width=True)
        with col_hist:
            fig_hist = build_maturity_histogram(flow.flat_distribution, flow.smile_distribution, target)
            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True)

        if target:
            sentence = _sigma_sentence(flow, target)
            if sentence:
                st.markdown(sentence)
    else:
        st.markdown("### Describe your trade view")
        st.caption("e.g. *\"Long USDBRL, 5% move, 3 months, high conviction\"*")

    # Structure recommendation
    if flow.market_state and flow.selector_result and flow.selector_result.shortlist:
        st.divider()

        ms = flow.market_state
        h = flow.view.horizon_days
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("r domestic", f"{ms.r_d:.2%}")
        c2.metric("r foreign (implied)", f"{ms.r_d - ms.r_f:.2%}")
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

        with st.expander("Scoring detail", expanded=True):
            rows = get_scoring_detail(ms)
            table_data = []
            for r in rows:
                tz  = r["dimensions"]["target_z_abs"]
                cr  = r["dimensions"]["carry_regime"]
                atm = r["dimensions"]["atmfsratio"]
                table_data.append({
                    "Structure":       r["display_name"],
                    "Overlay":         "✓" if r["overlay_only"] else "",
                    "Eligible":        "✓" if r["eligible"] else "✗ gated",
                    "target_z bucket": tz["bucket"],
                    "tz score":        tz["score"] if r["eligible"] else "—",
                    "carry bucket":    cr["bucket"],
                    "carry score":     cr["score"] if r["eligible"] else "—",
                    "atmfs bucket":    atm["bucket"],
                    "atmfs score":     atm["score"] if r["eligible"] else "—",
                    "Total":           r["total_score"] if r["total_score"] is not None else "—",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        st.subheader("Recommendation")
        primaries = [s for s in flow.selector_result.shortlist if not s.is_exotic]
        overlays  = [s for s in flow.selector_result.shortlist if s.is_exotic]

        for item in primaries:
            with st.container(border=True):
                st.markdown(f"**#{item.rank} {item.display_name}**")
                if item.optimised_for:
                    st.caption(item.optimised_for)
                if item.caution:
                    st.warning(item.caution, icon="⚠️")

        if overlays:
            with st.expander("Overlay structures"):
                for item in overlays:
                    st.markdown(f"**{item.display_name}**")
                    if item.optimised_for:
                        st.caption(item.optimised_for)
                    if item.caution:
                        st.caption(f"⚠️ {item.caution}")

    # Clarification message
    if "clarification" in st.session_state and st.session_state.clarification:
        st.info(st.session_state.clarification)
        st.session_state.clarification = ""

    # Debug log panel
    with st.expander("Debug log", expanded=False):
        entries = read_recent(50)
        if not entries:
            st.caption("No log entries yet.")
        else:
            import json as _json
            log_text = _json.dumps(entries, indent=2, default=str)
            col_copy, col_clear = st.columns([1, 1])
            with col_copy:
                st.code(log_text, language="json")
            with col_clear:
                if st.button("Clear log", key="clear_log"):
                    from pathlib import Path as _P
                    lp = _P(__file__).parent.parent / "logs" / "session.log"
                    lp.write_text("")
                    st.rerun()

    # Input (only on Trade View page)
    prompt = st.chat_input("Describe your trade view (pair, direction, magnitude, horizon)...")

    if prompt:
        api_configured = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or (hasattr(flow._client._client, "api_key") and flow._client._client.api_key)
        )
        if not api_configured:
            st.error("Please enter your Anthropic API key in the sidebar.")
            st.stop()

        st.session_state.last_prompt = prompt
        with st.spinner("Reading view..."):
            clarification = _extract_view(prompt)

        if clarification:
            st.session_state.clarification = clarification
        st.rerun()
