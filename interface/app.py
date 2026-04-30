"""
MacroTool — EM FX distribution view tool.

Takes a natural language trade view, extracts pair/direction/magnitude/horizon,
renders a price distribution cone, and shows how big the target move is in σ terms.

Run with:
    .venv/bin/streamlit run interface/app.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from conversation.flow import ConversationFlow
from interface.charts import build_distribution_fan, build_maturity_histogram
from knowledge_engine.structure_scorer import get_scoring_detail
from knowledge_engine.models import TradeView
from analytics.distributions import interpolate_vol
from data.snapshot_loader import load_snapshot
from interface.debug_log import (
    log_prompt, log_view_extracted,
    log_market_state, log_scorer_result, log_error,
)

# ---------------------------------------------------------------------------
# Sizing convention
# ---------------------------------------------------------------------------

LINEAR_NOTIONAL = 100.0   # base ccy units; equivalent linear-trade notional

_CCY_SYM = {"USD": "$", "EUR": "€", "GBP": "£"}


def _fmt_ccy(amount: float | None, ccy: str) -> str:
    """Format a base-ccy amount. Raises on unknown ccy — fail loud, not silent."""
    if amount is None:
        return "—"
    if ccy not in _CCY_SYM:
        raise ValueError(f"No currency symbol mapping for base ccy {ccy!r}")
    return f"{_CCY_SYM[ccy]}{amount:,.2f}"


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
        "SUPABASE_URL",
        "SUPABASE_KEY",
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

from interface.supabase_logger import log_query as _log_query, log_feedback as _log_feedback, reinit as _sb_reinit, init_status as _sb_status
_sb_reinit()
from knowledge_engine.loader import load_structure_profiles as _lsp
_lsp.cache_clear()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "flow" not in st.session_state:
    st.session_state.flow = ConversationFlow()
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "page" not in st.session_state:
    st.session_state.page = "Trade View"
if "target_rr" not in st.session_state:
    st.session_state.target_rr = 3.0
if "clarification" not in st.session_state:
    st.session_state.clarification = ""
if "pref_primary_objective" not in st.session_state:
    st.session_state.pref_primary_objective = "Balanced"
if "pref_structure_constraint" not in st.session_state:
    st.session_state.pref_structure_constraint = "No restriction"
if "pref_trade_management" not in st.session_state:
    st.session_state.pref_trade_management = "Standard hold"

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
    try:
        from importlib.metadata import version as _pkg_version
        st.caption(f"EM FX price distribution · v{_pkg_version('macrotool')}")
    except Exception:
        st.caption("EM FX price distribution")
    st.divider()

    # Navigation
    for label in ("Trade View", "Market Data", "Structure Selection", "Context Rules", "Query log"):
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

    sb_connected, sb_error = _sb_status()
    if sb_connected:
        st.success("Supabase connected")
    else:
        st.warning(f"Supabase: {sb_error}")

    st.divider()

    st.caption("Risk / Reward target")
    st.session_state.target_rr = st.slider(
        "Risk 1 to make",
        min_value=1.5,
        max_value=10.0,
        value=st.session_state.target_rr,
        step=0.5,
        format="%.1f×",
    )

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
        st.session_state.clarification = ""
        st.rerun()

    with st.expander("Pair reference"):
        st.caption("**USDBRL** — topside (call) skew, elevated vol, high carry")
        st.caption("**USDTRY** — strong topside skew, very high carry")
        st.caption("**EURPLN** — symmetric skew, normal vol, moderate carry")
        st.caption("**EURUSD** — near-symmetric skew, low vol, negative carry for long EUR")
        st.caption("**USDCNH** — mild topside skew, low vol, negative carry for long USD")
        st.caption("**USDMXN** — strong topside skew, high vol, high positive carry for short USD")
        st.caption("**USDJPY** — downside skew, medium vol, negative carry for long USD")

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

    if direction == "base_higher" and target <= ccy.spot:
        return "ERROR: For `Base higher`, target must be above spot."
    if direction == "base_lower" and target >= ccy.spot:
        return "ERROR: For `Base lower`, target must be below spot."

    magnitude_pct = abs(target / ccy.spot - 1.0) * 100.0
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
        )
    except Exception as e:
        log_error("supabase_log_query", e)

    st.session_state.last_prompt = _build_prompt_summary(pair, direction, horizon_days, target)
    return None


# ---------------------------------------------------------------------------
# Market Data page
# ---------------------------------------------------------------------------

_TENOR_ORDER = ["1W", "1M", "2M", "3M", "6M", "1Y"]
_DELTA_ORDER = ["10DP", "25DP", "ATM", "25DC", "10DC"]


def _render_query_log() -> None:
    from interface.supabase_logger import fetch_queries
    st.subheader("Query log")
    rows = fetch_queries()
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
        "pair":          "Pair",
        "direction":     "Direction",
        "magnitude_pct": "Mag %",
        "horizon_days":  "Horizon",
        "target_z":      "Target z",
        "carry_regime":  "Carry regime",
        "top_structure": "Top structure",
        "prompt":        "Prompt",
    })
    df = df[["Time", "Pair", "Direction", "Mag %", "Horizon", "Target z", "Carry regime", "Top structure", "Prompt"]]
    st.dataframe(df, use_container_width=True, hide_index=True)


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

            # Discount curve — base currency only
            base_ccy = pair[:3]
            base_curve_map = {
                "USD": getattr(ccy, "usd_df_curve", []),
                "EUR": getattr(ccy, "eur_df_curve", []),
                "GBP": getattr(ccy, "gbp_df_curve", []),
            }
            base_curve = base_curve_map.get(base_ccy, [])
            if base_curve:
                st.markdown("**Discount factors**")
                df_map = {d.tenor: d.df for d in base_curve}
                df_df = pd.DataFrame(
                    {f"{base_ccy} DF": [df_map.get(t, "") for t in tenors]},
                    index=tenors,
                )
                df_df.index.name = "Tenor"
                st.dataframe(df_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if st.session_state.page == "Market Data":
    _render_market_data()

elif st.session_state.page == "Query log":
    _render_query_log()

elif st.session_state.page == "Structure Selection":
    from interface.decision_parameters import render as _render_decision_params
    _render_decision_params()

elif st.session_state.page == "Context Rules":
    from interface.context_rules import render as _render_context_rules
    _render_context_rules()

else:
    # ---- Trade View page ----

    # Testing brief
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
        st.markdown("### Enter trade view")
        st.caption("Select the pair, direction, horizon, and target level.")

    # Structure recommendation
    if flow.market_state and flow.selector_result and flow.selector_result.shortlist:
        st.divider()

        ms = flow.market_state
        h = flow.view.horizon_days
        _is_call = flow.view.direction == "base_higher"
        _target = _target_price(flow)
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
                _fmt_ccy(_loss_budget, _base_ccy_top),
                help=f"Linear notional {_fmt_ccy(LINEAR_NOTIONAL, _base_ccy_top)} × stop %. "
                     "Each structure variant is sized so its max loss equals this.",
            )

        st.subheader("Structure scores")
        rows = get_scoring_detail(ms)
        table_data = []
        for r in rows:
            dims = r["dimensions"]
            eligible = r["eligible"]
            def _s(dim):
                return dims[dim]["score"] if eligible else None
            table_data.append({
                "Structure":      r["display_name"],
                "Target Z":       _s("target_z_abs"),
                "Carry regime":   _s("carry_regime"),
                "ATM/FS ratio":   _s("atmfsratio"),
                "Carry align":    _s("carry_alignment"),
                "Total":          r["total_score"] if eligible else None,
                "Overlay":        r["overlay_only"],
                "Eligible":       eligible,
            })

        score_df = pd.DataFrame(table_data)
        score_df = score_df.sort_values(
            ["Eligible", "Total"], ascending=[False, False]
        ).reset_index(drop=True)
        score_df.index = score_df.index + 1  # rank from 1

        def _color(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "color: #aaa"
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v > 0:
                return f"color: #1a7a1a; font-weight: bold"
            if v < 0:
                return "color: #b00000; font-weight: bold"
            return "color: #888"

        display_df = score_df.drop(columns=["Overlay", "Eligible"]).copy()
        display_df["Status"] = score_df.apply(
            lambda r: ("overlay" if r["Overlay"] else "") if r["Eligible"] else "gated", axis=1
        )
        display_df = display_df[["Structure", "Target Z", "Carry regime", "ATM/FS ratio", "Carry align", "Total", "Status"]]
        display_df[["Target Z", "Carry regime", "ATM/FS ratio", "Carry align", "Total"]] = (
            display_df[["Target Z", "Carry regime", "ATM/FS ratio", "Carry align", "Total"]].astype(object)
        )
        display_df.fillna("—", inplace=True)

        styled = display_df.style.map(
            _color, subset=["Target Z", "Carry regime", "ATM/FS ratio", "Carry align", "Total"]
        )
        st.dataframe(styled, use_container_width=True)

        # Structure variants
        from analytics.structure_pricer import price_variants as _price_variants
        _primary_items = flow.selector_result.shortlist[:3]

        _any_variants = any(
            _price_variants(ms, s.structure_id, target=_target, is_call=_is_call, stop_price=_stop_price, loss_budget=_loss_budget)
            for s in _primary_items
        )
        if _any_variants:
            st.subheader("Structure variants")
            st.caption(
                "Indicative pricing — flat ATM vol for all strikes. "
                "Premium and payoff as % of spot. "
                "**Payout/$1**: gross payoff at target per $1 of max loss (zero-cost seagull: "
                "loss on short wing at stop price, expiry basis — understates MtM risk before expiry)."
            )
            _base_ccy, _quote_ccy = flow.view.pair[:3], flow.view.pair[3:]
            _long_leg  = "call" if _is_call else "put"
            _short_leg = "put"  if _is_call else "call"
            _variant_title = {
                "vanilla":              f"{_base_ccy} {_long_leg}",
                "1x1_spread":           f"{_base_ccy} {_long_leg} / {_quote_ccy} {_short_leg} spread",
                "1x2_spread":           f"{_base_ccy} 1×2 {_long_leg} spread",
                "seagull":              f"{_base_ccy} {_long_leg} / {_quote_ccy} {_short_leg} spread + sold {_base_ccy} {_short_leg}",
                "european_digital":     f"{_base_ccy} {_long_leg} digital",
                "european_digital_rko": f"{_base_ccy} {_long_leg} digital + KO",
            }
            for _i, _item in enumerate(_primary_items):
                try:
                    _pvs = _price_variants(ms, _item.structure_id, target=_target, is_call=_is_call, stop_price=_stop_price, loss_budget=_loss_budget)
                except Exception as _e:
                    st.caption(f"DEBUG {_item.structure_id}: error — {_e}")
                    continue
                if not _pvs:
                    continue
                _title = _variant_title.get(_item.structure_id, _item.display_name)
                with st.expander(_title, expanded=(_i == 0)):
                    _rows = []
                    _has_barrier = any(pv.barrier is not None for pv in _pvs)
                    _has_wing    = any(pv.wing_ratio is not None for pv in _pvs)
                    for pv in _pvs:
                        _payoff = pv.payoff_at_target_pct
                        if _payoff and _payoff > 1e-6 and pv.max_loss_pct > 0:
                            _payout_per_1 = f"{_payoff / pv.max_loss_pct:.1f}×"
                        else:
                            _payout_per_1 = "—"
                        _prem_cell = (
                            "zero cost" if pv.is_zero_cost
                            else f"{pv.net_premium_pct:.1%}  ({_fmt_ccy(pv.net_premium_ccy, _base_ccy)})"
                        )
                        _payoff_cell = (
                            f"{_payoff:.0%}  ({_fmt_ccy(pv.payoff_at_target_ccy, _base_ccy)})"
                            if _payoff is not None else "—"
                        )
                        r = {
                            "Variant":    pv.variant_label,
                            "Strikes":    " / ".join(f"{K:.4f}" for K in pv.strikes),
                            "Notional":   _fmt_ccy(pv.structure_notional, _base_ccy),
                            "Premium":    _prem_cell,
                            "Break-even": f"{pv.breakeven:.4f}" if pv.breakeven is not None else "—",
                            "Payout at target": _payoff_cell,
                            "Max loss":   f"{pv.max_loss_pct:.1%}  ({_fmt_ccy(pv.max_loss_ccy, _base_ccy)})",
                            "Payout/$1":  _payout_per_1,
                        }
                        if _has_barrier:
                            r["Barrier"] = f"{pv.barrier:.4f}" if pv.barrier is not None else "—"
                        if _has_wing:
                            r["Wing ×"] = f"{pv.wing_ratio:.2f}" if pv.wing_ratio is not None else "—"
                        _rows.append(r)
                    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

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
                            )
                        except Exception:
                            pass
                        st.session_state[f"{_fb_key}_submitted"] = True
                        st.rerun()

    # Structure Evaluation — scenario tables for all shortlisted structures
    if (
        flow.market_state
        and flow.selector_result
        and flow.selector_result.shortlist
        and _target_price(flow) is not None
    ):
        _ev_ms = flow.market_state
        _ev_is_call = flow.view.direction == "base_higher"
        _ev_target = _target_price(flow)
        _ev_move = abs(_ev_target - _ev_ms.fwd) / _ev_ms.fwd
        _ev_stop_pct = _ev_move / flow.target_rr
        _ev_stop = _ev_ms.fwd * (1 - _ev_stop_pct) if _ev_is_call else _ev_ms.fwd * (1 + _ev_stop_pct)
        _ev_loss_budget = LINEAR_NOTIONAL * _ev_stop_pct

        from analytics.structure_pricer import price_variants as _pv_fn
        from analytics.scenario_generator import generate_scenarios as _gen_sc, FAMILIES as _SC_FAMILIES
        from analytics.scenario_pricer import price_scenarios as _price_sc
        from knowledge_engine.scenario_weighter import compute_family_weights as _compute_w
        from knowledge_engine.scenario_scorer  import score_structure       as _score_struct

        # Tier 1 weights — derived from MarketState (already view-conditioned).
        # Same weight vector applied to every structure → scores comparable.
        _ev_weighter = _compute_w(_ev_ms)
        _ev_weights  = _ev_weighter.weights

        _ev_base, _ev_quote = flow.view.pair[:3], flow.view.pair[3:]
        _ev_long = "call" if _ev_is_call else "put"
        _ev_short = "put" if _ev_is_call else "call"
        _ev_vtitles = {
            "vanilla":              f"{_ev_base} {_ev_long}",
            "1x1_spread":           f"{_ev_base} {_ev_long} / {_ev_quote} {_ev_short} spread",
            "1x2_spread":           f"{_ev_base} 1×2 {_ev_long} spread",
            "seagull":              f"{_ev_base} {_ev_long} / {_ev_quote} {_ev_short} spread + sold {_ev_base} {_ev_short}",
            "european_digital":     f"{_ev_base} {_ev_long} digital",
            "european_digital_rko": f"{_ev_base} {_ev_long} digital + KO",
        }

        _ev_inputs = {
            "spot": _ev_ms.spot,
            "forward": _ev_ms.fwd,
            "implied_vol": _ev_ms.vol,
            "tenor_years": _ev_ms.T,
            "target": _ev_target,
            "r_d": _ev_ms.r_d,
            "r_f": _ev_ms.r_f,
        }
        _ev_scenarios = _gen_sc(_ev_inputs)  # same scenario set for all structures

        # Build per-structure data (skip structures with no priceable variants)
        _ev_structs = []
        for _ev_item in flow.selector_result.shortlist[:3]:
            try:
                _ev_pvs = _pv_fn(
                    _ev_ms, _ev_item.structure_id,
                    target=_ev_target, is_call=_ev_is_call,
                    stop_price=_ev_stop, loss_budget=_ev_loss_budget,
                )
            except Exception:
                continue
            if not _ev_pvs:
                continue
            _ev_rows = _price_sc(
                _ev_pvs[0], _ev_item.structure_id, _ev_scenarios, _ev_inputs, _ev_is_call
            )
            _ev_score = _score_struct(_ev_rows, _ev_weights)
            _ev_structs.append({
                "item":    _ev_item,
                "pvs":     _ev_pvs,
                "rows":    _ev_rows,
                "score":   _ev_score,
                "label":   _ev_vtitles.get(_ev_item.structure_id, _ev_item.display_name),
            })

        if _ev_structs:
            # persist last results for debug / future scoring
            st.session_state["last_scenario_results"] = _ev_structs[-1]["rows"]

            st.subheader("Structure Evaluation")

            # Active context — show prominently so it's visible without expanding.
            _active_ctx = (
                _ev_weighter.fired[0].id.replace("_", " ").title()
                if _ev_weighter.fired else "Baseline (equal weights)"
            )
            st.markdown(f"**Scenario context:** {_active_ctx}")

            # Supporting market state detail for sense-checking.
            _carry_lbl = {0: "noisy", 1: "potential", 2: "high"}[_ev_ms.carry_regime]
            _dir_lbl = "with-carry" if _ev_ms.with_carry else "counter-carry"
            _tz_lbl = (
                f"target {abs(_ev_ms.target_z):.2f}σ from forward"
                if _ev_ms.target_z is not None else "no target"
            )
            _tenor_days = int(round(_ev_ms.T * 365))
            _tenor_lbl = f"{_tenor_days}d tenor"
            _vol_lbl = f"vol {_ev_ms.vol:.1%}"
            st.caption(
                f"carry {_carry_lbl} ({_dir_lbl})  ·  {_tz_lbl}  ·  "
                f"{_tenor_lbl}  ·  {_vol_lbl}.  "
                "Scenario MtM as % of entry spot.  P&L vs entry premium."
            )

            # Show the family weights that come out of Tier 1, plus which
            # contexts fired — fully transparent.
            with st.expander("Scenario weights for this trade", expanded=False):
                _w_rows = [
                    {
                        "Family": _fam.replace("_", " ").title(),
                        "Weight": f"{_ev_weights[_fam]:.1%}",
                    }
                    for _fam in _SC_FAMILIES
                    if _fam in _ev_weights
                ]
                st.dataframe(pd.DataFrame(_w_rows), use_container_width=True, hide_index=True)
                if _ev_weighter.fired:
                    st.markdown("**Active contexts**")
                    _ctx_rows = []
                    for _ctx in _ev_weighter.fired:
                        _adj_str = "  /  ".join(
                            f"{_fam.replace('_', ' ').title()} {_delta:+.2f}"
                            for _fam, _delta in _ctx.adjustments.items()
                        )
                        _ctx_rows.append({
                            "Context":     _ctx.id.replace("_", " "),
                            "Adjustments": _adj_str,
                            "Reasoning":   _ctx.comment,
                        })
                    st.dataframe(pd.DataFrame(_ctx_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption(
                        "No contexts active — every family kept its baseline weight "
                        f"of {_ev_weighter.baseline:.3f}."
                    )

            for _ev_s in _ev_structs:
                _pv0 = _ev_s["pvs"][0]
                _score = _ev_s["score"]
                _notional_str = (
                    _fmt_ccy(_pv0.structure_notional, _ev_base)
                    if _pv0.structure_notional is not None else None
                )
                _score_str = (
                    f"{_score.score_pct:+.2%}"
                    + (f"  ({_fmt_ccy(_score.score_ccy, _ev_base)})"
                       if _score.score_ccy is not None else "")
                )
                _struct_title = f"{_ev_s['label']} — {_pv0.variant_label}"
                if _notional_str:
                    _struct_title += f"  ·  Notional: {_notional_str}"
                _struct_title += f"  ·  Weighted P&L: {_score_str}"

                with st.expander(_struct_title, expanded=False):
                    # Family summary — index breakdown by family for fast lookup
                    _bd_by_family = {b.family: b for b in _score.families}

                    _summary_rows = []
                    for _fam in _SC_FAMILIES:
                        _bd = _bd_by_family.get(_fam)
                        if _bd is None:
                            continue
                        _summary_rows.append({
                            "Family":     _fam.replace("_", " ").title(),
                            "Scenarios":  _bd.n_scenarios,
                            "Avg P&L":    f"{_bd.avg_pnl_pct:+.2%}  ({_fmt_ccy(_bd.avg_pnl_ccy, _ev_base)})",
                            "Weight":     f"{_bd.weight:.1%}",
                            "Weighted contrib": (
                                f"{_bd.contrib_pct:+.2%}"
                                + (f"  ({_fmt_ccy(_bd.contrib_ccy, _ev_base)})"
                                   if _bd.contrib_ccy is not None else "")
                            ),
                        })
                    if _summary_rows:
                        st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)

                    # Reconstruct family-grouped row map for the Scenarios expander.
                    _ev_by_family: dict[str, list] = {}
                    for r in _ev_s["rows"]:
                        _ev_by_family.setdefault(r["family"], []).append(r)

                    # Scenarios — full per-scenario detail, grouped by family (markdown headers, no nested expanders)
                    with st.expander("Scenarios", expanded=False):
                        for _fam in _SC_FAMILIES:
                            if _fam not in _ev_by_family:
                                continue
                            _fam_label = _fam.replace("_", " ").title()
                            st.markdown(f"**{_fam_label}**")
                            _fam_rows = _ev_by_family[_fam]
                            _fam_df = pd.DataFrame([{
                                "Scenario":  r["scenario_id"],
                                "T%":        f"{r['time_fraction']:.0%}",
                                "Fwd":       f"{r['scenario_fwd']:.4f}",
                                "Spot":      f"{r['scenario_spot']:.4f}",
                                "Vol shift": f"{r['vol_shift']:+.0%}" if r["vol_shift"] != 0 else "—",
                                "Vol":       f"{r['scenario_vol']:.1%}",
                                "Price":     f"{r['price_pct']:.2%}  ({_fmt_ccy(r['price_ccy'], _ev_base)})",
                                "P&L":       f"{r['pnl_pct']:+.2%}  ({_fmt_ccy(r['pnl_ccy'], _ev_base)})",
                            } for r in _fam_rows])
                            st.dataframe(_fam_df, use_container_width=True, hide_index=True)

    # Clarification / error message
    if "clarification" in st.session_state and st.session_state.clarification:
        msg = st.session_state.clarification
        if msg.startswith("ERROR:"):
            st.error(msg[6:].strip())
        else:
            st.info(msg)
        st.session_state.clarification = ""

    if not flow.view:
        # Structured input (only on Trade View page)
        with st.form("trade_view_form", clear_on_submit=False):
            _pair_options = list(flow._snapshot.currencies.keys())
            _default_pair = _pair_options[0]
            _pair_ix = 0
            _dir_label_default = "Higher"
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
                _pair_spot = flow._snapshot.get(_default_pair).spot
                _fallback_target = _pair_spot * 1.05
                form_target = st.number_input(
                    "Target",
                    min_value=0.0001,
                    value=float(_fallback_target),
                    step=0.0001,
                    format="%.4f",
                )

            st.markdown("**Trade preferences**")
            st.caption("Optional for now — captured in the UI only, not yet applied to scoring.")

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
