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
from typing import Any

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from conversation.client import (
    get_api_key_for_provider,
    get_gemini_location,
    get_gemini_mode,
    get_gemini_project,
    has_gemini_adc_credentials,
    resolve_model,
    resolve_provider,
)
from conversation.flow import ConversationFlow, target_from_reference
from interface.charts import build_distribution_fan, build_maturity_histogram
from interface.security import current_user_email, is_admin_user, require_login
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
    sign = "-" if amount < 0 else ""
    return f"{sign}{ccy} {abs(amount):,.2f}"


def _fmt_ccy_label(amount: float | None, ccy: str) -> str:
    """Currency formatter for expander labels."""
    return _fmt_ccy(amount, ccy)


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
# LLM config
# ---------------------------------------------------------------------------

def _get_llm_provider() -> str:
    try:
        if "LLM_PROVIDER" in st.secrets:
            return resolve_provider(st.secrets["LLM_PROVIDER"])
    except Exception:
        pass
    return resolve_provider(os.environ.get("LLM_PROVIDER"))


def _get_provider_api_key(provider: str) -> str | None:
    secret_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "GEMINI_API_KEY"
    try:
        if secret_name in st.secrets:
            return st.secrets[secret_name]
    except Exception:
        pass
    return get_api_key_for_provider(provider)


def _get_provider_model(provider: str) -> str:
    secret_name = "ANTHROPIC_MODEL" if provider == "anthropic" else "GEMINI_MODEL"
    try:
        if secret_name in st.secrets:
            return resolve_model(provider, st.secrets[secret_name])
    except Exception:
        pass
    return resolve_model(provider, os.environ.get(secret_name))


def _provider_label(provider: str) -> str:
    return "Anthropic" if provider == "anthropic" else "Google Gemini"


def _get_gcp_service_account_info() -> dict[str, Any] | None:
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
    return None


def _get_gemini_vertex_credentials():
    info = _get_gcp_service_account_info()
    if not info:
        return None
    try:
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    except Exception:
        return None


def _gemini_status() -> tuple[bool, str]:
    mode = get_gemini_mode()
    if mode == "vertexai":
        project = get_gemini_project()
        location = get_gemini_location()
        if not project or not location:
            return False, "Vertex AI config missing project or location"
        service_account_info = _get_gcp_service_account_info()
        if service_account_info:
            if _get_gemini_vertex_credentials() is not None:
                return True, f"Vertex service account ready · {project} / {location}"
            return False, f"Vertex service account secret invalid · {project} / {location}"
        if has_gemini_adc_credentials():
            return True, f"Vertex AI ADC ready · {project} / {location}"
        return False, f"Vertex AI configured but ADC not detected · {project} / {location}"

    if _get_provider_api_key("gemini"):
        return True, "Gemini Developer API key ready"
    return False, "Gemini Developer API key not configured"


def _get_effective_snapshot():
    base = load_snapshot()
    overrides = st.session_state.get("market_overrides", {})
    return apply_overrides(base, overrides) if overrides else base


def _make_flow() -> ConversationFlow:
    provider = _get_llm_provider()
    return ConversationFlow(
        api_key=_get_provider_api_key(provider),
        snapshot=_get_effective_snapshot(),
        provider=provider,
        model=_get_provider_model(provider),
        credentials=_get_gemini_vertex_credentials() if provider == "gemini" else None,
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
    # Navigation
    for label in nav_labels:
        active = st.session_state.page == label
        if st.button(
            label,
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.page = label
            st.rerun()

    st.divider()

    active_provider = _get_llm_provider()
    active_model = _get_provider_model(active_provider)
    st.caption(f"LLM: {_provider_label(active_provider)} · {active_model}")
    if active_provider == "gemini":
        gemini_ready, gemini_message = _gemini_status()
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
    elif _get_provider_api_key(active_provider):
        st.success("API key ready")
    else:
        st.error(f"Server {_provider_label(active_provider)} API key not configured.")

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

    if st.button("↩ New view", use_container_width=True):
        st.session_state.flow = _make_flow()
        st.session_state.submitted = False
        st.session_state.last_prompt = ""
        st.session_state.clarification = ""
        st.session_state.chat_history = []
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
    if flow.market_state is not None:
        anchor = flow.market_state.fwd
    else:
        anchor = flow.ccy.spot
    return target_from_reference(anchor, flow.view.direction, flow.view.magnitude_pct)


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


def _variant_label_with_strikes(structure_id: str, pv) -> str:
    strikes = [f"{k:.4f}" for k in pv.strikes]
    label = pv.variant_label

    if structure_id == "vanilla" and strikes:
        return f"{label} {strikes[0]}"

    if structure_id == "1x1_spread" and len(strikes) >= 2:
        parts = label.split("/")
        if len(parts) == 2:
            return f"{parts[0].strip()} {strikes[0]} / {parts[1].strip()} {strikes[1]}"

    if structure_id == "seagull" and len(strikes) >= 3:
        if " + " in label:
            spread_part, wing_part = label.split(" + ", 1)
            spread_bits = spread_part.split("/")
            if len(spread_bits) == 2:
                return (
                    f"{spread_bits[0].strip()} {strikes[0]} / "
                    f"{spread_bits[1].strip()} {strikes[1]} + "
                    f"{wing_part.strip()} {strikes[2]}"
                )

    if structure_id in {"1x1.5_spread", "1x2_spread"} and len(strikes) >= 2:
        if " / " in label:
            left, right = label.split(" / ", 1)
            return f"{left.strip()} {strikes[0]} / {right.strip()} {strikes[1]}"

    if structure_id == "european_rko" and strikes:
        ko = f"{pv.barrier:.4f}" if pv.barrier is not None else "—"
        return f"{label} {strikes[0]}  ·  KO at {ko}"

    if structure_id == "european_digital" and strikes:
        return f"{label} {strikes[0]}"

    if structure_id == "european_digital_rko" and strikes:
        american_barrier = f"{pv.barrier:.4f}" if pv.barrier is not None else "—"
        return (
            f"{label}  ·  European barrier {strikes[0]}  ·  "
            f"American barrier {american_barrier}"
        )

    if strikes:
        return f"{label}  ·  Strikes: {' / '.join(strikes)}"
    return label


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
# Market Data page
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
        score_df.index = score_df.index + 1  # rank from 1

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

        # Structure variants
        from analytics.structure_pricer import price_variants as _price_variants
        _primary_items = flow.selector_result.shortlist

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
            _base_ccy = flow.view.pair[:3]
            for _i, _item in enumerate(_primary_items):
                try:
                    _pvs = _price_variants(ms, _item.structure_id, target=_target, is_call=_is_call, stop_price=_stop_price, loss_budget=_loss_budget)
                except Exception as _e:
                    st.caption(f"DEBUG {_item.structure_id}: error — {_e}")
                    continue
                if not _pvs:
                    continue
                _title = _item.display_name
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
                                user_email=USER_EMAIL,
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
        from analytics.scenario_generator import (
            GRID_COLS as _SC_GRID_COLS,
            generate_scenarios as _gen_sc,
            valid_grid_rows as _valid_grid_rows,
        )
        from analytics.scenario_pricer import price_scenarios as _price_sc
        from knowledge_engine.scenario_weighter import compute_family_weights as _compute_w
        from knowledge_engine.scenario_scorer  import score_structure       as _score_struct
        from conversation.explanation_context import (
            render_explanation_pack_overview as _render_expl_overview,
            render_structure_comparisons as _render_structure_comparisons,
            render_variant_comparisons as _render_variant_comparisons,
        )
        from knowledge_engine.comparator import (
            VariantEvaluation as _VariantEvaluation,
            build_recommendation_pack as _build_expl_pack,
            summarize_scenario_rows as _summarize_scenario_rows,
        )

        # Context weights — derived from MarketState + PM preferences.
        # Same weight vector applied to every structure → scores comparable.
        _ev_weighter = _compute_w(
            _ev_ms,
            primary_objective=getattr(flow, "primary_objective", "Balanced"),
            trade_management=getattr(flow, "trade_management", "Standard hold"),
        )
        _ev_weights  = _ev_weighter.weights
        _ev_multipliers = _ev_weighter.multipliers
        _ev_base_fired = getattr(_ev_weighter, "base_fired", None)
        if _ev_base_fired is not None:
            _ev_base_weights = {
                _cid: _ev_base_fired.multipliers[_cid] / sum(_ev_base_fired.multipliers.values())
                for _cid in _ev_base_fired.multipliers
            }
        else:
            _ev_base_weights = _ev_weights

        _ev_base = flow.view.pair[:3]

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
        for _ev_item in flow.selector_result.shortlist:
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
            _ev_variants = []
            for _ev_pv in _ev_pvs:
                _ev_rows = _price_sc(
                    _ev_pv, _ev_item.structure_id, _ev_scenarios, _ev_inputs, _ev_is_call
                )
                _ev_score = _score_struct(_ev_rows, _ev_weights)
                _ev_score_base = _score_struct(_ev_rows, _ev_base_weights)
                _ev_variants.append({
                    "pv": _ev_pv,
                    "rows": _ev_rows,
                    "score": _ev_score,
                    "score_base": _ev_score_base,
                })
            if not _ev_variants:
                continue
            _ev_structs.append({
                "item":     _ev_item,
                "variants": _ev_variants,
                "label":    _ev_item.display_name,
            })

        if _ev_structs:
            # persist last results for debug / future scoring
            st.session_state["last_scenario_results"] = _ev_structs[-1]["variants"][-1]["rows"]

            st.subheader("Structure Evaluation")

            # Active weighting — show prominently so it's visible without expanding.
            _base_fired = getattr(_ev_weighter, "base_fired", None)
            _overlay_fired = getattr(_ev_weighter, "overlay_fired", [])
            _fired_all = getattr(_ev_weighter, "fired", [])
            _active_parts = []
            if _base_fired:
                _active_parts.append(_base_fired.id.replace("_", " ").title())
            if _overlay_fired:
                _active_parts.extend(_ctx.id.replace("_", " ").title() for _ctx in _overlay_fired)
            if not _active_parts and _fired_all:
                _active_parts.append(_fired_all[0].id.replace("_", " ").title())
            _active_ctx = " + ".join(_active_parts) if _active_parts else "Baseline grid"
            st.markdown(f"**Active scenario weighting:** {_active_ctx}")

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

            # Show the active grid-cell multipliers/weights.
            _expander_label = f"Selected scenario grid — {_active_ctx}"
            with st.expander(_expander_label, expanded=False):
                _w_rows = []
                for _row in _valid_grid_rows(_ev_ms.T):
                    for _col in _SC_GRID_COLS:
                        _cid = f"{_row}|{_col}"
                        if _cid not in _ev_multipliers:
                            continue
                        _w_rows.append({
                            "Row": _row,
                            "Scenario": _col,
                            "Multiplier": f"{_ev_multipliers[_cid]:.1f}",
                            "Weight": f"{_ev_weights[_cid]:.1%}",
                        })
                st.dataframe(pd.DataFrame(_w_rows), use_container_width=True, hide_index=True)
                if _fired_all:
                    _ctx_rows = [{
                        "Layer": "Base" if _ctx == _base_fired else "Overlay",
                        "Weighting": _ctx.id.replace("_", " "),
                        "Reasoning": _ctx.comment,
                    } for _ctx in _fired_all]
                    st.dataframe(pd.DataFrame(_ctx_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No context-specific weighting active — the baseline grid applies unchanged.")

            if IS_ADMIN:
                try:
                    _expl_variants = {
                        _s["item"].structure_id: [_v["pv"] for _v in _s["variants"]]
                        for _s in _ev_structs
                    }
                    _expl_scores = {
                        _s["item"].structure_id: _s["variants"][0]["score"]
                        for _s in _ev_structs
                    }
                    _expl_variant_evals = {
                        _s["item"].structure_id: [
                            _VariantEvaluation(
                                variant=_v["pv"],
                                rows=_v["rows"],
                                base_score=_v["score_base"],
                                pm_score=_v["score"],
                                aggregates=_summarize_scenario_rows(_v["rows"]),
                            )
                            for _v in _s["variants"]
                        ]
                        for _s in _ev_structs
                    }
                    _expl_pack = _build_expl_pack(
                        _ev_ms,
                        flow.selector_result,
                        _expl_variants,
                        _expl_scores,
                        variant_evaluations_by_structure=_expl_variant_evals,
                    )
                    with st.expander("Explanation pack preview", expanded=False):
                        st.code(_render_expl_overview(_expl_pack), language="text")
                        if _expl_pack.variant_comparisons:
                            with st.expander("Variant comparisons", expanded=False):
                                st.code(
                                    _render_variant_comparisons(_expl_pack.variant_comparisons),
                                    language="text",
                                )
                        if _expl_pack.comparisons:
                            with st.expander("Structure comparisons", expanded=False):
                                st.code(
                                    _render_structure_comparisons(list(_expl_pack.comparisons.values())),
                                    language="text",
                                )
                        if not _expl_pack.variant_comparisons and not _expl_pack.comparisons:
                            st.caption("No comparator sections available for this pack.")
                except Exception as _e:
                    with st.expander("Explanation pack preview", expanded=False):
                        st.caption(f"Unable to build explanation pack preview: {_e}")

            for _ev_s in _ev_structs:
                with st.expander(_ev_s["label"], expanded=False):
                    for _ix, _ev_v in enumerate(_ev_s["variants"]):
                        _pv0 = _ev_v["pv"]
                        _score = _ev_v["score"]
                        _score_base = _ev_v["score_base"]
                        _notional_str = (
                            _fmt_ccy(_pv0.structure_notional, _ev_base)
                            if _pv0.structure_notional is not None else None
                        )
                        _variant_title = _variant_label_with_strikes(_ev_s["item"].structure_id, _pv0)
                        if _notional_str:
                            _variant_title += f"  ·  Notional: {_notional_str}"
                        elif _pv0.is_zero_cost and _pv0.max_loss_pct < 1e-9:
                            _variant_title += "  ·  Notional: unscaled"
                        _base_pct = f"{_score_base.score_pct:.2%}"
                        _base_ccy = (
                            f"  ({_fmt_ccy_label(_score_base.score_ccy, _ev_base)})"
                            if _score_base.score_ccy is not None
                            else ("  (unscaled)" if _pv0.structure_notional is None else "")
                        )
                        _ctx_pct = f"{_score.score_pct:.2%}"
                        _ctx_ccy = (
                            f"  ({_fmt_ccy_label(_score.score_ccy, _ev_base)})"
                            if _score.score_ccy is not None
                            else ("  (unscaled)" if _pv0.structure_notional is None else "")
                        )
                        _variant_title += (
                            f"  ·  Scenario weighted P&L: {_base_pct}{_base_ccy}"
                            f"  ·  PM overlay weighted P&L: {_ctx_pct}{_ctx_ccy}"
                        )

                        with st.expander(_variant_title, expanded=(_ix == 0)):
                            _bd_by_cell = {b.scenario_id: b for b in _score.cells}
                            _summary_rows = []
                            for _row in _valid_grid_rows(_ev_ms.T):
                                for _col in _SC_GRID_COLS:
                                    _cid = f"{_row}|{_col}"
                                    _bd = _bd_by_cell.get(_cid)
                                    if _bd is None:
                                        continue
                                    _summary_rows.append({
                                        "Row": _row,
                                        "Scenario": _col,
                                        "P&L": f"{_bd.pnl_pct:+.2%}  ({_fmt_ccy(_bd.pnl_ccy, _ev_base)})",
                                        "Multiplier": f"{_bd.multiplier:.1f}",
                                        "Weight": f"{_bd.normalized_weight:.1%}",
                                        "Weighted contrib": (
                                            f"{_bd.contrib_pct:+.2%}"
                                            + (f"  ({_fmt_ccy(_bd.contrib_ccy, _ev_base)})" if _bd.contrib_ccy is not None else "")
                                        ),
                                    })
                            if _summary_rows:
                                st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)

                            with st.expander("Scenarios", expanded=False):
                                _ev_by_row: dict[str, list] = {}
                                for r in _ev_v["rows"]:
                                    _ev_by_row.setdefault(r["row"], []).append(r)
                                for _row in _valid_grid_rows(_ev_ms.T):
                                    if _row not in _ev_by_row:
                                        continue
                                    st.markdown(f"**{_row}**")
                                    _row_rows = sorted(_ev_by_row[_row], key=lambda x: _SC_GRID_COLS.index(x["col"]))
                                    _row_df = pd.DataFrame([{
                                        "Scenario":  r["col"],
                                        "T%":        f"{r['time_fraction']:.0%}",
                                        "Fwd":       f"{r['scenario_fwd']:.4f}",
                                        "Spot":      f"{r['scenario_spot']:.4f}",
                                        "Vol shift": r["vol_shift"] if isinstance(r["vol_shift"], str) else (f"{r['vol_shift']:+.0%}" if r["vol_shift"] != 0 else "—"),
                                        "Vol":       f"{r['scenario_vol']:.1%}",
                                        "Price":     f"{r['price_pct']:.2%}  ({_fmt_ccy(r['price_ccy'], _ev_base)})",
                                        "P&L":       f"{r['pnl_pct']:+.2%}  ({_fmt_ccy(r['pnl_ccy'], _ev_base)})",
                                    } for r in _row_rows])
                                    st.dataframe(_row_df, use_container_width=True, hide_index=True)

    # Advisor chat — only when explanation pack is available
    if flow.view and flow.explanation_pack_context:
        st.divider()
        st.subheader("Ask the advisor")

        _chat_system = (
            "You are a trade structuring advisor for a macro fund PM. "
            "A deterministic engine has produced an EM FX options recommendation. "
            "Your role is to help the PM understand and interrogate it in a continuing conversation.\n\n"
            "You have access to the full recommendation explanation pack below: the chosen structure, "
            "full variant ranking by scenario-weighted P&L, pairwise comparisons with key alternatives, "
            "and the primary risk factors.\n\n"
            "On each turn:\n"
            "- Answer the PM's question directly and qualitatively.\n"
            "- For any question about structure selection, comparisons, trade-offs, or payoff "
            "characteristics: use ONLY what is in the explanation pack. Do not supplement with "
            "general knowledge about how these structures typically behave.\n"
            "- If the pack does not contain enough to answer the question fully, say so explicitly "
            "rather than filling the gap from general knowledge.\n"
            "- If a structure comparison is not in the pack, say it is not available.\n"
            "- Keep answers concise. Do not re-narrate the full recommendation unprompted.\n"
            "- General knowledge is only permitted for questions that are genuinely out of scope "
            "of the pack — e.g. macro backdrop, broader market context, timing. In those cases, "
            "flag clearly that you are drawing on general knowledge, not the engine output.\n\n"
            "Constraints:\n"
            "- Do not reveal internal scoring weights, thresholds, or formulas.\n"
            "- Do not invent or embellish any claim about structure behaviour, payoff, or risk "
            "that is not explicitly stated in the pack below.\n"
            "- Do not invent strikes, premiums, barriers, or P&L figures.\n"
            "- Speak like a senior EM structurer briefing a PM, not a generic assistant.\n\n"
            "[RECOMMENDATION EXPLANATION PACK]\n"
            + flow.explanation_pack_context
        )

        for _msg in st.session_state.chat_history:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])

        if _chat_input := st.chat_input("Ask about the recommendation…"):
            st.session_state.chat_history.append({"role": "user", "content": _chat_input})
            with st.chat_message("user"):
                st.markdown(_chat_input)

            with st.chat_message("assistant"):
                _response_placeholder = st.empty()
                _accumulated = ""
                try:
                    for _chunk in flow._client.stream(
                        st.session_state.chat_history,
                        system=_chat_system,
                    ):
                        _accumulated += _chunk
                        _response_placeholder.markdown(_accumulated + "▌")
                    _response_placeholder.markdown(_accumulated)
                    st.session_state.chat_history.append({"role": "assistant", "content": _accumulated})
                except Exception as _chat_err:
                    _response_placeholder.error(f"LLM error: {_chat_err}")

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
