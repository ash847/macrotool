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

from conversation.flow import ConversationFlow, _parse_view_tag
from conversation import context_builder
from interface.charts import build_distribution_fan, build_maturity_histogram

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
# Session state
# ---------------------------------------------------------------------------

if "flow" not in st.session_state:
    st.session_state.flow = ConversationFlow()
if "submitted" not in st.session_state:
    st.session_state.submitted = False

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

    st.divider()

    if flow.view:
        st.subheader("View")
        st.write(f"**Pair:** {flow.view.pair}")
        st.write(f"**Direction:** {flow.view.direction.replace('_', ' ')}")
        st.write(f"**Horizon:** {flow.view.horizon_days}d")
        if flow.view.magnitude_pct:
            st.write(f"**Target move:** {flow.view.magnitude_pct:.1f}%")

    st.divider()

    if st.button("↩ New view", use_container_width=True):
        st.session_state.flow = ConversationFlow()
        st.session_state.submitted = False
        st.rerun()

    with st.expander("Pair reference"):
        st.caption("**USDBRL** — NDF, topside (call) skew, elevated vol")
        st.caption("**USDTRY** — NDF, strong topside skew, normal vol")
        st.caption("**EURPLN** — Deliverable, symmetric skew, normal vol")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_price(flow: ConversationFlow) -> float | None:
    """Compute the target spot price from the view's magnitude_pct."""
    if not (flow.view and flow.view.magnitude_pct):
        return None
    sign = 1 if flow.view.direction == "base_higher" else -1
    return flow.ccy.spot * (1 + sign * flow.view.magnitude_pct / 100)


def _sigma_sentence(flow: ConversationFlow, target: float) -> str:
    """Single sentence: how big is the target in σ terms."""
    flat = flow.flat_distribution
    if not flat:
        return ""
    try:
        sigma_sqrtT = math.log(flat.terminal_plus1s / flat.terminal_median)
        z = math.log(target / flat.terminal_median) / sigma_sqrtT
        direction_word = "appreciation" if flow.view.direction == "base_higher" else "depreciation"
        return (
            f"A {flow.view.magnitude_pct:.1f}% {flow.view.pair} {direction_word} "
            f"to **{target:.4f}** represents **{z:+.1f}σ** "
            f"from the forward at the {flow.view.horizon_days}d horizon."
        )
    except (ValueError, ZeroDivisionError):
        return ""


# ---------------------------------------------------------------------------
# View extraction (one LLM call, no text displayed)
# ---------------------------------------------------------------------------

def _extract_view(prompt: str) -> str | None:
    """
    Run one LLM intake call to extract the VIEW tag.
    Returns the full LLM response text (used for clarification if no tag found).
    Sets flow.view, flow.ccy, and computes distributions on success.
    """
    system = context_builder.build_intake_prompt()
    full_text = ""
    for chunk in flow._client.stream([{"role": "user", "content": prompt}], system):
        full_text += chunk

    view = _parse_view_tag(full_text)
    if view:
        flow.view = view
        flow.ccy = flow._snapshot.get(view.pair)
        flow._run_engines()
        return None  # success, no clarification needed
    else:
        return full_text  # return clarification text for display


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

# Show charts if view already extracted
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
if flow.selector_result and flow.selector_result.shortlist:
    st.divider()
    st.subheader("Structure recommendation")

    vanilla = [s for s in flow.selector_result.shortlist if not s.is_exotic]
    exotics = [s for s in flow.selector_result.shortlist if s.is_exotic]

    for item in vanilla:
        with st.container(border=True):
            st.markdown(f"**{item.rank}. {item.display_name}**")
            if item.optimised_for:
                st.caption(item.optimised_for)
            st.write(item.rationale)
            if item.caution:
                st.warning(item.caution, icon="⚠️")

    if exotics:
        with st.expander("Exotic comparison structures"):
            for item in exotics:
                st.markdown(f"**{item.display_name}**")
                if item.optimised_for:
                    st.caption(item.optimised_for)
                if item.caution:
                    st.caption(f"Note: {item.caution}")

# Clarification message (if LLM couldn't parse the view)
if "clarification" in st.session_state and st.session_state.clarification:
    st.info(st.session_state.clarification)
    st.session_state.clarification = ""

# Input
prompt = st.chat_input("Describe your trade view (pair, direction, magnitude, horizon)...")

if prompt:
    api_configured = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or (hasattr(flow._client._client, "api_key") and flow._client._client.api_key)
    )
    if not api_configured:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    with st.spinner("Reading view..."):
        clarification = _extract_view(prompt)

    if clarification:
        st.session_state.clarification = clarification
    st.rerun()
