"""The Agent chat's settings strip: a one-line summary of how the active trade is
sized, and an editor for the CHAT's own sizing method / λ / R:R / preferences and
the PM's distribution for the active trade (pair + expiry). Capital W is global.

Changes are staged in the editor and applied with one button, which re-runs the
active trade and posts a labelled note (``ConversationService.apply_settings``) —
so the model never quotes figures sized under the previous settings.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import streamlit as st

from analytics.sizing import curve_key
from interface.prefs import MERGED_PREF_OPTIONS, merged_pref_fields, merged_pref_label
from workspace.settings import ChatSettings, curves_from, with_distribution

_METHODS = {"Fixed loss": "fixed_loss", "Kelly": "kelly"}


def _trade_tag(session) -> str | None:
    if session.view is None:
        return None
    return f"{session.view.pair} {session.expiry_for(session.view).strftime('%d-%b-%y')}"


def sizing_summary(settings: ChatSettings, session) -> str:
    """How the active trade is actually sized, in one line."""
    tag = _trade_tag(session)
    pack = session.pack
    if settings.sizing_method == "kelly":
        if tag and pack is not None and getattr(pack, "kelly_fallback", False):
            return (f"⚠️ **Kelly selected — no distribution for {tag}**, so this trade is "
                    f"sized **fixed-loss** (R:R {settings.target_rr:g})")
        if tag and pack is not None and pack.sizing_method == "kelly":
            return f"**Kelly λ {settings.kelly_lambda:g}** · your distribution for {tag}"
        return f"**Kelly λ {settings.kelly_lambda:g}** · distribution set per trade"
    return f"**Fixed-loss** · R:R {settings.target_rr:g}"


def render_agent_settings(
    svc,
    *,
    busy: bool,
    capital: float,
    set_capital: Callable[[float], None],
    capital_ccy: str,
    on_error: Callable[[str, Exception], None],
) -> None:
    conv = st.session_state.ws_conv
    session = st.session_state.agent_flow.session
    settings = ChatSettings.from_dict(conv.settings)
    pref_label = merged_pref_label(settings.structure_constraint, settings.trade_management)

    st.markdown(
        f"<div style='font-size:0.9rem'>Sizing: {sizing_summary(settings, session)} · "
        f"W {capital:,.0f} {capital_ccy} · {pref_label}</div>",
        unsafe_allow_html=True,
    )

    k = f"ags_{conv.id[:8]}_"          # widget keys are per chat
    with st.expander("✎ Sizing & preferences for this chat", expanded=False):
        c1, c2 = st.columns(2)
        method_label = c1.radio(
            "Size trades by", list(_METHODS),
            index=0 if settings.sizing_method == "fixed_loss" else 1,
            horizontal=True, key=k + "method",
        )
        method = _METHODS[method_label]
        new_w = c2.number_input(
            f"Capital W ({capital_ccy}) — all chats", min_value=0.0, value=float(capital),
            step=1_000_000.0, format="%.0f", key=k + "w",
        )
        c3, c4 = st.columns(2)
        rr = c3.slider(
            "Risk 1 to make (fixed-loss)", min_value=1.5, max_value=10.0, step=0.5,
            value=float(settings.target_rr), format="%.1f×", key=k + "rr",
            help="Used for fixed-loss sizing — including a Kelly chat's trades that have "
                 "no distribution yet.",
        )
        lam = c4.slider(
            "Fractional Kelly (λ)", min_value=0.1, max_value=1.0, step=0.05,
            value=float(settings.kelly_lambda), key=k + "lam",
            disabled=method != "kelly",
        )
        labels = list(MERGED_PREF_OPTIONS)
        pref = st.selectbox("Structure & management style", labels,
                            index=labels.index(pref_label), key=k + "pref")

        # The distribution belongs to the ACTIVE trade (pair + expiry), not the chat.
        distributions = dict(conv.distributions or {})
        dist_change = None                      # (key, probs|None, bins|None)
        if method == "kelly":
            if session.view is None or session.pack is None:
                st.caption("Your distribution is set per trade (pair + expiry). It appears "
                           "here once this chat has a trade.")
            else:
                from interface.kelly_inline import render_kelly_elicitation

                expiry = session.expiry_for(session.view)
                key = curve_key(session.view.pair, expiry)
                st.markdown(f"**Distribution for {_trade_tag(session)}**")
                ms = session.pack.market_state
                probs, bins, edited = render_kelly_elicitation(
                    SimpleNamespace(spot=ms.spot, fwd=ms.fwd, vol=ms.vol, T=ms.T),
                    session.pack.target, session.view.direction,
                    pair=session.view.pair, expiry=expiry,
                    key_prefix=k + "dist_", write_session=False,
                    seed_curve=curves_from(distributions).get(key),
                )
                if probs is not None:
                    dist_change = (key, probs if edited else None, bins if edited else None)

        if st.button("Apply", type="primary", key=k + "apply", disabled=busy,
                     help="Re-runs the chat's active trade under these settings."):
            sc, tm = merged_pref_fields(pref)
            new = ChatSettings(sizing_method=method, kelly_lambda=float(lam),
                               target_rr=float(rr), structure_constraint=sc,
                               trade_management=tm)
            if dist_change is not None:
                distributions = with_distribution(distributions, *dist_change)
            if abs(float(new_w) - float(capital)) > 0.5:
                set_capital(float(new_w))
                session.linear_notional = float(new_w)
            try:
                out = svc.apply_settings(
                    conv, session, new, distributions,
                    seq=st.session_state.ws_seq, persist=st.session_state.ws_seq > 0,
                )
            except Exception as e:
                on_error("agent_apply_settings", e)
                st.error(f"Couldn't apply the settings ({type(e).__name__}).")
                return
            st.session_state.ws_conv = out.conversation
            st.session_state.ws_last_settings = new.to_dict()
            if out.message:
                st.session_state.agent_chat.append(("assistant", out.message))
                st.session_state.ws_seq += 1
            st.rerun()
        if busy:
            st.caption("Available once the current answer finishes.")
