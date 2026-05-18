"""Advisor chat panel for the Trade View page."""

from __future__ import annotations

import streamlit as st

from conversation.flow import ConversationFlow


def render_advisor_chat(flow: ConversationFlow) -> None:
    if not (flow.view and flow.explanation_pack_context):
        return

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
        "- For questions genuinely outside the pack's scope (e.g. macro backdrop, broader "
        "market context, timing), preface your answer with exactly: "
        "'This is outside my expertise. However, my general knowledge would suggest...' "
        "and then continue.\n\n"
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
                    max_tokens=4096,
                ):
                    _accumulated += _chunk
                    _response_placeholder.markdown(_accumulated + "▌")
                _response_placeholder.markdown(_accumulated)
                st.session_state.chat_history.append({"role": "assistant", "content": _accumulated})
            except Exception as _chat_err:
                _response_placeholder.error(f"LLM error: {_chat_err}")
