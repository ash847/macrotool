"""
Bridges Streamlit session state with ConversationFlow.

Each page render in Streamlit is a full script re-execution. This module
ensures the flow object and message history survive across renders.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from conversation.flow import ConversationFlow


def init() -> None:
    """Called once per render. Initialises state on first load."""
    if "flow" not in st.session_state:
        st.session_state.flow = ConversationFlow()
    if "messages" not in st.session_state:
        # Each entry: {"role": str, "content": str, "charts": list | None, "metrics": dict | None}
        st.session_state.messages = []


def get_flow() -> ConversationFlow:
    return st.session_state.flow


def add_user_message(text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text})


def add_assistant_message(
    text: str,
    charts: list | None = None,
    metrics: dict | None = None,
) -> None:
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "charts": charts, "metrics": metrics}
    )


def reset() -> None:
    """Start a new conversation, preserving the flow object (avoids re-importing)."""
    st.session_state.flow.reset()
    st.session_state.messages = []
