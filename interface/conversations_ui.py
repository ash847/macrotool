"""Streamlit glue for saved conversations (workspace/): store selection and the
sidebar conversation list. The Agent page itself lives in ``interface/app.py``.

Session-state keys used across both:
  ws_service  ConversationService for this browser session
  ws_warning  why persistence is degraded (or None)
  ws_conv     the open Conversation
  ws_open     pending request: a conversation id to open, or NEW
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from workspace.service import ConversationService
from workspace.store import InMemoryStore, StoreError, SupabaseStore

NEW = "__new__"


def _build_store():
    from interface.supabase_logger import get_service_client

    client = get_service_client()
    if client is None:
        return InMemoryStore(), (
            "Saved conversations are unavailable (no database connection) — this chat "
            "lasts for this browser session only."
        )
    store = SupabaseStore(client)
    try:
        store.check()
    except StoreError as e:
        try:
            from interface.debug_log import log_error
            log_error("workspace_store_check", e)
        except Exception:
            pass
        return InMemoryStore(), (
            "Saved conversations are unavailable (database tables not set up) — this "
            "chat lasts for this browser session only."
        )
    return store, None


def get_workspace(user_email: str | None) -> tuple[ConversationService, str | None]:
    """The session's ConversationService, built once per browser session."""
    email = user_email or "anonymous"
    svc = st.session_state.get("ws_service")
    if svc is None or svc.user_email != email:
        store, warning = _build_store()
        svc = ConversationService(store, email)
        st.session_state.ws_service = svc
        st.session_state.ws_warning = warning
    return svc, st.session_state.get("ws_warning")


def request_open(target: str) -> None:
    st.session_state.ws_open = target
    st.rerun()


def _short_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b")
    except Exception:
        return ""


def render_conversation_sidebar(user_email: str | None) -> None:
    """Sidebar list: New conversation + the user's recent conversations."""
    svc, _ = get_workspace(user_email)
    # The sidebar renders before the Agent page processes an open request, so a
    # pending request (just clicked) is the conversation about to be shown.
    pending = st.session_state.get("ws_open")
    current = st.session_state.get("ws_conv")
    if pending == NEW:
        current_id = None
    elif pending is not None:
        current_id = pending
    else:
        current_id = current.id if current is not None else None

    st.markdown("**Conversations**")
    if st.button("＋ New conversation", key="ws_new", use_container_width=True):
        request_open(NEW)
    try:
        convs = svc.list_conversations(limit=30)
    except StoreError:
        st.caption("Couldn't load saved conversations.")
        return
    if not convs:
        st.caption("No saved conversations yet.")
        return
    for c in convs:
        if st.button(
            c.title,
            key=f"ws_c_{c.id}",
            use_container_width=True,
            type="primary" if c.id == current_id else "secondary",
            help=f"Last active {_short_date(c.updated_at)}",
        ):
            request_open(c.id)
