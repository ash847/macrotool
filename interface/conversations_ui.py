"""Streamlit glue for saved conversations (workspace/): store selection and the
sidebar conversation list. The Agent page itself lives in ``interface/app.py``.

Session-state keys used across both:
  ws_service  ConversationService for this browser session
  ws_warning  why persistence is degraded (or None)
  ws_conv     the open Conversation
  ws_open     pending request: a conversation id to open, or NEW
"""

from __future__ import annotations

import time
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
    """Open a saved conversation (or NEW) on the Agent page, from any page."""
    st.session_state.ws_open = target
    st.session_state.page = "Agent"
    st.rerun()


def is_new_chat_open() -> bool:
    """True when the Agent page is showing (or about to show) a fresh, unsaved chat."""
    pending = st.session_state.get("ws_open")
    if pending is not None:
        return pending == NEW
    if st.session_state.get("ws_conv") is None:
        return True
    return st.session_state.get("ws_seq", 0) == 0


def _short_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b")
    except Exception:
        return ""


_LIST_TTL_S = 60.0


def _cached_conversations(svc) -> list:
    """The conversation list, re-queried only when the open chat changes (a new turn,
    rename, archive, switch) or after a minute — the sidebar renders on every rerun
    of every page, and Trade View reruns on each widget change."""
    conv = st.session_state.get("ws_conv")
    key = (svc.user_email, conv.id if conv else None, conv.updated_at if conv else None,
           conv.title if conv else None)
    cached = st.session_state.get("ws_list_cache")
    now = time.monotonic()
    if cached and cached[0] == key and now - cached[1] < _LIST_TTL_S:
        return cached[2]
    convs = svc.list_conversations(limit=30)
    st.session_state.ws_list_cache = (key, now, convs)
    return convs


def render_conversation_sidebar(user_email: str | None, on_agent_page: bool = True) -> None:
    """Sidebar list of the user's saved conversations (shown on every page). New chats
    start from the "New Chat" nav button."""
    svc, _ = get_workspace(user_email)
    # The sidebar renders before the Agent page processes an open request, so a
    # pending request (just clicked) is the conversation about to be shown. Nothing
    # is highlighted off the Agent page.
    pending = st.session_state.get("ws_open")
    current = st.session_state.get("ws_conv")
    if pending == NEW:
        current_id = None
    elif pending is not None:
        current_id = pending
    elif on_agent_page and current is not None:
        current_id = current.id
    else:
        current_id = None

    st.markdown("**Previous conversations**")
    try:
        convs = _cached_conversations(svc)
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
