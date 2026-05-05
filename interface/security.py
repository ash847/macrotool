"""
Shared auth/admin helpers for the Streamlit app.
"""

from __future__ import annotations

import streamlit as st


def auth_configured() -> bool:
    try:
        auth = st.secrets.get("auth", {})
        if not (auth.get("redirect_uri") and auth.get("cookie_secret")):
            return False
        # Provider config lives under [auth.<provider>] (e.g. [auth.google])
        providers = [v for k, v in auth.items() if hasattr(v, "get")]
        return any(
            p.get("client_id") and p.get("client_secret") and p.get("server_metadata_url")
            for p in providers
        )
    except Exception:
        return False


def current_user_email() -> str | None:
    try:
        if st.user.is_logged_in:
            return getattr(st.user, "email", None)
    except Exception:
        return None
    return None


def is_admin_user() -> bool:
    email = current_user_email()
    if not email:
        return False
    try:
        admins = st.secrets.get("admin_emails", [])
    except Exception:
        admins = []
    return email in admins


def assert_admin() -> None:
    require_login()
    if not is_admin_user():
        st.error("Admin access required.")
        st.stop()


def require_login() -> None:
    if not auth_configured():
        st.title("MacroTool")
        st.error("Authentication is not configured. Add the `[auth]` block and `admin_emails` to Streamlit secrets.")
        st.stop()

    if not st.user.is_logged_in:
        st.title("MacroTool")
        st.write("Sign in to continue.")
        st.button("Sign in with Google", on_click=st.login)
        st.stop()
