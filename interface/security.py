"""
Shared auth/admin helpers for the Streamlit app.

Roles (fail-restricted): every signed-in user is either an ``admin`` (listed in the
``admin_emails`` secret) or a ``tester`` (everyone else). Testers see a limited surface
— only the Trade View and Kelly Sizing pages, and a filtered Trade View output (see
``can_see`` / ``_ROLE_VISIBILITY``). Anyone NOT explicitly elevated to admin gets the
restricted view, so a new external tester is limited by default.

Local dev override: set ``MACROTOOL_DEV=1`` to bypass Google auth entirely and force a
role via ``MACROTOOL_FORCE_ROLE`` (``admin`` | ``tester``, default ``tester``) and an
identity via ``MACROTOOL_DEV_EMAIL``. These env vars are ignored unless ``MACROTOOL_DEV``
is set, and are never set in the Streamlit Cloud deployment — production stays fail-closed.
"""

from __future__ import annotations

import os

import streamlit as st

# The distinct output blocks on the Trade View page. ``can_see(block)`` gates each one.
TRADE_VIEW_BLOCKS = (
    "testing_brief",
    "view_charts",
    "market_state",
    "scores_table",
    "recommended_variants",
    "structure_evaluation",
    "feedback",
    "trade_chat",
)

# What each non-admin role may see on Trade View. Admins see everything (short-circuited
# in ``can_see``). Testers get the "minimal" surface: market state, the recommended priced
# trades, the in-context chat, plus the testing-brief and feedback scaffolding that the
# rollout depends on — but NOT the distribution charts, the raw affinity scores table, or
# the Structure Evaluation (scenario scoring / context / P&L drivers). Edit this one set to
# retune what a tester sees.
_ROLE_VISIBILITY: dict[str, set[str]] = {
    "tester": {
        "testing_brief",
        "market_state",
        "recommended_variants",
        "trade_chat",
    },
}


def _dev_mode() -> bool:
    """Local-only auth bypass. Off unless MACROTOOL_DEV is explicitly set."""
    return os.environ.get("MACROTOOL_DEV", "").lower() in ("1", "true", "yes")


def _forced_role() -> str | None:
    """The dev-forced role, or None when not in dev mode."""
    if not _dev_mode():
        return None
    role = os.environ.get("MACROTOOL_FORCE_ROLE", "tester").lower()
    return role if role in ("admin", "tester") else "tester"


# Login providers this app knows how to offer, in display order. Each maps the
# ``[auth.<key>]`` secrets block to the button label shown on the sign-in screen.
# Add a provider by adding one entry here — require_login() picks up any that are
# configured, so e.g. Auth0 (magic-link / passwordless email, for testers without a
# Google account) appears automatically once its secrets are set, no other code change.
_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("google", "Sign in with Google"),
    ("auth0", "Sign in with magic link"),
)


def _provider_configured(provider: str) -> bool:
    try:
        block = st.secrets.get("auth", {}).get(provider, {})
        return bool(
            block.get("client_id") and block.get("client_secret") and block.get("server_metadata_url")
        )
    except Exception:
        return False


def _configured_providers() -> list[tuple[str, str]]:
    """(provider_key, button_label) for every provider with complete secrets, in
    ``_PROVIDERS`` display order."""
    return [(key, label) for key, label in _PROVIDERS if _provider_configured(key)]


def auth_configured() -> bool:
    try:
        auth = st.secrets.get("auth", {})
        if not (auth.get("redirect_uri") and auth.get("cookie_secret")):
            return False
        return bool(_configured_providers())
    except Exception:
        return False


def current_user_email() -> str | None:
    if _dev_mode():
        return os.environ.get("MACROTOOL_DEV_EMAIL", "dev@local")
    try:
        if st.user.is_logged_in:
            return getattr(st.user, "email", None)
    except Exception:
        return None
    return None


def is_admin_user() -> bool:
    forced = _forced_role()
    if forced is not None:
        return forced == "admin"
    email = current_user_email()
    if not email:
        return False
    try:
        admins = st.secrets.get("admin_emails", [])
    except Exception:
        admins = []
    return email in admins


def user_role() -> str:
    """Resolve the signed-in user's role. Fail-restricted: anyone who is not an admin
    is a tester (limited surface). The dev override wins when MACROTOOL_DEV is set."""
    forced = _forced_role()
    if forced is not None:
        return forced
    return "admin" if is_admin_user() else "tester"


def can_see(block: str, role: str | None = None) -> bool:
    """Whether ``block`` (a TRADE_VIEW_BLOCKS entry) is visible for ``role`` (defaults to
    the current user's role). Admins see everything; other roles are gated by
    ``_ROLE_VISIBILITY``."""
    role = role or user_role()
    if role == "admin":
        return True
    return block in _ROLE_VISIBILITY.get(role, set())


def personal_weights_emails() -> list[str]:
    """Allowlist of users permitted a personal scenario-weights profile. Read from
    the `personal_weights_emails` secret (like `admin_emails`). Defensive: any failure
    (e.g. no Streamlit secrets in a test/engine context) → empty list → everyone uses
    the global profile."""
    try:
        emails = st.secrets.get("personal_weights_emails", [])
        return list(emails) if emails else []
    except Exception:
        return []


def can_have_personal_weights(email: str | None) -> bool:
    return bool(email) and email in personal_weights_emails()


def assert_admin() -> None:
    require_login()
    if not is_admin_user():
        st.error("Admin access required.")
        st.stop()


def require_login() -> None:
    if _dev_mode():
        return
    if not auth_configured():
        st.title("MacroTool")
        st.error("Authentication is not configured. Add the `[auth]` block and `admin_emails` to Streamlit secrets.")
        st.stop()

    if not st.user.is_logged_in:
        st.title("MacroTool")
        st.write("Sign in to continue.")
        providers = _configured_providers()
        for i, (key, label) in enumerate(providers):
            if i > 0:
                st.caption("or")
            st.button(label, key=f"login_{key}", on_click=st.login, args=(key,))
        st.stop()
