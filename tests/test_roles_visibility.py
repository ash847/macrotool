"""Role resolution + Trade View output gating (interface/security.py).

Fail-restricted model: admins (in admin_emails) see everything; everyone else is a
tester with a limited surface. A local dev override (MACROTOOL_DEV) forces role/identity
for testing without Google auth.
"""

from __future__ import annotations

import importlib

import interface.security as security


def test_admin_sees_every_block():
    for block in security.TRADE_VIEW_BLOCKS:
        assert security.can_see(block, role="admin") is True


def test_tester_hidden_and_visible_blocks():
    # Hidden for testers: the analytical internals + the feedback form (pending redesign).
    for block in ("view_charts", "scores_table", "structure_evaluation", "feedback"):
        assert security.can_see(block, role="tester") is False
    # Visible for testers: market state, the priced trades, chat, testing brief.
    for block in ("market_state", "recommended_variants", "trade_chat", "testing_brief"):
        assert security.can_see(block, role="tester") is True


def test_unknown_role_sees_nothing():
    for block in security.TRADE_VIEW_BLOCKS:
        assert security.can_see(block, role="stranger") is False


def test_dev_override_forces_tester(monkeypatch):
    monkeypatch.setenv("MACROTOOL_DEV", "1")
    monkeypatch.setenv("MACROTOOL_FORCE_ROLE", "tester")
    assert security.user_role() == "tester"
    assert security.is_admin_user() is False
    assert security.current_user_email() == "dev@local"
    assert security.can_see("scores_table") is False
    assert security.can_see("market_state") is True


def test_dev_override_forces_admin(monkeypatch):
    monkeypatch.setenv("MACROTOOL_DEV", "1")
    monkeypatch.setenv("MACROTOOL_FORCE_ROLE", "admin")
    monkeypatch.setenv("MACROTOOL_DEV_EMAIL", "boss@fund.com")
    assert security.user_role() == "admin"
    assert security.is_admin_user() is True
    assert security.current_user_email() == "boss@fund.com"
    assert security.can_see("structure_evaluation") is True


def test_dev_override_off_by_default(monkeypatch):
    monkeypatch.delenv("MACROTOOL_DEV", raising=False)
    monkeypatch.setenv("MACROTOOL_FORCE_ROLE", "admin")  # ignored without MACROTOOL_DEV
    assert security._forced_role() is None


def test_fail_restricted_default_is_tester(monkeypatch):
    """A signed-in user who is NOT an admin resolves to tester (not the old mid-tier)."""
    monkeypatch.delenv("MACROTOOL_DEV", raising=False)
    monkeypatch.setattr(security, "is_admin_user", lambda: False)
    assert security.user_role() == "tester"


# ---------------------------------------------------------------------------
# Multi-provider login (Google + Auth0 magic-link) — interface/security.py
# ---------------------------------------------------------------------------

_FULL_AUTH_BLOCK = {"redirect_uri": "https://x/~/+/oauth2callback", "cookie_secret": "s" * 64}
_GOOGLE_BLOCK = {"client_id": "g_id", "client_secret": "g_secret", "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration"}
_AUTH0_BLOCK = {"client_id": "a0_id", "client_secret": "a0_secret", "server_metadata_url": "https://tenant.us.auth0.com/.well-known/openid-configuration"}


def _patch_secrets(monkeypatch, auth: dict):
    monkeypatch.setattr(security.st, "secrets", {"auth": auth})


def test_no_providers_configured_means_not_configured(monkeypatch):
    _patch_secrets(monkeypatch, {**_FULL_AUTH_BLOCK})
    assert security._configured_providers() == []
    assert security.auth_configured() is False


def test_google_only_offers_one_button(monkeypatch):
    _patch_secrets(monkeypatch, {**_FULL_AUTH_BLOCK, "google": _GOOGLE_BLOCK})
    assert security._configured_providers() == [("google", "Sign in with Google")]
    assert security.auth_configured() is True


def test_auth0_only_offers_magic_link_button(monkeypatch):
    _patch_secrets(monkeypatch, {**_FULL_AUTH_BLOCK, "auth0": _AUTH0_BLOCK})
    assert security._configured_providers() == [("auth0", "Sign in with magic link")]
    assert security.auth_configured() is True


def test_both_providers_configured_offers_both_in_order(monkeypatch):
    _patch_secrets(monkeypatch, {**_FULL_AUTH_BLOCK, "google": _GOOGLE_BLOCK, "auth0": _AUTH0_BLOCK})
    assert security._configured_providers() == [
        ("google", "Sign in with Google"),
        ("auth0", "Sign in with magic link"),
    ]


def test_incomplete_provider_block_is_not_configured(monkeypatch):
    # Missing client_secret — shouldn't count as configured.
    _patch_secrets(monkeypatch, {**_FULL_AUTH_BLOCK, "auth0": {"client_id": "x"}})
    assert security._configured_providers() == []


def test_missing_redirect_or_cookie_secret_fails_closed(monkeypatch):
    monkeypatch.setattr(security.st, "secrets", {"auth": {"google": _GOOGLE_BLOCK}})  # no redirect_uri/cookie_secret
    assert security.auth_configured() is False
