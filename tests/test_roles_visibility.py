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
    # Hidden for testers: the analytical internals.
    for block in ("view_charts", "scores_table", "structure_evaluation"):
        assert security.can_see(block, role="tester") is False
    # Visible for testers: market state, the priced trades, chat, + rollout scaffolding.
    for block in ("market_state", "recommended_variants", "trade_chat", "testing_brief", "feedback"):
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
