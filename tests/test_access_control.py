from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from interface import security, access_requests


class Stopped(Exception):
    pass


@pytest.fixture
def screen(monkeypatch):
    monkeypatch.delenv("MACROTOOL_DEV", raising=False)
    claims = {"iss": "https://example.auth0.com/", "sub": "email|123", "email_verified": True,
              "email": " Tester@Example.com "}
    user = SimpleNamespace(is_logged_in=True, get=claims.get)
    provider = {"client_id": "id", "client_secret": "secret",
                "server_metadata_url": "https://example.auth0.com/.well-known/openid-configuration"}
    fake = Mock()
    fake.user = user
    fake.secrets = {"auth": {"redirect_uri": "callback", "cookie_secret": "cookie", "auth0": provider},
                    "allowed_emails": ["tester@example.com"]}
    fake.stop.side_effect = Stopped
    fake.button.return_value = False
    monkeypatch.setattr(security, "st", fake)
    return fake, claims


def test_admitted(screen):
    security.require_login()
    screen[0].stop.assert_not_called()


@pytest.mark.parametrize("allowed", [None, [], "tester@example.com", [42], ["someone@example.com"]])
def test_missing_or_invalid_list_denies(screen, allowed):
    screen[0].secrets["allowed_emails"] = allowed
    screen[0].secrets["admin_emails"] = ["tester@example.com"]
    with pytest.raises(Stopped):
        security.require_login()
    screen[0].write.assert_called_with("Sorry but looks like your email is not approved for access. Would you like access?")


@pytest.mark.parametrize("key,value", [("email_verified", False), ("email_verified", "true"),
                                     ("iss", "https://accounts.google.com"), ("sub", "google-oauth2|123"),
                                     ("email", None)])
def test_untrusted_identity_denied(screen, key, value):
    screen[1][key] = value
    with pytest.raises(Stopped):
        security.require_login()
    assert "verified email" in screen[0].error.call_args.args[0]


def test_logged_out_only_offers_magic_link(screen):
    screen[0].user.is_logged_in = False
    with pytest.raises(Stopped):
        security.require_login()
    assert screen[0].button.call_args.args[0] == "Sign in with magic link"


def test_google_config_is_not_sufficient(screen):
    auth = screen[0].secrets["auth"]
    auth["google"] = auth.pop("auth0")
    assert not security.auth_configured()


def test_request_ui_stays_blocked(screen, monkeypatch):
    screen[0].secrets["allowed_emails"] = []
    screen[0].button.return_value = True
    sender = Mock(return_value=True)
    monkeypatch.setattr(access_requests, "request_access", sender)
    with pytest.raises(Stopped):
        security.require_login()
    sender.assert_called_once_with({}, "tester@example.com")
    screen[0].success.assert_called_once()


def test_request_throttle_and_recipient(monkeypatch):
    access_requests._request_state.clear()
    sender = Mock()
    monkeypatch.setattr(access_requests, "send_contact", sender)
    assert access_requests.request_access({}, "tester@example.com")
    assert not access_requests.request_access({}, "TESTER@example.com")
    assert sender.call_args.args[0]["to_email"] == "ashwath.venkataraman@gmail.com"
    assert sender.call_args.kwargs["subject"] == "MacroTool beta access request"
    for index in range(19):
        assert access_requests.request_access({}, f"tester{index}@example.com")
    assert not access_requests.request_access({}, "another@example.com")
    access_requests._request_state.clear()


def test_failed_send_is_throttled(monkeypatch):
    access_requests._request_state.clear()
    monkeypatch.setattr(access_requests, "send_contact", Mock(side_effect=RuntimeError("private smtp error")))
    with pytest.raises(RuntimeError):
        access_requests.request_access({}, "tester@example.com")
    assert not access_requests.request_access({}, "tester@example.com")
    access_requests._request_state.clear()
