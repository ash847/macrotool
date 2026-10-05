import smtplib
from unittest.mock import MagicMock, Mock

import pytest
from streamlit.testing.v1 import AppTest

from interface.about import load_about_content
from interface.contact import contact_ready, send_contact
from interface.navigation import allowed_page, available_pages


CONFIG = {"host": "smtp.example.com", "username": "user", "password": "test-password",
          "from_email": "app@example.com", "to_email": "owner@example.com"}


def test_tester_pages_and_stale_routes():
    assert available_pages(False) == ("Agent", "About")
    for page in available_pages(True):
        assert allowed_page(page, False) == (page if page in ("Agent", "About") else "Agent")
    assert "Trade view" in available_pages(True)
    assert allowed_page("unknown", True) == "Agent"


def test_about_copy_is_separate_from_code():
    content = load_about_content()
    assert set(content) == {"what", "how", "who", "sizing_fixed_loss", "sizing_kelly"}
    assert "50+" in content["what"]
    assert "deterministic" in content["how"] and "as of the date indicated, not live" in content["how"]


def about_app(config=None):
    app = AppTest.from_string("from interface.about import render_about\nrender_about(user_email='tester@example.com')")
    app.secrets["contact_email"] = config or {}
    return app.run()


def test_about_panels_and_unconfigured_contact():
    app = about_app()
    assert not app.exception
    assert [panel.label for panel in app.expander] == ["What the tool does", "How it does it", "Sizing", "Who we are", "Contact us"]
    assert [heading.value for heading in app.subheader] == ["Fixed loss", "Kelly"]
    content = load_about_content()
    assert "Risk 1 to make" in content["sizing_fixed_loss"]
    assert "you can lose more than the premium" in content["sizing_fixed_loss"]
    assert "total portfolio capital" in content["sizing_kelly"]
    assert "comparable sizing budget" in content["sizing_fixed_loss"]
    assert "stress-loss estimate" in content["sizing_fixed_loss"]
    assert "not a guaranteed maximum loss" in content["sizing_fixed_loss"]
    assert "shared capital input across chats" in content["sizing_fixed_loss"]
    assert "shared capital input across chats" in content["sizing_kelly"]
    assert "market-implied distribution" in content["sizing_kelly"]
    assert "long-term compounded capital growth, based on your estimated probabilities" in content["sizing_kelly"]
    assert "[here](https://en.wikipedia.org/wiki/Kelly_criterion)" in content["sizing_kelly"]
    assert app.button[0].label == "Send" and app.button[0].disabled


def test_form_sends_signed_in_user_and_prevents_immediate_repeat(monkeypatch):
    send = Mock()
    monkeypatch.setattr("interface.about.send_contact", send)
    app = about_app(CONFIG)
    app.text_area[0].set_value("Useful tool").run()
    app.button[0].click().run()
    assert not app.exception and app.success
    assert not app.info
    assert any("email the MacroTool team" in element.value for element in app.markdown)
    send.assert_called_once_with(CONFIG, "tester@example.com", "Useful tool")
    app.button[0].click().run()
    assert send.call_count == 1 and app.warning


def test_form_failure_preserves_draft_and_does_not_claim_success(monkeypatch):
    monkeypatch.setattr("interface.about.send_contact", Mock(side_effect=RuntimeError("secret provider error")))
    app = about_app(CONFIG)
    app.text_area[0].set_value("My comments").run()
    app.button[0].click().run()
    assert not app.exception and app.error and not app.success
    assert app.text_area[0].value == "My comments"
    assert "secret provider error" not in app.error[0].value


@pytest.mark.parametrize("security", ["ssl", "starttls"])
def test_mail_is_encrypted_and_recipient_is_fixed(monkeypatch, security):
    constructor = MagicMock()
    smtp = constructor.return_value.__enter__.return_value
    smtp.send_message.return_value = {}
    monkeypatch.setattr("interface.contact.smtplib.SMTP_SSL", constructor)
    monkeypatch.setattr("interface.contact.smtplib.SMTP", constructor)
    send_contact({**CONFIG, "security": security}, "tester@example.com", "Hello\nTo: not-a-recipient@example.com")
    message = smtp.send_message.call_args.args[0]
    assert message["To"] == "owner@example.com" and message["Reply-To"] == "tester@example.com"
    assert message.get_content_type() == "text/plain"
    smtp.login.assert_called_once_with("user", "test-password")
    assert smtp.starttls.call_count == (1 if security == "starttls" else 0)


@pytest.mark.parametrize("comments", ["", "   ", "x" * 5001])
def test_invalid_messages_never_connect(monkeypatch, comments):
    smtp = Mock()
    monkeypatch.setattr("interface.contact.smtplib.SMTP", smtp)
    with pytest.raises(ValueError):
        send_contact(CONFIG, "tester@example.com", comments)
    smtp.assert_not_called()


def test_unconfigured_and_header_injection_are_rejected():
    assert not contact_ready({})
    with pytest.raises(ValueError):
        send_contact({}, "tester@example.com", "hello")
    with pytest.raises(ValueError):
        send_contact(CONFIG, "tester@example.com\nBcc: attacker@example.com", "hello")


def test_refused_delivery_is_not_success(monkeypatch):
    constructor = MagicMock()
    constructor.return_value.__enter__.return_value.send_message.return_value = {"owner@example.com": (550, b"Rejected")}
    monkeypatch.setattr("interface.contact.smtplib.SMTP", constructor)
    with pytest.raises(smtplib.SMTPRecipientsRefused):
        send_contact(CONFIG, "tester@example.com", "hello")


def test_both_contact_recipients_are_in_smtp_envelope(monkeypatch):
    constructor = MagicMock()
    smtp = constructor.return_value.__enter__.return_value
    smtp.send_message.return_value = {}
    monkeypatch.setattr("interface.contact.smtplib.SMTP", constructor)
    recipients = ["ashwath.venkataraman@gmail.com", "vincent_craignou@hotmail.com"]
    send_contact({**CONFIG, "to_email": recipients}, "tester@example.com", "hello")
    assert smtp.send_message.call_args.kwargs["to_addrs"] == recipients
    assert str(smtp.send_message.call_args.args[0]["To"]) == ", ".join(recipients)


@pytest.mark.parametrize("recipients", [[], [""], [None], ["a@example.com\nBcc: b@example.com"], "a@example.com,b@example.com"])
def test_bad_recipient_configuration_disables_send(recipients):
    assert not contact_ready({**CONFIG, "to_email": recipients})
