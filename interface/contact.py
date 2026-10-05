"""Server-side plain-text contact mail with authenticated, encrypted SMTP."""

import smtplib
import ssl
from email.message import EmailMessage

MAX_COMMENT_LENGTH = 5000
COOLDOWN_SECONDS = 60


def contact_ready(config):
    required = ("host", "username", "password", "from_email")
    return all(isinstance(config.get(key), str) and config[key].strip() for key in required) and bool(contact_recipients(config))


def contact_recipients(config):
    value = config.get("to_email", [])
    recipients = [value] if isinstance(value, str) else value
    if not isinstance(recipients, list) or not recipients:
        return []
    if any(not isinstance(address, str) or not address.strip() or "@" not in address
           or any(char in address for char in "\r\n,;") for address in recipients):
        return []
    return list(dict.fromkeys(address.strip() for address in recipients))


def send_contact(config, user_email, comments):
    comments = comments.strip()
    if not comments or len(comments) > MAX_COMMENT_LENGTH:
        raise ValueError("Please enter between 1 and 5,000 characters.")
    if not contact_ready(config):
        raise ValueError("Contact delivery is not configured.")
    if not user_email or any(char in user_email for char in "\r\n"):
        raise ValueError("Please sign in before sending a message.")
    message = EmailMessage()
    message["Subject"] = "MacroTool contact message"
    message["From"] = config["from_email"]
    recipients = contact_recipients(config)
    message["To"] = ", ".join(recipients)
    message["Reply-To"] = user_email
    message.set_content(f"From signed-in user: {user_email}\n\n{comments}")
    mode = config.get("security", "starttls")
    context = ssl.create_default_context()
    if mode == "ssl":
        connection = smtplib.SMTP_SSL(config["host"], int(config.get("port", 465)), timeout=15, context=context)
    elif mode == "starttls":
        connection = smtplib.SMTP(config["host"], int(config.get("port", 587)), timeout=15)
    else:
        raise ValueError("SMTP security must be ssl or starttls.")
    with connection as smtp:
        if mode == "starttls":
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
        smtp.login(config["username"], config["password"])
        refused = smtp.send_message(message, to_addrs=recipients)
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)
