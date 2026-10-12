"""Process-wide throttling for verified users requesting beta access."""

import hashlib
import threading
import time

import streamlit as st

from interface.contact import send_contact


@st.cache_resource
def _request_state():
    return threading.Lock(), {}


def request_access(config, email):
    identity = hashlib.sha256(email.strip().lower().encode()).hexdigest()
    lock, attempts = _request_state()
    now = time.monotonic()
    with lock:
        expired = [key for key, timestamp in attempts.items() if now - timestamp >= 86400]
        for key in expired:
            del attempts[key]
        if identity in attempts or sum(now - timestamp < 3600 for timestamp in attempts.values()) >= 20:
            return False
        attempts[identity] = now
    request_config = {**config, "to_email": config.get("access_request_to_email", "ashwath.venkataraman@gmail.com")}
    send_contact(
        request_config,
        email,
        "This verified user has requested access to MacroTool.\n"
        "To approve, add their email to allowed_emails in Streamlit secrets.\n"
        "No access has been granted automatically.",
        subject="MacroTool beta access request",
    )
    return True
