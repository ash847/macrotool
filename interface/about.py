"""About and contact page shared by signed-in admins and testers."""

import json
import time
from pathlib import Path

import streamlit as st

from interface.contact import COOLDOWN_SECONDS, MAX_COMMENT_LENGTH, contact_ready, send_contact


def load_about_content():
    return json.loads(Path(__file__).with_name("about_content.json").read_text(encoding="utf-8"))


def render_about(*, user_email):
    content = load_about_content()
    st.title("About")
    with st.expander("What the tool does"):
        st.markdown(content["what"])
    with st.expander("How it does it"):
        st.markdown(content["how"])
    with st.expander("Sizing"):
        st.subheader("Fixed loss")
        st.markdown(content["sizing_fixed_loss"])
        st.subheader("Kelly")
        st.markdown(content["sizing_kelly"])
    with st.expander("Who we are"):
        if content["who"].strip():
            st.markdown(content["who"])
        else:
            st.caption("Team information will be added shortly.")
    with st.expander("Contact us"):
        try:
            config = dict(st.secrets.get("contact_email", {}))
        except Exception:
            config = {}
        ready = contact_ready(config) and bool(user_email)
        st.write("Questions, comments or feedback? Use the box below to email the MacroTool team.")
        st.caption("Your signed-in email address is included so we can reply to you. Please do not include confidential trade or account information.")
        if not ready:
            st.info("Email contact is currently unavailable. Please try again later.")
        with st.form("about_contact"):
            comments = st.text_area("Comments", max_chars=MAX_COMMENT_LENGTH, key="about_comments", height=160)
            submitted = st.form_submit_button("Send", disabled=not ready)
        if submitted:
            now = time.monotonic()
            last_sent = st.session_state.get("about_contact_last_sent")
            if last_sent is not None and now - last_sent < COOLDOWN_SECONDS:
                st.warning("Please wait a minute before sending another message.")
                return
            if not comments.strip():
                st.warning("Please enter a message before sending.")
                return
            try:
                with st.spinner("Sending…"):
                    send_contact(config, user_email, comments)
            except Exception:
                st.error("We couldn't confirm that your message was sent. Your text has been kept; please try again later.")
            else:
                st.session_state.about_contact_last_sent = now
                st.success("Your message has been accepted by our email service. Thank you.")
