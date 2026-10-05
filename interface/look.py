"""Look-and-feel CSS layer.

Streamlit's theme config (.streamlit/config.toml) covers colour, font, base size and
corner radius, but not spacing or card styling. This injects interface/look.css on every
run to cover the rest. The selectors target Streamlit's own markup (data-testid
attributes) and may need a touch-up after a Streamlit upgrade — if they stop matching,
the page just loses the extra polish; nothing breaks.

MACROTOOL_LOOK_CSS overrides the file for design experiments: set it to another path to
try different CSS, or to an empty string to switch the layer off.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

_DEFAULT_CSS = Path(__file__).with_name("look.css")


def apply_look_css() -> None:
    override = os.environ.get("MACROTOOL_LOOK_CSS")
    if override is None:
        css_file = _DEFAULT_CSS
    elif override == "":
        return
    else:
        css_file = Path(override)
    if css_file.is_file():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
