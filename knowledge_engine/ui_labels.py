"""Plain-English labels and tooltips — one place, read by the UI and the Agent.

The wording lives in ``knowledge/defaults/ui_labels.json`` (editable without Python).
This module is the only accessor, so a label changed there changes on screen and in
the Agent's vocabulary together. Pure: no Streamlit, no IO beyond the cached JSON load.
"""

from __future__ import annotations

from knowledge_engine.loader import load_ui_labels


def _entry(key: str) -> dict:
    try:
        return load_ui_labels()["labels"][key]
    except KeyError:
        raise KeyError(f"No UI label '{key}' in knowledge/defaults/ui_labels.json") from None


def label(key: str, **fmt) -> str:
    """The on-screen label for ``key``. ``fmt`` fills placeholders such as ``{ccy}``."""
    text = _entry(key)["label"]
    return text.format(**fmt) if fmt else text


def tip(key: str) -> str:
    """The one-line tooltip for ``key``, with the market term appended when there is
    one (the tooltip is the one place the shorthand is shown by default)."""
    e = _entry(key)
    text = e["tip"]
    return f"{text} (Market term: {e['term']}.)" if e.get("term") else text


def carry_vs_vol_label(regime: int) -> str:
    """Plain word for the engine's carry regime (0 / 1 / 2)."""
    return load_ui_labels()["carry_vs_vol_values"][str(int(regime))]


def glossary_text() -> str:
    """The TERMS block for the Agent's system prompt: every label that stands for a
    market term, as 'Label (market term: X): meaning'. Built from the same entries as
    the UI so the chat and the screen cannot drift apart."""
    lines = []
    for e in load_ui_labels()["labels"].values():
        if not e.get("term"):
            continue
        name = e["label"].split(" (")[0].replace("{ccy}", "ccy")
        lines.append(f"- {name} (market term: {e['term']}): {e['tip']}")
    return "\n".join(lines)


def audience_rules_text() -> str:
    """The AUDIENCE AND STYLE block for the Agent's system prompt."""
    return "\n".join(f"- {rule}" for rule in load_ui_labels()["audience_rules"])
