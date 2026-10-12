"""PM menu mapped to stable engine fields.

Capped-upside avoidance is vanilla-only; tail avoidance requires a verified finite
loss bound. Early monetisation is retired from the menu, not erased from saved chats.
Primary objective remains Balanced; management overlay mappings remain compatible.
"""

from __future__ import annotations

# label → (structure_constraint, trade_management)
MERGED_PREF_OPTIONS: dict[str, tuple[str, str]] = {
    "No restriction · standard hold":       ("No restriction", "Standard hold"),
    "Avoid capped upside":                  ("Avoid capped structures", "Standard hold"),
    "Avoid tails — defined loss only":     ("Avoid tail-risky structures", "Need defendable mark-to-market"),
}

DEFAULT_MERGED_PREF = "No restriction · standard hold"

# The engine value primary_objective is fixed to now that the widget is gone.
FIXED_PRIMARY_OBJECTIVE = "Balanced"


def merged_pref_fields(label: str) -> tuple[str, str]:
    """(structure_constraint, trade_management) for a menu label; unknown labels fall
    back to the unrestricted default."""
    return MERGED_PREF_OPTIONS.get(label, MERGED_PREF_OPTIONS[DEFAULT_MERGED_PREF])


def merged_pref_label(structure_constraint: str, trade_management: str) -> str:
    """Reverse-map engine fields to a menu label (for seeding the widget from
    session state); non-matching combinations land on the default."""
    for label, (sc, tm) in MERGED_PREF_OPTIONS.items():
        if sc == structure_constraint and tm == trade_management:
            return label
    return DEFAULT_MERGED_PREF
