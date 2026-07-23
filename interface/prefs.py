"""Merged PM preference menu (Trade View intake form).

The UI asks ONE question — structure constraint and trade-management style are
intrinsically linked on the desk — and maps the answer to the two engine fields.
The engine keeps both fields (plus primary_objective, now fixed at "Balanced"), so
affinity scoring and context selection are untouched; only the UI surface shrank.

Note: with primary_objective pinned to "Balanced", the preference-aware contexts
conditioned on "Keep cost low" / "Keep risk clean" (cheap_carry, conservative_carry)
are unreachable from the UI — same status as the already-dormant contexts.
"""

from __future__ import annotations

# label → (structure_constraint, trade_management)
MERGED_PREF_OPTIONS: dict[str, tuple[str, str]] = {
    "No restriction · standard hold":       ("No restriction", "Standard hold"),
    "Avoid capped upside":                  ("Avoid capped structures", "Standard hold"),
    "May monetise early — keep it simple":  ("Avoid complex structures", "May monetise early"),
    "Clean, defendable risk":               ("Avoid tail-risky structures", "Need defendable mark-to-market"),
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
