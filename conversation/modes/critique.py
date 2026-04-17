"""
Critique mode helpers.

The full orchestration logic lives in ConversationFlow (flow.py).
This module provides utilities specific to critique mode.
"""

from __future__ import annotations

from knowledge_engine.models import CritiqueOutput


VERDICT_LABELS = {
    "appropriate_for_view": "✓ Appropriate for view",
    "suboptimal_but_defensible": "~ Suboptimal but defensible",
    "materially_misaligned": "✗ Materially misaligned",
}


def format_verdict_label(verdict: str) -> str:
    """Returns a human-readable verdict label for display."""
    return VERDICT_LABELS.get(verdict, verdict)


def critique_summary(critique: CritiqueOutput) -> dict:
    """
    Returns a summary dict suitable for Streamlit metric display.
    """
    return {
        "verdict": format_verdict_label(critique.verdict),
        "pm_structure": critique.pm_structure,
        "recommended_alternative": critique.recommended_alternative or "—",
        "dimension_scores": critique.dimension_scores,
    }
