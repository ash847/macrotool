"""
Recommend mode helpers.

The full orchestration logic lives in ConversationFlow (flow.py).
This module provides utilities specific to recommend mode.
"""

from __future__ import annotations

from knowledge_engine.models import StructureSelectionResult, SizingOutput, TradeView
from data.schema import CurrencySnapshot


def format_step_summary(
    step: str,
    view: TradeView,
    selector_result: StructureSelectionResult | None,
    sizing: SizingOutput | None,
) -> str:
    """
    Returns a one-line status summary for each step.
    Used by the Streamlit UI to show progress indicators.
    """
    summaries = {
        "INTAKE": "Extracting trade view...",
        "VALIDATION": "Contextualising against market data...",
        "STRUCTURE_REC": "Presenting structure recommendations...",
        "SIZING": "Computing position sizing...",
        "ENTRY_EXIT": "Finalising trade specification...",
        "DONE": "Trade recommendation complete.",
    }
    return summaries.get(step, step)


def get_top_structure_name(selector_result: StructureSelectionResult | None) -> str | None:
    """Returns display name of the top non-exotic structure, or None."""
    if not selector_result:
        return None
    for item in selector_result.shortlist:
        if not item.is_exotic:
            return item.display_name
    return None
