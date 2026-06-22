"""
Trade-View engine container.

Historically this module held a conversational LLM state machine (INTAKE →
VALIDATION → … → DONE). That legacy chat path has been retired — the live
conversational surface is the **Agent** page (``agentic/``). What remains here is
the deterministic engine container the Trade View / Batch screens use: it holds
the view + snapshot + config and runs the engine chain via
``agentic.standard_pack.build_pack``.

``Step`` is kept for the chart-selection helpers in ``interface/charts.py``.
``MacroToolClient`` is retained only so the sidebar "Test LLM connection" button
can reach a provider client; it is not used by the engine path.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from config.loader import load_config
from config.schema import ResolvedConfig, SessionOverrides
from data.snapshot_loader import load_snapshot
from data.schema import CurrencySnapshot, MarketSnapshot
from knowledge_engine.critique_engine import evaluate_structure
from analytics.market_state import MarketState
from knowledge_engine.models import (
    CritiqueOutput,
    SizingOutput,
    StructureSelectionResult,
    TradeView,
)
from knowledge_engine.sizing_engine import compute_sizing
from analytics.models import PriceDistribution, MaturityHistogram

from conversation.client import MacroToolClient
from agentic.standard_pack import build_pack, target_from_reference  # re-exported for back-compat


class Step(str, Enum):
    INTAKE = "INTAKE"
    VALIDATION = "VALIDATION"
    STRUCTURE_REC = "STRUCTURE_REC"
    SIZING = "SIZING"
    ENTRY_EXIT = "ENTRY_EXIT"
    CRITIQUE = "CRITIQUE"
    DONE = "DONE"


class ConversationFlow:
    """Holds Trade-View session state and runs the deterministic engine chain.

    The Streamlit Trade View screen sets the view/preferences on this object and
    calls ``_run_engines()`` directly (no LLM in the loop on that screen).
    """

    def __init__(
        self,
        api_key: str | None = None,
        snapshot: MarketSnapshot | None = None,
        provider: str = "anthropic",
        model: str | None = None,
        credentials: Any | None = None,
    ):
        # Retained only for the sidebar "Test LLM connection" button; the engine
        # path below never calls the client.
        self._client = MacroToolClient(
            api_key=api_key,
            provider=provider,
            model=model,
            credentials=credentials,
        )
        self._snapshot: MarketSnapshot = snapshot or load_snapshot()

        self.step: Step = Step.INTAKE
        self.session_overrides = SessionOverrides()

        self.cfg: ResolvedConfig = load_config()
        self.target_rr: float | None = None
        self.structure_constraint: str = "No restriction"
        self.primary_objective: str = "Balanced"
        self.trade_management: str = "Standard hold"
        self.user_email: str | None = None  # active scenario-weights profile selector
        self.view: TradeView | None = None
        self.ccy: CurrencySnapshot | None = None
        self.market_state: MarketState | None = None
        self.selector_result: StructureSelectionResult | None = None
        self.sizing: SizingOutput | None = None
        self.critique: CritiqueOutput | None = None
        self.flat_distribution: PriceDistribution | None = None
        self.smile_distribution: PriceDistribution | None = None
        self.maturity_histogram: MaturityHistogram | None = None

    def reset(self) -> None:
        """Reset to an empty session (new Trade View)."""
        self.step = Step.INTAKE
        self.session_overrides = SessionOverrides()
        self.cfg = load_config()
        self.target_rr = None
        self.structure_constraint = "No restriction"
        self.primary_objective = "Balanced"
        self.trade_management = "Standard hold"
        self.user_email = None
        self.view = None
        self.ccy = None
        self.market_state = None
        self.selector_result = None
        self.sizing = None
        self.critique = None
        self.flat_distribution = None
        self.smile_distribution = None
        self.maturity_histogram = None

    # ------------------------------------------------------------------
    # Engine path
    # ------------------------------------------------------------------

    def _run_engines(self) -> None:
        """Compute MarketState, run structure scorer, sizing, distributions, and
        (critique mode) the critique engine.

        The deterministic chain is delegated to ``agentic.standard_pack.build_pack``
        so the Trade View screen and the Agent loop share one implementation.
        """
        pack = build_pack(
            self.view,
            self.ccy,
            self.cfg,
            structure_constraint=self.structure_constraint,
        )
        self.market_state = pack.market_state
        self.selector_result = pack.selector_result
        self.sizing = pack.sizing
        self.flat_distribution = pack.flat_distribution
        self.smile_distribution = pack.smile_distribution
        self.maturity_histogram = pack.maturity_histogram

        if self.view.mode == "critique" and self.view.pm_structure_description:
            self.critique = evaluate_structure(
                self.view,
                self.view.pm_structure_description,
                self.selector_result,
                self.sizing,
            )
        else:
            self.critique = None

    def _recompute_sizing(self) -> None:
        """Re-run sizing after a config change (e.g., PREF_CHANGE)."""
        if self.view and self.ccy and self.selector_result:
            top = self.selector_result.shortlist[0] if self.selector_result.shortlist else None
            if top:
                self.sizing = compute_sizing(self.view, self.ccy, top, self.cfg)
