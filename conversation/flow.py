"""
Conversation flow state machine.

States:
  INTAKE         → PM provides view in natural language; LLM extracts [VIEW: {...}] tag
  VALIDATION     → LLM contextualises view against market (auto-triggered after intake)
  STRUCTURE_REC  → LLM explains structure shortlist and scenario matrix
  SIZING         → LLM narrates Kelly sizing, stop, tranches, TPs
  ENTRY_EXIT     → LLM produces complete trade memo
  CRITIQUE       → LLM evaluates PM's supplied structure (critique mode only)
  DONE           → Follow-up Q&A with full context

Usage:
    flow = ConversationFlow()
    for chunk in flow.advance("I'm long USDBRL, high conviction..."):
        print(chunk, end="", flush=True)
    # After generator exhausted, flow.step has advanced
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Generator

from config.loader import load_config
from config.override_detector import extract_overrides
from config.resolver import resolve as resolve_config
from config.schema import ResolvedConfig, SessionOverrides
from data.snapshot_loader import load_snapshot
from data.schema import CurrencySnapshot, MarketSnapshot
from knowledge_engine.conventions import resolve as resolve_conventions
from knowledge_engine.critique_engine import evaluate_structure
from analytics.distributions import interpolate_atm_vol
from analytics.market_state import MarketState, compute_market_state
from knowledge_engine.models import (
    CritiqueOutput,
    SizingOutput,
    StructureSelectionResult,
    TradeView,
)
from knowledge_engine.sizing_engine import compute_sizing
from knowledge_engine.structure_scorer import score_structures
from analytics.distributions import (
    compute_flat_vol_distribution,
    compute_smile_distribution,
    compute_maturity_histogram,
)
from analytics.models import PriceDistribution, MaturityHistogram

from conversation.client import MacroToolClient
import conversation.tracing as _tracing
from conversation import context_builder


class Step(str, Enum):
    INTAKE = "INTAKE"
    VALIDATION = "VALIDATION"
    STRUCTURE_REC = "STRUCTURE_REC"
    SIZING = "SIZING"
    ENTRY_EXIT = "ENTRY_EXIT"
    CRITIQUE = "CRITIQUE"
    DONE = "DONE"


_VIEW_TAG = re.compile(r'\[VIEW:\s*(\{.*?\})\]', re.DOTALL)


class ConversationFlow:
    """
    Manages conversation state for a single PM session.

    Call advance(user_message) to get a generator of text chunks.
    The generator must be fully consumed before calling advance() again.
    """

    def __init__(
        self,
        api_key: str | None = None,
        snapshot: MarketSnapshot | None = None,
    ):
        self._client = MacroToolClient(api_key=api_key)
        self._snapshot: MarketSnapshot = snapshot or load_snapshot()
        self._session_span = _tracing.new_session_span("macrotool-session")

        self.step: Step = Step.INTAKE
        self.messages: list[dict] = []
        self.session_overrides = SessionOverrides()

        self.cfg: ResolvedConfig = load_config()
        self.target_rr: float | None = None
        self.view: TradeView | None = None
        self.ccy: CurrencySnapshot | None = None
        self.market_state: MarketState | None = None
        self.selector_result: StructureSelectionResult | None = None
        self.sizing: SizingOutput | None = None
        self.critique: CritiqueOutput | None = None
        self.flat_distribution: PriceDistribution | None = None
        self.smile_distribution: PriceDistribution | None = None
        self.maturity_histogram: MaturityHistogram | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advance(self, user_message: str) -> Generator[str, None, None]:
        """
        Process a PM message. Yields text chunks as they stream.

        The generator must be fully exhausted for state to be correctly updated.
        After exhaustion, self.step reflects the new state.
        """
        self.messages.append({"role": "user", "content": user_message})

        if self.step == Step.INTAKE:
            yield from self._run_intake()
        elif self.step == Step.STRUCTURE_REC:
            yield from self._run_structure_rec()
        elif self.step in (Step.DONE, Step.CRITIQUE):
            yield from self._run_followup()

    def reset(self) -> None:
        """Start a new conversation."""
        self.step = Step.INTAKE
        self.messages = []
        self.session_overrides = SessionOverrides()
        self.cfg = load_config()
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
    # Step handlers
    # ------------------------------------------------------------------

    def _run_intake(self) -> Generator[str, None, None]:
        """INTAKE: extract VIEW tag, then immediately narrate market context.

        All API calls use the original messages list (ending with the user turn)
        so we never send a conversation ending with an assistant message.
        Combined response is recorded once at the end.
        """
        system = context_builder.build_intake_prompt(self._snapshot)

        # First call: extract VIEW — stream but don't record yet
        first_text = ""
        for chunk in self._stream_traced(self.messages, system, "INTAKE_view_extraction"):
            first_text += chunk
            yield chunk

        view = _parse_view_tag(first_text)

        if view is None:
            # LLM asked for clarification — record and stay in INTAKE
            clean, _ = extract_overrides(first_text)
            self.messages.append({"role": "assistant", "content": clean.strip()})
            return

        # VIEW parsed — run knowledge engine
        self.view = view
        self.ccy = self._snapshot.get(view.pair)
        self._run_engines()
        try:
            self._session_span.update(
                tags=[view.pair, view.mode],
                metadata={"pair": view.pair, "mode": view.mode, "direction": view.direction},
            )
        except Exception:
            pass

        # Second call: validation — messages still ends with user turn (correct)
        yield "\n\n"
        validation_system = context_builder.build_validation_prompt(
            view, self.ccy, self.selector_result,
            self.flat_distribution, self.smile_distribution, self.maturity_histogram,
            target_rr=self.target_rr,
        )
        second_text = ""
        for chunk in self._stream_traced(self.messages, validation_system, "INTAKE_validation"):
            second_text += chunk
            yield chunk

        if view.mode == "critique":
            # Third call: critique — still using original messages
            yield "\n\n"
            critique_system = context_builder.build_critique_prompt(
                view, self.ccy, self.selector_result, self.sizing, self.critique
            )
            third_text = ""
            for chunk in self._stream_traced(self.messages, critique_system, "INTAKE_critique"):
                third_text += chunk
                yield chunk
            combined = first_text + "\n\n" + second_text + "\n\n" + third_text
            self.step = Step.DONE
        else:
            # Third call: structure recommendation — no PM input needed
            yield "\n\n---\n\n"
            structure_system = context_builder.build_structure_rec_prompt(
                view, self.ccy, self.selector_result, target_rr=self.target_rr,
            )
            third_text = ""
            for chunk in self._stream_traced(self.messages, structure_system, "INTAKE_structure_rec"):
                third_text += chunk
                yield chunk
            combined = first_text + "\n\n" + second_text + "\n\n" + third_text
            self.step = Step.DONE

        # Record the full combined response once
        clean, _ = extract_overrides(combined)
        clean = _VIEW_TAG.sub("", clean).strip()
        self.messages.append({"role": "assistant", "content": clean})

    def _run_structure_rec(self) -> Generator[str, None, None]:
        system = context_builder.build_structure_rec_prompt(
            self.view, self.ccy, self.selector_result, target_rr=self.target_rr,
        )
        yield from self._stream_and_record(system)
        self._apply_pref_changes()
        self.step = Step.SIZING

    def _run_sizing(self) -> Generator[str, None, None]:
        # Re-compute sizing in case prefs changed
        self._recompute_sizing()
        system = context_builder.build_sizing_prompt(
            self.view, self.ccy, self.selector_result, self.sizing
        )
        yield from self._stream_and_record(system)
        self._apply_pref_changes()
        self.step = Step.ENTRY_EXIT

    def _run_entry_exit(self) -> Generator[str, None, None]:
        system = context_builder.build_entry_exit_prompt(
            self.view, self.ccy, self.selector_result, self.sizing
        )
        yield from self._stream_and_record(system)
        self._apply_pref_changes()
        self.step = Step.DONE

    def _run_followup(self) -> Generator[str, None, None]:
        system = context_builder.build_followup_prompt(
            self.view, self.ccy, self.selector_result, self.sizing, self.critique
        )
        yield from self._stream_and_record(system)
        self._apply_pref_changes()
        # Stay in DONE

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stream_and_record(self, system: str) -> Generator[str, None, None]:
        """Stream a response and append the cleaned text to message history."""
        full_text = ""
        for chunk in self._stream_traced(self.messages, system, self.step.value):
            full_text += chunk
            yield chunk

        clean_text, _ = extract_overrides(full_text)
        clean_text = _VIEW_TAG.sub("", clean_text).strip()
        self.messages.append({"role": "assistant", "content": clean_text})

    def _stream_traced(
        self, messages: list[dict], system: str, step_name: str
    ) -> Generator[str, None, None]:
        """Stream a response, logging it as a Langfuse generation."""
        gen = _tracing.new_generation(
            name=step_name,
            model=self._client.model,
            input={"system": system, "messages": messages},
            session_span=self._session_span,
        )
        full_text = ""
        for chunk in self._client.stream(messages, system):
            full_text += chunk
            yield chunk
        gen.update(output=full_text)
        gen.end()
        _tracing.flush()

    def _apply_pref_changes(self) -> None:
        """Parse PREF_CHANGE tags from last response and re-resolve config."""
        _, overrides = extract_overrides(self._client.last_response)
        for override in overrides:
            self.session_overrides.apply(
                override.field_path,
                override.value,
                override.scope,
                override.raw_text,
            )
        if overrides:
            self.cfg = resolve_config(self.cfg, None, self.session_overrides)

    def _run_engines(self) -> None:
        """Compute MarketState, run structure scorer, sizing, distributions, and (if critique) critique engine."""
        from pricing.forwards import DEFAULT_SETTLEMENT_RATES, build_rate_context
        T = self.view.horizon_years
        r_d = DEFAULT_SETTLEMENT_RATES.get(self.view.pair, 0.043)
        rate_ctx = build_rate_context(self.ccy, T, r_d)
        atm_vol = interpolate_atm_vol(self.ccy, self.view.horizon_days)
        target: float | None = None
        if self.view.magnitude_pct is not None:
            sign = 1 if self.view.direction == "base_higher" else -1
            target = self.ccy.spot * (1 + sign * self.view.magnitude_pct / 100)
        self.market_state = compute_market_state(
            spot=rate_ctx.spot,
            fwd=rate_ctx.forward,
            vol=atm_vol,
            T=T,
            r_d=rate_ctx.r_d,
            r_f=rate_ctx.r_f,
            target=target,
            direction=self.view.direction,
        )
        self.selector_result = score_structures(self.market_state)
        top_structure = self.selector_result.shortlist[0] if self.selector_result.shortlist else None
        if top_structure:
            self.sizing = compute_sizing(self.view, self.ccy, top_structure, self.cfg)

        # Pre-compute price distributions (best-effort; non-fatal if they fail)
        try:
            self.flat_distribution = compute_flat_vol_distribution(
                self.ccy, self.view.horizon_days
            )
            self.smile_distribution = compute_smile_distribution(
                self.ccy, self.view.horizon_days
            )
            self.maturity_histogram = compute_maturity_histogram(
                self.flat_distribution, self.smile_distribution
            )
        except Exception:
            pass  # distributions are enrichment; don't break the conversation flow

        if self.view.mode == "critique" and self.view.pm_structure_description:
            self.critique = evaluate_structure(
                self.view,
                self.view.pm_structure_description,
                self.selector_result,
                self.sizing,
            )

    def _recompute_sizing(self) -> None:
        """Re-run sizing after a config change (e.g., PREF_CHANGE)."""
        if self.view and self.ccy and self.selector_result:
            top = self.selector_result.shortlist[0] if self.selector_result.shortlist else None
            if top:
                self.sizing = compute_sizing(self.view, self.ccy, top, self.cfg)



# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

def _parse_view_tag(text: str) -> TradeView | None:
    """Extract and parse a [VIEW: {...}] tag from LLM output."""
    match = _VIEW_TAG.search(text)
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    required = {"pair", "direction", "direction_conviction", "horizon_days"}
    if not required.issubset(data):
        return None

    return TradeView(
        pair=data["pair"],
        direction=data["direction"],
        direction_conviction=data["direction_conviction"],
        timing_conviction=data.get("timing_conviction", "medium"),
        horizon_days=int(data["horizon_days"]),
        magnitude_pct=data.get("magnitude_pct"),
        budget_usd=data.get("budget_usd"),
        max_loss_usd=data.get("max_loss_usd"),
        catalyst=data.get("catalyst"),
        mode=data.get("mode", "recommend"),
        pm_structure_description=data.get("pm_structure_description"),
    )
