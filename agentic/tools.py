"""Agent-facing tools + Python dispatch (provider-neutral schemas).

Two tools:
  - run_standard_pack (Tier 1): builds the whole deterministic pack for a view.
    The LLM supplies the *view* (pair/direction/horizon/magnitude/mode); Python
    runs build_pack. Cached — identical view inputs never recompute.
  - price_structure (Tier 2): prices one PM-named structure against the frozen
    pack. Refuses if no pack exists yet (this enforces "standard pack first").

Schemas are plain dicts (name/description/input_schema) — the provider adapters
translate them. Dispatch returns (content_text, is_error).
"""

from __future__ import annotations

from agentic.price_structure import (
    ClarificationNeeded,
    PricedStructure,
    PricingUnavailable,
    price_structure,
)
from agentic.render import render_pack, render_priced_structure, render_unavailable
from agentic.session import AgentSession
from agentic.standard_pack import build_pack
from agentic.structure_request import StructureRequestError
from knowledge_engine.models import TradeView

_DIRECTIONS = ("base_higher", "base_lower")
_CONVICTIONS = ("high", "medium", "low")
_MODES = ("recommend", "critique")
# Pairs wired into the engine (rate context, df curves). The snapshot carries more.
SUPPORTED_PAIRS = ("USDBRL", "USDTRY", "EURPLN", "GBPUSD")


TOOL_SCHEMAS = [
    {
        "name": "run_standard_pack",
        "description": (
            "Establish or change the trade VIEW and run the full deterministic engine "
            "(market state, structure scoring, sizing, distributions). Call this whenever "
            "the PM states or changes the pair, direction, tenor, target/magnitude, or mode. "
            "Returns the labelled standard pack. You must call this before pricing any "
            "structure. Direction is relative to the base currency: 'base_higher' = base "
            "appreciates, 'base_lower' = base depreciates. You provide the view only — never "
            "compute any number yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "e.g. USDBRL, USDTRY, EURPLN, GBPUSD"},
                "direction": {"type": "string", "enum": list(_DIRECTIONS)},
                "horizon_days": {"type": "integer", "description": "tenor in days"},
                "magnitude_pct": {
                    "type": "number",
                    "description": "expected move size in %, e.g. 6.0; omit if no target",
                },
                "direction_conviction": {"type": "string", "enum": list(_CONVICTIONS)},
                "mode": {"type": "string", "enum": list(_MODES)},
            },
            "required": ["pair", "direction", "horizon_days"],
        },
    },
    {
        "name": "price_structure",
        "description": (
            "Price a specific structure the PM names, against the CURRENT view's market "
            "state. Provide a short request string in the structure grammar, e.g. "
            "'34 vs 25 1x1.5', '25Δ vanilla', 'digital 10%', 'ATMF vs target 1x2'. "
            "Direction, target, weights, and strikes are supplied by the engine — you only "
            "name the structure. Requires that run_standard_pack has already been called."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "structure request, e.g. '34 vs 25 1x1.5' or 'digital 10%'",
                },
            },
            "required": ["request"],
        },
    },
]


def dispatch(session: AgentSession, name: str, args: dict) -> tuple[str, bool]:
    """Run a tool by name. Returns (content_text, is_error)."""
    try:
        if name == "run_standard_pack":
            return _run_standard_pack(session, args), False
        if name == "price_structure":
            return _price_structure(session, args)
        return f"Unknown tool '{name}'.", True
    except _ToolError as e:
        return str(e), True


class _ToolError(Exception):
    pass


def _run_standard_pack(session: AgentSession, args: dict) -> str:
    pair = args.get("pair")
    direction = args.get("direction")
    horizon_days = args.get("horizon_days")

    if pair not in SUPPORTED_PAIRS:
        raise _ToolError(
            f"Unsupported pair '{pair}'. Supported: {', '.join(SUPPORTED_PAIRS)}."
        )
    if direction not in _DIRECTIONS:
        raise _ToolError(f"direction must be one of {_DIRECTIONS}, got '{direction}'.")
    if not isinstance(horizon_days, int) or horizon_days <= 0:
        raise _ToolError("horizon_days must be a positive integer.")

    view = TradeView(
        pair=pair,
        direction=direction,
        direction_conviction=args.get("direction_conviction", "medium"),
        horizon_days=horizon_days,
        magnitude_pct=args.get("magnitude_pct"),
        mode=args.get("mode", "recommend"),
    )

    cached = session.get_cached(view)
    if cached is not None:
        session.view, session.pack = view, cached
        return render_pack(cached, view) + "\n\n(reused cached pack — view unchanged)"

    ccy = session.snapshot.get(view.pair)
    pack = build_pack(view, ccy, session.cfg, structure_constraint=session.structure_constraint)
    session.store(view, pack)
    session.view, session.pack = view, pack
    return render_pack(pack, view)


def _price_structure(session: AgentSession, args: dict) -> tuple[str, bool]:
    if session.pack is None:
        return (
            "No standard pack yet — call run_standard_pack with the PM's view first, "
            "then price the structure.",
            True,
        )

    request = args.get("request", "")
    ms = session.pack.market_state
    try:
        result = price_structure(
            request,
            ms,
            is_call=session.pack.is_call,
            target=session.pack.target,
            smile=getattr(ms, "surface", None),
        )
    except StructureRequestError as e:
        return f"Invalid structure request — {e.detail}", True

    if isinstance(result, ClarificationNeeded):
        return result.question, False  # not an error — ask the PM
    if isinstance(result, PricingUnavailable):
        return render_unavailable(result), False
    if isinstance(result, PricedStructure):
        session.priced.append(result)
        return render_priced_structure(result), False
    return "Unexpected pricing result.", True
