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
import re

from agentic.family_registry import resolve_family
from agentic.render import (
    render_pack,
    render_priced_structure,
    render_recommended,
    render_unavailable,
)
from agentic.session import AgentSession
from agentic.standard_pack import build_pack
from agentic.structure_request import StructureRequestError, _normalize, _strip_direction_words
from knowledge_engine.models import TradeView

# A leg token is present if the remainder has a digit, %, or a leg keyword.
_HAS_LEG = re.compile(r"[0-9%]|atmf|atm|sigma|target|tgt")

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
            "the PM states or changes the pair, tenor, target, or mode. Returns the labelled "
            "standard pack. You must call this before pricing any structure.\n"
            "How to express the target — pick ONE:\n"
            "  • TARGET LEVEL: the PM names an absolute spot level (e.g. 'USDBRL to 5.60', "
            "'targets 30'). Pass target_level=5.60 and DO NOT pass direction or "
            "magnitude_pct — the engine infers direction from the forward (you don't know "
            "the forward yet, so never guess direction from a price level).\n"
            "  • MAGNITUDE: the PM gives a percentage move (e.g. '6% higher', 'down 4%'). "
            "Pass magnitude_pct AND an explicit direction — the PM must have said higher/"
            "lower/up/down. Never invent the direction.\n"
            "  • PURE DIRECTIONAL (no target): pass direction only.\n"
            "Direction is relative to the base currency: 'base_higher' = base appreciates, "
            "'base_lower' = base depreciates. You provide the view only — never compute a "
            "number yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "e.g. USDBRL, USDTRY, EURPLN, GBPUSD"},
                "horizon_days": {"type": "integer", "description": "tenor in days"},
                "target_level": {
                    "type": "number",
                    "description": "absolute spot level the PM named, e.g. 5.60. Engine infers direction.",
                },
                "direction": {
                    "type": "string",
                    "enum": list(_DIRECTIONS),
                    "description": "required with magnitude_pct or for a pure directional view; omit with target_level",
                },
                "magnitude_pct": {
                    "type": "number",
                    "description": "percentage move size, e.g. 6.0; use only when the PM gave a %, with direction",
                },
                "direction_conviction": {"type": "string", "enum": list(_CONVICTIONS)},
                "mode": {"type": "string", "enum": list(_MODES)},
            },
            "required": ["pair", "horizon_days"],
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


def _family_only(request: str) -> str | None:
    """If the request names a family with no leg detail (e.g. "1x1.5 spread"),
    return its family id; else None. Used to fall back to the pack's recommended
    construction instead of demanding explicit strikes."""
    norm = _normalize(request)
    fam, token = resolve_family(norm)
    if not fam:
        return None
    rest = _strip_direction_words(norm.replace(token, " ", 1))
    return None if _HAS_LEG.search(rest) else fam


def _forward_for(session: AgentSession, pair: str, horizon_days: int) -> float:
    """The outright forward for pair/tenor — used to infer direction from a target
    level. Python computes it; the LLM never sees or guesses the forward."""
    from pricing.forwards import rate_context_for_snapshot

    ccy = session.snapshot.get(pair)
    return rate_context_for_snapshot(ccy, horizon_days / 365.0).forward


def _run_standard_pack(session: AgentSession, args: dict) -> str:
    pair = args.get("pair")
    horizon_days = args.get("horizon_days")
    direction = args.get("direction")
    magnitude_pct = args.get("magnitude_pct")
    target_level = args.get("target_level")

    if pair not in SUPPORTED_PAIRS:
        raise _ToolError(
            f"Unsupported pair '{pair}'. Supported: {', '.join(SUPPORTED_PAIRS)}."
        )
    if not isinstance(horizon_days, (int, float)) or horizon_days <= 0:
        raise _ToolError("horizon_days must be a positive integer.")
    horizon_days = int(horizon_days)

    if target_level is not None:
        # An absolute level: infer direction + magnitude from the FORWARD (the
        # engine knows it; the LLM does not). target>fwd → base appreciates (call);
        # target<fwd → base depreciates (put). magnitude is measured off the forward
        # so target_from_reference(fwd, dir, mag) reconstructs target_level exactly.
        fwd = _forward_for(session, pair, horizon_days)
        direction = "base_higher" if target_level >= fwd else "base_lower"
        magnitude_pct = abs(target_level / fwd - 1.0) * 100.0
    elif direction not in _DIRECTIONS:
        raise _ToolError(
            "Provide either a target_level (absolute spot level), or a direction "
            f"(one of {_DIRECTIONS}) — with magnitude_pct for a % move, or alone for a "
            "pure directional view."
        )

    view = TradeView(
        pair=pair,
        direction=direction,
        direction_conviction=args.get("direction_conviction", "medium"),
        horizon_days=horizon_days,
        magnitude_pct=magnitude_pct,
        mode=args.get("mode", "recommend"),
    )

    cached = session.get_cached(view)
    if cached is not None:
        session.view, session.pack = view, cached
        return render_pack(cached, view) + "\n\n(reused cached pack — view unchanged)"

    ccy = session.snapshot.get(view.pair)
    # NOTE (deferred — fix after the Kelly integration work on the separate worktree
    # lands): the agent uses GLOBAL scenario weights (no user_email), the session's
    # default PM preferences (Balanced / Standard hold / No restriction), and the
    # session target_rr. Trade View / Batch instead use the logged-in user's personal
    # weights profile and the PM-preference widgets, so their rankings can differ from
    # the agent's. Thread user_email + PM prefs (+ aligned target_rr) through here once
    # Kelly is merged so all surfaces rank identically.
    pack = build_pack(
        view, ccy, session.cfg,
        structure_constraint=session.structure_constraint,
        primary_objective=session.primary_objective,
        trade_management=session.trade_management,
        target_rr=session.target_rr,
        linear_notional=session.linear_notional,
    )
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
    base_ccy = session.view.pair[:3] if session.view is not None else "base ccy"

    # Family-only request (e.g. "1x1.5 spread", "the digital") → return the
    # already-priced recommended construction from the pack, don't demand strikes.
    fam_only = _family_only(request)
    if fam_only is not None:
        rec = next((r for r in session.pack.recommended if r.structure_id == fam_only), None)
        if rec is not None:
            return render_recommended(rec, base_ccy), False

    ms = session.pack.market_state
    try:
        result = price_structure(
            request,
            ms,
            is_call=session.pack.is_call,
            target=session.pack.target,
            loss_budget=session.pack.loss_budget,
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
        # Characterize the off-menu structure in the same IP-clean vocabulary as the
        # recommended set (scored against the frozen pack) so the LLM can contrast it.
        from agentic.price_structure import characterize_against_pack
        tags = characterize_against_pack(
            result.variant, result.request.family, ms,
            is_call=session.pack.is_call, target=session.pack.target,
            smile=getattr(ms, "surface", None), weights=session.pack.scenario_weights,
        )
        return render_priced_structure(result, tags, base_ccy), False
    return "Unexpected pricing result.", True
