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
from agentic.shortlist import INSPECTION_DISPLAYS, inspection_tables, render_inspection_tables, render_shortlist, shortlist_reference
from agentic.structure_request import StructureRequestError, _normalize, _strip_direction_words
from knowledge_engine.models import TradeView
from agentic.dashboard import FIELD_LABELS, LAYOUTS
from knowledge_engine.tail_policy import TAIL_CONSTRAINTS, assign_variant_tails, tail_exclusion_reason, tail_risk_text

# A leg token is present if the remainder has a digit, %, or a leg keyword.
_HAS_LEG = re.compile(r"[0-9%]|atmf|atm|sigma|target|tgt")

_DIRECTIONS = ("base_higher", "base_lower")
_CONVICTIONS = ("high", "medium", "low")
_MODES = ("recommend", "critique")
# Supported pairs are NOT hardcoded: the run_standard_pack gate below checks the loaded
# market snapshot (session.snapshot.currencies), and the agent's system prompt is built
# per-session from that same list — so adding a pair to the snapshot exposes it with no
# code change. Every snapshot pair is priceable (all have a USD/EUR/GBP base + df curve).


TOOL_SCHEMAS = [
    {
        "name": "set_tail_constraint",
        "description": "Set or clear the directional tail preference without changing the view. Rebuilds recommendations under the constraint. lower_spot/higher_spot are absolute spot directions; against_view/with_view are resolved by Python using the current view direction. Unknown risks fail closed. Use none to clear only this directional preference; any separate no-tails setting still applies.",
        "input_schema": {
            "type": "object", "properties": {"tail_constraint": {"type": "string", "enum": list(TAIL_CONSTRAINTS)}},
            "required": ["tail_constraint"], "additionalProperties": False,
        },
    },
    {
        "name": "inspect_recommendations",
        "description": "Read or compare the already-priced recommendations without repricing. Use ranks for 'explain #2', 'compare 1 and 3', or risk/sizing details. Use family for 'why was vanilla absent?'. Supply the SHORTLIST REFERENCE from the relevant pack; stale references are rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "shortlist_ref": {"type": "string"},
                "ranks": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1, "maxItems": 5, "uniqueItems": True},
                "family": {"type": "string"},
                "layout": {"type": "string", "enum": list(LAYOUTS), "description": "With display=dashboard: trades_as_columns for one transposed table (default), or trades_as_rows."},
                "fields": {"type": "array", "items": {"type": "string", "enum": list(FIELD_LABELS)}, "minItems": 1, "maxItems": len(FIELD_LABELS), "uniqueItems": True, "description": "With display=dashboard: requested fields in presentation order. Omit for the standard summary, including one loss budget for the actual sized trade, directional tails and top contributor. No cell values are accepted."},
                "driver_count": {"type": "integer", "minimum": 1, "maximum": 3, "description": "With display=dashboard: how many retained contributors/detractors per trade; default 1."},
                "display": {
                    "type": "string", "enum": list(INSPECTION_DISPLAYS),
                    "description": "Python-rendered output: dashboard for one combined table with configurable fields, layout and driver_count; trade_details for the legacy comparison; contributors or detractors for one side; drivers for both sides; both for separate details and driver tables; none for prose-only inspection. Select explicitly even if the facts are already in chat. Do not supply numerical cells.",
                },
            },
            "required": ["shortlist_ref", "display"],
            "additionalProperties": False,
        },
    },
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
                "pair": {"type": "string", "description": "any pair in the loaded market data (e.g. USDBRL, EURUSD, USDJPY); the tool returns the available set if not present"},
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
                "tail_constraint": {"type": "string", "enum": list(TAIL_CONSTRAINTS), "description": "Optional directional tail constraint. Omit to preserve the chat's existing preference. none explicitly clears it."},
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
        if name == "set_tail_constraint":
            return _set_tail_constraint(session, args), False
        if name == "inspect_recommendations":
            return _inspect_recommendations(session, args), False
        if name == "run_standard_pack":
            return _run_standard_pack(session, args), False
        if name == "price_structure":
            return _price_structure(session, args)
        return f"Unknown tool '{name}'.", True
    except _ToolError as e:
        return str(e), True


class _ToolError(Exception):
    pass


def _inspection_family(pack, requested):
    known_families = {rec.structure_id for rec in pack.recommended}
    known_families.update(item.structure_id for item in pack.selector_result.shortlist)
    canonical = requested.strip().lower().replace(" ", "_") if isinstance(requested, str) else None
    family = canonical if canonical in known_families else _family_only(requested) if isinstance(requested, str) else None
    if family is None:
        raise _ToolError("Unknown family reference; ask the PM to name the structure.")
    return family


def inspection_ranks(pack, args):
    if "family" in args:
        family = _inspection_family(pack, args["family"])
        return [rec.rank for rec in pack.recommended if rec.structure_id == family]
    return args.get("ranks", [rec.rank for rec in pack.recommended[:5]])


def requested_inspection_tables(pack, args):
    try:
        return inspection_tables(args.get("display", "none"), inspection_ranks(pack, args),
                                 layout=args.get("layout"), fields=args.get("fields"), driver_count=args.get("driver_count"))
    except ValueError as error:
        raise _ToolError(str(error)) from error


def _inspect_recommendations(session: AgentSession, args: dict) -> str:
    pack, view = session.pack, session.view
    if pack is None or view is None:
        raise _ToolError("No standard pack yet — establish the view first.")
    reference = shortlist_reference(pack, view)
    if args.get("shortlist_ref") != reference:
        raise _ToolError("That shortlist reference is no longer current. Ask which displayed shortlist the PM means; do not reinterpret old ranks against the current list.")
    display = args.get("display", "none")
    if display not in INSPECTION_DISPLAYS:
        raise _ToolError("Choose a supported display: " + ", ".join(INSPECTION_DISPLAYS))
    if "family" in args and "ranks" in args:
        raise _ToolError("Provide ranks or a family, not both.")
    if set(args) - {"shortlist_ref", "display", "ranks", "family", "layout", "fields", "driver_count"}:
        raise _ToolError("Unsupported parameters: choose presentation fields, never supply numerical cell values.")
    try:
        inspection_tables(display, [], layout=args.get("layout"), fields=args.get("fields"), driver_count=args.get("driver_count"))
    except ValueError as error:
        raise _ToolError(str(error)) from error
    if "family" in args:
        family = _inspection_family(pack, args["family"])
        matches = [rec for rec in pack.recommended if rec.structure_id == family]
        if matches:
            table = render_inspection_tables(pack, view, requested_inspection_tables(pack, args))
            return f"SHORTLIST REFERENCE: {reference}\nMultiple variants may share a family; refer to exact ranks.\n{table}\n" + "\n\n".join(
                f"Engine rank {rec.rank}: {'in' if rec in pack.recommended[:5] else 'outside'} the displayed top five.\n"
                + render_recommended(rec, view.pair[:3]) for rec in matches
            )
        exclusions = [item for item in pack.tail_exclusions if item["structure_id"] == family]
        if exclusions:
            return "Excluded by the active directional tail constraint:\n" + "\n".join(
                f"{item['variant']}: {item['reason']}" for item in exclusions
            )
        if any(item.structure_id == family for item in pack.selector_result.shortlist):
            return "The family was shortlisted, but no priced recommendation was retained. The detailed pricing reason was not recorded in this pack; do not invent it."
        return "The family was not retained by the engine's eligibility/scoring stage. The detailed exclusion reason was not recorded in this pack; do not invent it."
    ranks = args.get("ranks", [rec.rank for rec in pack.recommended[:5]])
    if not isinstance(ranks, list) or not 1 <= len(ranks) <= 5 or any(type(rank) is not int or rank < 1 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise _ToolError("Provide one to five distinct positive integer ranks.")
    selected = [rec for rec in pack.recommended if rec.rank in ranks]
    if len(selected) != len(ranks):
        raise _ToolError("One or more ranks do not exist in this shortlist. Do not guess a replacement trade.")
    requested = requested_inspection_tables(pack, args)
    return (
        f"SHORTLIST REFERENCE: {reference}\nAlready-priced trades; no recomputation.\n"
        + (render_inspection_tables(pack, view, requested)
           if display != "none" else render_shortlist(pack, view, ranks))
        + "\n\n" + "\n\n".join(render_recommended(rec, view.pair[:3]) for rec in selected)
    )


def _family_only(request: str) -> str | None:
    """If the request names a family with no leg detail (e.g. "1x1.5 spread"),
    return its family id; else None. Used to fall back to the pack's recommended
    construction instead of demanding explicit strikes."""
    norm = _normalize(request)
    if norm == "linear":
        return "linear"
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


def _kelly_curve_kwargs(session: AgentSession, view: TradeView) -> dict:
    """The PM's stated curve for this trade only (else none → market distribution)."""
    curve = session.stated_curve_for(view)
    return {"kelly_probs": curve[0], "kelly_bins": curve[1]} if curve else {}


def _set_tail_constraint(session, args):
    from knowledge_engine.tail_policy import tail_constraint_label

    constraint = args.get("tail_constraint")
    if constraint not in TAIL_CONSTRAINTS:
        raise _ToolError("Choose a supported directional tail constraint.")
    if session.view is None:
        session.tail_constraint = constraint
        return tail_constraint_label(constraint) + ". This will apply when a view is established."
    view = session.view
    return _run_standard_pack(session, {
        "pair": view.pair, "horizon_days": view.horizon_days,
        "direction": view.direction, "magnitude_pct": view.magnitude_pct,
        "direction_conviction": view.direction_conviction, "mode": view.mode,
        "tail_constraint": constraint,
    })


def _run_standard_pack(session: AgentSession, args: dict) -> str:
    pair = args.get("pair")
    horizon_days = args.get("horizon_days")
    direction = args.get("direction")
    magnitude_pct = args.get("magnitude_pct")
    target_level = args.get("target_level")

    available = tuple(session.snapshot.currencies.keys())
    if pair not in available:
        raise _ToolError(
            f"Unsupported pair '{pair}'. Supported: {', '.join(available)}."
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

    from dataclasses import replace
    original_session = session
    constraint = args.get("tail_constraint", session.tail_constraint)
    if constraint not in TAIL_CONSTRAINTS:
        raise _ToolError("Choose a supported directional tail constraint.")
    session = replace(session, tail_constraint=constraint)
    cached = session.get_cached(view)
    if cached is not None:
        original_session.view, original_session.pack = view, cached
        original_session.tail_constraint = constraint
        return render_pack(cached, view) + "\n\n(reused cached pack — view unchanged)"

    ccy = session.snapshot.get(view.pair)
    # Settings are the CHAT's own (sizing regime, λ, R:R, preferences — set by the
    # chat's settings strip) + the PM's personal scenario-weights profile (user_email).
    pack = build_pack(
        view, ccy, session.cfg,
        structure_constraint=session.structure_constraint,
        tail_constraint=session.tail_constraint,
        primary_objective=session.primary_objective,
        trade_management=session.trade_management,
        target_rr=session.target_rr,
        linear_notional=session.linear_notional,
        sizing_method=session.sizing_method,
        kelly_lambda=session.kelly_lambda,
        user_email=session.user_email,
        **_kelly_curve_kwargs(session, view),
    )
    session.store(view, pack)
    original_session.view, original_session.pack = view, pack
    original_session.tail_constraint = constraint
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
            if session.structure_constraint == "Avoid tail-risky structures" and getattr(rec.variant, "can_lose_beyond_premium", None) is not False:
                return "Excluded by the active no-tails preference: this construction is not classified as unable to lose beyond premium paid.", False
            return render_recommended(rec, base_ccy), False

    ms = session.pack.market_state
    try:
        result = price_structure(
            request,
            ms,
            is_call=session.pack.is_call,
            target=session.pack.target,
            loss_budget=session.pack.loss_budget,
            linear_notional=session.linear_notional,
            smile=getattr(ms, "surface", None),
            # Same regime as the pack: Kelly on the PM's stated distribution for this
            # trade when the pack was Kelly-sized, else fixed-loss.
            sizing_spec=getattr(session.pack, "sizing_spec", None),
            structure_constraint=session.structure_constraint,
        )
    except StructureRequestError as e:
        return f"Invalid structure request — {e.detail}", True

    if isinstance(result, ClarificationNeeded):
        return result.question, False  # not an error — ask the PM
    if isinstance(result, PricingUnavailable):
        return render_unavailable(result), False
    if isinstance(result, PricedStructure):
        from dataclasses import replace

        trace = result.variant.sizing_trace
        if trace is not None:
            context = {
                "requested_method": session.sizing_method,
                "fallback_reason": "No stated Kelly distribution for this pair/expiry" if session.pack.kelly_fallback else None,
            }
            if trace.effective_method == "fixed_loss":
                origin = next((
                    rec.variant.sizing_trace for rec in session.pack.recommended
                    if rec.variant.sizing_trace is not None and rec.variant.sizing_trace.budget_distance is not None
                ), None)
                if origin is not None:
                    for name in ("budget_distance", "budget_reference", "budget_target", "budget_input_rr"):
                        context[name] = getattr(origin, name)
            result.variant.sizing_trace = replace(trace, **context)
        session.priced.append(result)
        from agentic.structure_request import to_variant_dict
        assign_variant_tails(result.request.family, result.variant, session.pack.is_call,
                             construction=to_variant_dict(result.request))
        tail_warning = tail_exclusion_reason(result.variant, session.pack.resolved_tail_constraint, session.view.direction)
        # Characterize the off-menu structure in the same IP-clean vocabulary as the
        # recommended set (scored against the frozen pack) so the LLM can contrast it.
        from agentic.price_structure import characterize_against_pack
        tags = characterize_against_pack(
            result.variant, result.request.family, ms,
            is_call=session.pack.is_call, target=session.pack.target,
            smile=getattr(ms, "surface", None), weights=session.pack.scenario_weights,
        )
        warning = f"CONFLICT WITH ACTIVE TAIL CONSTRAINT: {tail_warning}. Priced only for inspection, not an eligible recommendation.\n" if tail_warning else ""
        return warning + tail_risk_text(result.variant) + "\n" + render_priced_structure(result, tags, base_ccy), False
    return "Unexpected pricing result.", True
