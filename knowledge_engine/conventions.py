"""
ConventionsResolver: given a currency pair, returns a fully-typed ResolvedConventions
object. This is what the context builder injects into the LLM system prompt.

The resolver reads from knowledge/facts/*.json. It never modifies the facts —
conventions are immutable market infrastructure.
"""

from __future__ import annotations

from knowledge_engine.loader import load_convention_facts
from knowledge_engine.models import ResolvedConventions


def resolve(pair: str) -> ResolvedConventions:
    """
    Resolve all conventions for a currency pair.

    Returns a ResolvedConventions object with all relevant fields populated.
    Fields that are N/A for a given instrument type (e.g. NDF-specific fields
    for a deliverable pair) are set to None.
    """
    raw = load_convention_facts(pair)
    instrument_type = raw["instrument_type"]

    # Settlement details
    if instrument_type == "NDF":
        ndf = raw["ndf"]
        settlement_currency = ndf["settlement_currency"]
        fixing_source       = ndf["fixing_source"]
        fixing_time_local   = ndf["fixing_time_local"]
        fixing_timezone     = ndf["fixing_timezone"]
        fixing_time_london  = ndf.get("fixing_time_london")
        fixing_lag_days     = ndf["fixing_lag_days"]
        settlement_days     = ndf["settlement_days_after_fixing"]
    else:  # Deliverable
        deliv               = raw["deliverable"]
        settlement_currency = deliv["settlement_currency"]
        fixing_source       = None
        fixing_time_local   = None
        fixing_timezone     = None
        fixing_time_london  = None
        fixing_lag_days     = None
        settlement_days     = raw["spot"]["settlement_days"]

    opts = raw["options"]

    return ResolvedConventions(
        pair             = pair,
        instrument_type  = instrument_type,
        settlement_currency = settlement_currency,
        fixing_source    = fixing_source,
        fixing_time_local = fixing_time_local,
        fixing_timezone  = fixing_timezone,
        fixing_time_london = fixing_time_london,
        fixing_lag_days  = fixing_lag_days,
        settlement_days  = settlement_days,
        options_cut      = opts["cut"],
        premium_currency = opts["premium_currency"],
        delta_convention = opts["delta_convention"],
        smile_convention = opts["smile_convention"],
        liquid_tenors    = opts["liquid_tenors"],
        liquid_strikes   = opts["liquid_strikes"],
        risk_notes       = raw.get("risk_notes", []),
        carry_notes      = raw.get("carry", {}).get("notes", []),
        market_structure = raw.get("market_structure", {}),
    )


def format_for_context(conventions: ResolvedConventions) -> str:
    """
    Format resolved conventions as plain text for injection into the LLM system prompt.
    """
    lines = [f"CONVENTIONS — {conventions.pair}"]
    lines.append(f"  Instrument type: {conventions.instrument_type}")
    lines.append(f"  Settlement currency: {conventions.settlement_currency}")

    if conventions.instrument_type == "NDF":
        lines.append(f"  Fixing source: {conventions.fixing_source}")
        lines.append(f"  Fixing time: {conventions.fixing_time_local} {conventions.fixing_timezone}"
                     + (f" / {conventions.fixing_time_london} London" if conventions.fixing_time_london else ""))
        if conventions.fixing_lag_days is not None:
            lines.append(f"  Fixing lag: {conventions.fixing_lag_days} business day(s)")
        lines.append(f"  Settlement: T+{conventions.settlement_days} after fixing")
    else:
        lines.append(f"  Settlement: T+{conventions.settlement_days} (physical delivery)")

    lines.append(f"  Options cut: {conventions.options_cut}")
    lines.append(f"  Premium currency: {conventions.premium_currency}")
    lines.append(f"  Delta convention: {conventions.delta_convention}")
    lines.append(f"  Smile convention: {conventions.smile_convention}")
    lines.append(f"  Liquid tenors: {', '.join(conventions.liquid_tenors)}")
    lines.append(f"  Liquid strikes: {conventions.liquid_strikes}")

    if conventions.risk_notes:
        lines.append("  Risk notes:")
        for note in conventions.risk_notes:
            lines.append(f"    — {note}")

    return "\n".join(lines)
