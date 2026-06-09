"""Labelled text renderers for the agent's tool results.

The agent narrates over these strings — it never sees raw objects. Renderers
label "baseline (from the standard pack)" vs a PM-requested structure so the
agent can cite the right source, and they surface only computed numbers.
"""

from __future__ import annotations

from agentic.price_structure import PricedStructure, PricingUnavailable
from agentic.standard_pack import StandardPack
from knowledge_engine.models import TradeView


def render_pack(pack: StandardPack, view: TradeView) -> str:
    """Render the deterministic standard pack as labelled text for the agent."""
    ms = pack.market_state
    direction = "Long" if view.direction == "base_higher" else "Short"
    lines: list[str] = []

    lines.append(
        f"VIEW: {direction} {view.pair} · {view.horizon_days}d"
        + (f" · target {pack.target:.4f}" if pack.target is not None else " · no target")
    )

    lines.append("\nMARKET CONTEXT (computed):")
    lines.append(f"  spot={ms.spot:.4f}  fwd={ms.fwd:.4f}  atm_vol={ms.vol:.4%}")
    if ms.with_carry:
        carry = (
            "WITH the carry — long the higher-yielding currency, so the forward drift is in "
            "your favour (you effectively sell forward at a premium to spot). Carry SUPPORTS "
            "this view."
        )
    else:
        carry = (
            "COUNTER to the carry — long the lower-yielding currency; the forward drift works "
            "against this view."
        )
    lines.append(f"  CARRY: this view is {carry} (c={ms.c:+.3f}, regime={ms.carry_regime})")
    if ms.atmfsratio is not None:
        lines.append(
            f"  carry-capture payout ratio={ms.atmfsratio:.2f} (payout of the carry-capturing "
            "spread; higher → carry capture is better rewarded. This is NOT a measure of "
            "whether carry helps or hurts your view.)"
        )
    if ms.target_z is not None:
        lines.append(f"  target_z(fwd)={ms.target_z:+.2f}σ  put_call={ms.put_call}")

    if pack.recommended:
        lines.append(
            "\nRECOMMENDED STRUCTURES (specific, priced — best variant per family by "
            "scenario-weighted P&L; use these):"
        )
        if pack.loss_budget is not None:
            lines.append(
                f"  (each variant sized so its max loss = the loss budget "
                f"{pack.loss_budget:,.2f} base ccy, on a 100-unit linear notional, R:R-derived)"
            )
        for r in pack.recommended:
            score = f"  score(wPnL)={r.score_ccy:+.2f}" if r.score_ccy is not None else ""
            lines.append(f"  {r.rank}. {r.display_name} [{r.structure_id}]{score}")
            lines.append("     " + _variant_summary(r.variant))
            ccy = _ccy_summary(r.variant)
            if ccy:
                lines.append("     " + ccy)
            if r.major_risk:
                lines.append(f"     risk (engine): {r.major_risk}")
            lines.append(f"     — {r.rationale}")
    else:
        # No representative priced (e.g. no target supplied) — fall back to families.
        lines.append("\nSTRUCTURE SHORTLIST (scored families):")
        for s in pack.selector_result.shortlist:
            tag = " (overlay)" if s.is_exotic else ""
            lines.append(f"  {s.rank}. {s.display_name} [{s.structure_id}]{tag} — {s.rationale}")
        if not pack.selector_result.shortlist:
            lines.append("  (no eligible structures for this view)")

    if pack.sizing is not None:
        sz = pack.sizing
        lines.append("\nSIZING (baseline, top structure):")
        lines.append(
            f"  kelly={sz.kelly_fraction:.3f} (conviction {sz.kelly_conviction_used})"
            f"  adjusted={sz.adjusted_kelly:.3f}"
        )
        if sz.kelly_notional_usd is not None:
            lines.append(f"  notional≈{sz.kelly_notional_usd:,.0f} (base ccy)")
        if sz.stop_level is not None:
            sd = f"{sz.stop_distance_pct:.2%}" if sz.stop_distance_pct is not None else "n/a"
            lines.append(f"  stop={sz.stop_level:.4f} (dist {sd})")

    if pack.smile_distribution is not None or pack.flat_distribution is not None:
        lines.append("\nDISTRIBUTIONS: available (smile + flat) for scenario context.")

    return "\n".join(lines)


def _ccy_summary(v) -> str | None:
    """Base-ccy notional/premium/max-loss on the standard linear-notional basis
    (sized so max loss = the R:R-derived loss budget — same as the Trade View
    variants table). None when the variant wasn't dollar-sized."""
    if v.structure_notional is None:
        return None
    return (
        f"sized: notional≈{v.structure_notional:,.0f}  premium≈{v.net_premium_ccy:,.0f}  "
        f"max_loss≈{v.max_loss_ccy:,.0f} (base ccy, linear-notional basis)"
    )


def _variant_summary(v) -> str:
    """One-line strikes + premium + payoff + RR for a PricedVariant."""
    strikes = ", ".join(f"{k:.4f}" for k in v.strikes)
    parts = [f"strikes=[{strikes}]"]
    if v.wing_ratio is not None:
        # Seagull: long 1 / short 1 / wing sold at wing_ratio units (sized to fund
        # the structure to zero cost — NOT 1x1x1).
        parts.append(f"legs=1×1×{v.wing_ratio:g} (long/short/wing)")
    if v.barrier:
        parts.append(f"barrier={v.barrier:.4f}")
    parts.append(f"premium={v.net_premium_pct:.2%}")
    if v.payoff_at_target_pct is not None:
        parts.append(f"payoff@target={v.payoff_at_target_pct:.2%}")
    if v.rr_at_target is not None:
        parts.append(f"rr={v.rr_at_target:.2f}")
    parts.append(f"max_loss={v.max_loss_pct:.2%}")
    if v.is_zero_cost:
        parts.append("zero-cost")
    return "  ".join(parts)


def render_priced_structure(ps: PricedStructure) -> str:
    """Render a single PM-requested priced structure (Tier-2 result)."""
    v = ps.variant
    lines = [f"PM-REQUESTED STRUCTURE: {ps.request.canonical}", "  " + _variant_summary(v)]
    ccy = _ccy_summary(v)
    if ccy:
        lines.append("  " + ccy)
    if ps.warnings:
        lines.append("  warnings: " + "; ".join(ps.warnings))
    return "\n".join(lines)


def render_recommended(rec) -> str:
    """Render a recommended (already-priced) structure pulled from the pack."""
    lines = [
        f"RECOMMENDED {rec.display_name} [{rec.structure_id}]",
        "  " + _variant_summary(rec.variant),
    ]
    ccy = _ccy_summary(rec.variant)
    if ccy:
        lines.append("  " + ccy)
    if rec.major_risk:
        lines.append(f"  risk (engine): {rec.major_risk}")
    lines.append(f"  — {rec.rationale}")
    return "\n".join(lines)


def render_unavailable(u: PricingUnavailable) -> str:
    return f"COULD NOT PRICE '{u.request.canonical}': {u.detail}"
