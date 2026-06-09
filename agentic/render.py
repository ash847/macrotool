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
    lines.append(
        f"  carry c={ms.c:+.3f}  regime={ms.carry_regime}  with_carry={ms.with_carry}"
    )
    if ms.atmfsratio is not None:
        lines.append(f"  atmfsratio={ms.atmfsratio:.2f}")
    if ms.target_z is not None:
        lines.append(f"  target_z(fwd)={ms.target_z:+.2f}σ  put_call={ms.put_call}")

    if pack.recommended:
        lines.append(
            "\nRECOMMENDED STRUCTURES (specific, priced — best variant per family by "
            "scenario-weighted P&L; use these):"
        )
        for r in pack.recommended:
            score = f"  score(wPnL)={r.score_ccy:+.2f}" if r.score_ccy is not None else ""
            lines.append(f"  {r.rank}. {r.display_name} [{r.structure_id}]{score}")
            lines.append("     " + _variant_summary(r.variant))
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


def _variant_summary(v) -> str:
    """One-line strikes + premium + payoff + RR for a PricedVariant."""
    strikes = ", ".join(f"{k:.4f}" for k in v.strikes)
    parts = [f"strikes=[{strikes}]"]
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
    if v.structure_notional is not None:
        lines.append(
            f"  sized: notional≈{v.structure_notional:,.0f}, "
            f"premium≈{v.net_premium_ccy:,.0f}, max_loss≈{v.max_loss_ccy:,.0f} (base ccy)"
        )
    if ps.warnings:
        lines.append("  warnings: " + "; ".join(ps.warnings))
    return "\n".join(lines)


def render_recommended(rec) -> str:
    """Render a recommended (already-priced) structure pulled from the pack."""
    return (
        f"RECOMMENDED {rec.display_name} [{rec.structure_id}]\n  "
        + _variant_summary(rec.variant)
        + f"\n  — {rec.rationale}"
    )


def render_unavailable(u: PricingUnavailable) -> str:
    return f"COULD NOT PRICE '{u.request.canonical}': {u.detail}"
