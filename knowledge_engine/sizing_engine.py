"""
Sizing engine — Kelly-based position sizing with vol-regime adjustment.

Computes:
  1. Adjusted Kelly fraction  (conviction × vol-regime multiplier)
  2. Base notional            (from budget, max_loss, or direct)
  3. Kelly-adjusted notional  (base × adjusted_kelly)
  4. Vol-derived stop level   (ATR multiple from current spot)
  5. Tranche schedule         (if timing conviction is low/medium)
  6. Take-profit levels       (if magnitude_pct is known)

All parameters come from ResolvedConfig (which the session override system can
modify on the fly). No hardcoded values here.

Premium estimation for notional-from-budget:
    ATM call premium ≈ σ × √T × spot × 0.4  (Δ≈0.5, simplified Black approximation)
    This is intentionally rough — it gives a ballpark notional that the PM can adjust.
    The exact premium is computed by the pricing engine in Step 2.
"""

from __future__ import annotations

import math

from config.schema import ResolvedConfig
from data.schema import CurrencySnapshot
from knowledge_engine.models import (
    SizingOutput,
    StructureShortlistItem,
    TakeProfitLevel,
    TradeView,
)


def compute_sizing(
    view: TradeView,
    snapshot: CurrencySnapshot,
    top_structure: StructureShortlistItem,
    cfg: ResolvedConfig,
) -> SizingOutput:
    """
    Compute full sizing for the top recommended structure.

    Args:
        view:          The PM's structured trade view.
        snapshot:      Current market snapshot for the pair.
        top_structure: The highest-ranked structure from the selector.
        cfg:           Resolved config (all sizing parameters live here).

    Returns:
        SizingOutput with all sizing components and explicit reasoning.
    """
    notes: list[str] = []
    sizing_cfg = cfg.sizing

    # ------------------------------------------------------------------
    # 1. Kelly fraction (conviction-adjusted)
    # ------------------------------------------------------------------
    kelly_fraction  = cfg.get_kelly_fraction(view.direction_conviction)
    kelly_source    = cfg.source_trace.get("sizing.kelly.default_fraction", "default")
    notes.append(
        f"Kelly fraction: {kelly_fraction:.2f} — {view.direction_conviction} conviction "
        f"({_kelly_rationale(view.direction_conviction, kelly_fraction, sizing_cfg.kelly)})"
    )

    # ------------------------------------------------------------------
    # 2. Vol adjustment (1.0 until regime classification is re-introduced)
    # ------------------------------------------------------------------
    vol_adj = 1.0
    adjusted_kelly = kelly_fraction * vol_adj

    # ------------------------------------------------------------------
    # 3. Base notional from budget/max_loss/direct
    # ------------------------------------------------------------------
    base_notional, budget_type = _estimate_base_notional(view, snapshot, cfg)
    if base_notional:
        notes.append(f"Base notional estimate: ${base_notional:,.0f} ({budget_type})")

    kelly_notional = (base_notional * adjusted_kelly) if base_notional else None
    if kelly_notional:
        notes.append(f"Kelly-adjusted notional: ${kelly_notional:,.0f} (= ${base_notional:,.0f} × {adjusted_kelly:.3f})")

    # ------------------------------------------------------------------
    # 4. Vol-derived stop level
    # ------------------------------------------------------------------
    atm_vol    = snapshot.get_atm_vol("1M") or 0.15  # 1M ATM, decimal
    spot       = snapshot.spot
    atr_mult   = sizing_cfg.stop.atr_multiple
    daily_vol  = atm_vol / math.sqrt(252)
    daily_range = spot * daily_vol
    stop_distance = daily_range * atr_mult

    if view.direction == "base_higher":
        stop_level = spot - stop_distance   # stop below current spot for long base
    else:
        stop_level = spot + stop_distance   # stop above current spot for short base

    stop_dist_pct = (stop_distance / spot) * 100
    notes.append(
        f"Stop level: {stop_level:.4f} ({stop_dist_pct:.2f}% from spot) — "
        f"vol-derived: {atr_mult:.1f}× daily range "
        f"(ATM vol={atm_vol*100:.1f}%, daily range≈{daily_range:.4f})"
    )

    # ------------------------------------------------------------------
    # 5. Tranche schedule
    # ------------------------------------------------------------------
    tranche_schedule = None
    tranche_count = None
    if top_structure.sizing_modifier == "tranche_entry" or view.timing_conviction in ("low", "medium"):
        sched = cfg.get_tranche_schedule(view.timing_conviction)
        tranche_schedule = sched.weights
        tranche_count    = sched.count
        pct_str = ", ".join(f"{w*100:.0f}%" for w in sched.weights)
        notes.append(
            f"Tranche entry ({view.timing_conviction} timing conviction): "
            f"{sched.count} tranches [{pct_str}]"
        )

    # ------------------------------------------------------------------
    # 6. Take-profit levels
    # ------------------------------------------------------------------
    tp_levels = _compute_take_profits(view, spot, cfg)
    if tp_levels:
        for tp in tp_levels:
            notes.append(
                f"TP {tp.at_pct_of_target*100:.0f}% of target: "
                f"reduce {tp.reduce_by_pct*100:.0f}%"
                + (f" (spot {tp.target_spot:.4f})" if tp.target_spot else "")
            )

    return SizingOutput(
        kelly_fraction         = kelly_fraction,
        kelly_conviction_used  = view.direction_conviction,
        kelly_source           = kelly_source,
        vol_adjustment         = vol_adj,
        adjusted_kelly         = adjusted_kelly,
        base_notional_usd      = base_notional,
        kelly_notional_usd     = kelly_notional,
        budget_type            = budget_type,
        stop_level             = stop_level,
        stop_distance_pct      = stop_dist_pct,
        daily_range_est        = daily_range,
        tranche_schedule       = tranche_schedule,
        tranche_count          = tranche_count,
        take_profit_levels     = tp_levels,
        notes                  = notes,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_base_notional(
    view: TradeView,
    snapshot: CurrencySnapshot,
    cfg: ResolvedConfig,
) -> tuple[float | None, str]:
    """
    Estimate base notional from the PM's budget/max_loss/direct input.

    Returns (notional, budget_type_str).
    """
    if view.notional_usd is not None:
        return view.notional_usd, "direct"

    spot = snapshot.spot
    atm_vol = snapshot.get_atm_vol("1M") or 0.15
    T = view.horizon_years

    # Simplified ATM call premium estimate: ~0.4 × σ × √T (as fraction of spot)
    # This is the rough formula for an ATM option with Δ≈0.5.
    atm_prem_fraction = 0.4 * atm_vol * math.sqrt(T)

    if view.budget_usd is not None:
        # Budget = premium paid → notional = budget / (premium per unit of spot)
        # premium per unit of notional ≈ atm_prem_fraction
        if atm_prem_fraction > 0:
            notional = view.budget_usd / atm_prem_fraction
        else:
            notional = view.budget_usd * 20   # fallback if T→0
        return notional, "from_budget"

    if view.max_loss_usd is not None:
        # For options: max loss = premium paid ≈ same as budget case
        if atm_prem_fraction > 0:
            notional = view.max_loss_usd / atm_prem_fraction
        else:
            notional = view.max_loss_usd * 20
        return notional, "from_max_loss"

    return None, "unknown"


def _compute_take_profits(
    view: TradeView,
    spot: float,
    cfg: ResolvedConfig,
) -> list[TakeProfitLevel]:
    """Compute take-profit levels. Returns empty list if no target level known."""
    if not view.has_target_level:
        return []

    target_move = spot * view.magnitude_pct / 100
    if view.direction == "base_higher":
        target_spot = spot + target_move
    else:
        target_spot = spot - target_move

    tp_cfg = cfg.sizing.take_profit
    levels: list[TakeProfitLevel] = []

    for scale_name, scale in [("scale_1", tp_cfg.scale_1), ("scale_2", tp_cfg.scale_2)]:
        frac = scale.at_pct_of_target
        if view.direction == "base_higher":
            tp_spot = spot + target_move * frac
        else:
            tp_spot = spot - target_move * frac

        levels.append(TakeProfitLevel(
            at_pct_of_target = frac,
            reduce_by_pct    = scale.reduce_position_by,
            target_spot      = tp_spot,
            note             = f"{scale.reduce_position_by*100:.0f}% reduction",
        ))

    # Runner
    levels.append(TakeProfitLevel(
        at_pct_of_target = 1.0,
        reduce_by_pct    = 0.0,
        target_spot      = target_spot,
        note             = tp_cfg.runner_note,
    ))

    return levels


def _kelly_rationale(conviction: str, fraction: float, kelly_cfg) -> str:
    match conviction:
        case "high":
            return (f"{fraction:.2f} = high-conviction fraction. "
                    f"Full Kelly ({kelly_cfg.max_fraction:.2f}) not used — "
                    "variance reduction outweighs marginal EV.")
        case "low":
            return (f"{fraction:.2f} = low-conviction fraction. "
                    "Reduced size reflects uncertainty on direction.")
        case _:
            return (f"{fraction:.2f} = default half-Kelly. "
                    "Standard risk-adjusted sizing for medium conviction.")


def format_sizing_for_context(sizing: SizingOutput) -> str:
    """Format sizing output as plain text for LLM context injection."""
    lines = ["SIZING OUTPUT"]
    for note in sizing.notes:
        lines.append(f"  {note}")

    if sizing.tranche_schedule:
        lines.append(
            f"  Tranche weights: {[f'{w*100:.0f}%' for w in sizing.tranche_schedule]}"
        )

    return "\n".join(lines)
