"""Agent-facing financial meanings, independent of legacy sizing fields."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TradeEconomics:
    premium_pct: float
    premium_direction: Literal["paid", "received", "zero_cost"]
    sizing_loss_pct: float | None
    sizing_loss_method: str
    sizing_reference: float | None
    loss_budget: float | None
    contractual_loss_status: Literal["bounded", "unbounded", "unknown"]
    contractual_max_loss_pct: float | None
    contractual_loss_reason: str
    target: float | None
    evaluation_days: int
    expiry_days: int
    valuation_kind: Literal["expiry_payoff", "mark_to_market"]
    target_net_pnl_pct: float | None
    target_return_on_premium: float | None
    ratio_status: Literal["available", "not_applicable", "unavailable"]
    ratio_reason: str | None
    target_pnl_reason: str | None
    basis: str = "fraction of base-currency notional"
    valuation_convention: str = (
        "Scenario value converted at target spot minus entry premium; "
        "entry premium is not accrued, matching the scenario engine"
    )


def compute_trade_economics(
    variant, structure_id, ms, *, target, is_call, loss_budget=None,
    stop_price=None, evaluation_days=None, surface=None,
) -> TradeEconomics:
    """Describe a priced variant without mutating pricing, sizing or ranking."""
    expiry_days = round(ms.T * 365)
    days = expiry_days if evaluation_days is None else evaluation_days
    if not 0 <= days <= expiry_days:
        raise ValueError("Evaluation horizon must be between entry and expiry")
    premium = variant.net_premium_pct
    direction = "zero_cost" if variant.is_zero_cost or abs(premium) <= 1e-9 else (
        "paid" if premium > 0 else "received"
    )
    loss_method = "absolute net premium (legacy sizing proxy)"
    if structure_id == "seagull":
        loss_method = (
            "package stress value at sizing reference (not a contractual loss bound)"
            if stop_price is not None else "unavailable: no sizing reference supplied"
        )
    loss_status = "unknown"
    max_loss = None
    loss_reason = "Contractual maximum loss is not calculated for this package; do not infer it from the sizing proxy"
    if structure_id in {"vanilla", "european_digital", "european_rko"} and premium >= 0:
        loss_status = "bounded"
        max_loss = premium
        loss_reason = "Non-negative terminal payoff of the long option; premium at risk on the stated base-currency basis"

    net_pnl = None
    pnl_reason = None
    supported = {
        "vanilla", "1x1_spread", "1x1.5_spread", "1x2_spread",
        "1x2x1_spread", "seagull", "european_digital", "european_rko",
    }
    if target is None:
        pnl_reason = "No target specified"
    elif not math.isfinite(target) or target <= 0:
        pnl_reason = "Target must be a positive finite spot"
    elif structure_id not in supported:
        pnl_reason = "Target-only valuation unavailable for this product; path state may be required"
    else:
        from analytics.scenario_pricer import _value_variant

        try:
            remaining = 0.0 if days == expiry_days else ms.T - days / 365
            value = _value_variant(
                structure_id, variant, target, ms.vol, remaining,
                ms.r_d, ms.r_f, ms.spot, is_call, surface,
            )
            net_pnl = value / target - premium
            if not math.isfinite(net_pnl):
                net_pnl = None
                pnl_reason = "Target valuation returned a non-finite result"
        except (ValueError, ArithmeticError, IndexError, TypeError):
            pnl_reason = "Engine target valuation unavailable for these inputs"

    ratio = None
    if direction != "paid":
        ratio_status = "not_applicable"
        ratio_reason = "Not applicable — no premium outlay"
    elif net_pnl is None:
        ratio_status = "unavailable"
        ratio_reason = pnl_reason
    else:
        ratio_status = "available"
        ratio_reason = None
        ratio = net_pnl / premium

    return TradeEconomics(
        premium, direction,
        None if structure_id == "seagull" and stop_price is None else variant.max_loss_pct,
        loss_method, stop_price, loss_budget,
        loss_status, max_loss, loss_reason, target, days, expiry_days,
        "expiry_payoff" if days == expiry_days else "mark_to_market",
        net_pnl, ratio, ratio_status, ratio_reason, pnl_reason,
    )
