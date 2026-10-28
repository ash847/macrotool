"""Explicit legacy entry/display and sizing policies, separate from leg valuation.

These are compatibility rules, not generic statements about maximum loss.
Changing them requires a financial-policy decision rather than a refactor.
"""

import math

from analytics.product_model import ResolvedContract, Right
from analytics.resolved_pricing import PricingContext, terminal_payoff, value_resolved


def entry_zero_cost(contract: ResolvedContract, premium_pct: float) -> bool:
    return contract.family not in ("vanilla", "1x1_spread") and abs(premium_pct) < 0.0001


def entry_target_payoff(contract: ResolvedContract, target: float) -> float:
    if contract.family == "1x1_spread":
        # Preserve the historical capped display even for reversed input strikes.
        first, second = contract.legs
        sign = 1 if first.right == Right.CALL else -1
        return min(max(sign * (target - first.strike), 0.0),
                   sign * (second.strike - first.strike)) / target
    return float(terminal_payoff(contract, target))


def sizing_loss_proxy(contract: ResolvedContract, premium_pct: float, ms,
                      *, stop_price=None, surface=None) -> float:
    family = contract.family
    if family in ("vanilla", "1x1_spread"):
        return premium_pct
    if family == "1x2x1_spread":
        return max(premium_pct, 0.0)
    if family == "seagull":
        if stop_price is None:
            return 0.0
        # The stop is a forward; normalization remains the inception spot.
        spot = stop_price * math.exp(-(ms.r_d - ms.r_f) * ms.T)
        context = PricingContext(spot, ms.T, ms.vol, ms.r_d, ms.r_f, surface)
        return abs(value_resolved(contract, context)) / ms.spot
    if family in ("1x1.5_spread", "1x2_spread", "ratio_spread"):
        # Existing premium-based sizing proxy does not capitalize open-tail loss.
        return abs(premium_pct)
    raise ValueError(f"No approved sizing policy for {family}")
