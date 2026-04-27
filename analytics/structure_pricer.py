"""
Structure variant pricer.

Prices the concrete instances defined in structure_variants.json.
All premiums and payoffs expressed as a fraction of spot (e.g., 0.03 = 3% of spot).
Digital payout is normalised to spot, so a "15% digital" costs 15% of spot to receive
spot if ITM — same basis as vanilla premium percentages.

Uses ATM vol (ms.vol) for all strikes (flat smile approximation).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from analytics.market_state import MarketState
from analytics.strike_resolver import otm_call_strike, otm_put_strike
from pricing.black_scholes import black76_call, black76_put
from pricing.digital import digital_call, digital_put
from pricing.digital_rko import digital_rko_call, digital_rko_put

_VARIANTS_PATH = Path(__file__).parent.parent / "knowledge" / "defaults" / "structure_variants.json"


def _load_variants() -> dict:
    with open(_VARIANTS_PATH) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@dataclass
class PricedVariant:
    variant_label: str
    strikes: list[float]          # main strikes in leg order (long first)
    barrier: float | None         # for RKO structures
    net_premium_pct: float        # fraction of spot; 0.0 for zero-cost
    breakeven: float | None       # spot at expiry where P&L = 0 (None for digital / zero-cost)
    payoff_at_target_pct: float | None  # gross payoff at target as fraction of spot
    rr_at_target: float | None    # gross payoff / premium; None if zero-cost or no target
    max_loss_pct: float           # fraction of spot
    wing_ratio: float | None      # seagull only: units of wing sold per unit of spread
    is_zero_cost: bool


def price_variants(
    ms: MarketState,
    structure_id: str,
    target: float | None = None,
    is_call: bool = True,
) -> list[PricedVariant]:
    """Price all defined variants for a structure. Returns [] if no variants defined."""
    cfg = _load_variants()
    if structure_id not in cfg:
        return []

    variants = cfg[structure_id]
    F, vol, T, r_d, r_f, spot = ms.fwd, ms.vol, ms.T, ms.r_d, ms.r_f, ms.spot
    DF = math.exp(-r_d * T)
    vol_sqrtT = vol * math.sqrt(T)

    if structure_id == "vanilla":
        return _vanilla(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target)
    if structure_id == "1x1_spread":
        return _spread(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target)
    if structure_id == "seagull":
        return _seagull(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target)
    if structure_id == "european_digital":
        return _digital(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target)
    if structure_id == "european_digital_rko":
        return _digital_rko(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target)
    return []


# ---------------------------------------------------------------------------
# Vanilla
# ---------------------------------------------------------------------------

def _vanilla(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target
) -> list[PricedVariant]:
    result = []
    for v in variants:
        delta = v["delta"]
        if is_call:
            K = otm_call_strike(F, vol, T, delta)
            prem = black76_call(F, K, T, vol, DF)
            be = K + prem
        else:
            K = otm_put_strike(F, vol, T, delta)
            prem = black76_put(F, K, T, vol, DF)
            be = K - prem

        prem_pct = prem / spot
        payoff_pct, rr = None, None
        if target is not None:
            raw = max(target - K, 0.0) if is_call else max(K - target, 0.0)
            payoff_pct = raw / spot
            rr = (payoff_pct / prem_pct) if prem_pct > 1e-8 else None

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K],
            barrier=None,
            net_premium_pct=prem_pct,
            breakeven=be,
            payoff_at_target_pct=payoff_pct,
            rr_at_target=rr,
            max_loss_pct=prem_pct,
            wing_ratio=None,
            is_zero_cost=False,
        ))
    return result


# ---------------------------------------------------------------------------
# 1x1 Spread
# ---------------------------------------------------------------------------

def _spread(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target
) -> list[PricedVariant]:
    result = []
    for v in variants:
        ld, sd = v["long_delta"], v["short_delta"]
        if is_call:
            K_long  = otm_call_strike(F, vol, T, ld)
            K_short = otm_call_strike(F, vol, T, sd)
            prem_long  = black76_call(F, K_long,  T, vol, DF)
            prem_short = black76_call(F, K_short, T, vol, DF)
        else:
            K_long  = otm_put_strike(F, vol, T, ld)
            K_short = otm_put_strike(F, vol, T, sd)
            prem_long  = black76_put(F, K_long,  T, vol, DF)
            prem_short = black76_put(F, K_short, T, vol, DF)

        net_prem = prem_long - prem_short
        prem_pct = net_prem / spot

        if is_call:
            max_payoff = K_short - K_long   # capped upside
            be = K_long + net_prem
        else:
            max_payoff = K_long - K_short
            be = K_long - net_prem

        payoff_pct, rr = None, None
        if target is not None:
            if is_call:
                raw = min(max(target - K_long, 0.0), max_payoff)
            else:
                raw = min(max(K_long - target, 0.0), max_payoff)
            payoff_pct = raw / spot
            rr = (payoff_pct / prem_pct) if prem_pct > 1e-8 else None

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K_long, K_short],
            barrier=None,
            net_premium_pct=prem_pct,
            breakeven=be,
            payoff_at_target_pct=payoff_pct,
            rr_at_target=rr,
            max_loss_pct=prem_pct,
            wing_ratio=None,
            is_zero_cost=False,
        ))
    return result


# ---------------------------------------------------------------------------
# Seagull  (spread + zero-cost OTM wing on opposite side)
# ---------------------------------------------------------------------------

def _seagull(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target
) -> list[PricedVariant]:
    result = []
    for v in variants:
        ld, sd, wd = v["spread_long"], v["spread_short"], v["wing_delta"]

        # Spread legs (in direction of view)
        if is_call:
            K1 = otm_call_strike(F, vol, T, ld)   # long call (lower K)
            K2 = otm_call_strike(F, vol, T, sd)   # short call (higher K)
            prem1 = black76_call(F, K1, T, vol, DF)
            prem2 = black76_call(F, K2, T, vol, DF)
            # Wing: OTM put (opposite direction)
            K3 = otm_put_strike(F, vol, T, wd)
            prem3 = black76_put(F, K3, T, vol, DF)
        else:
            K1 = otm_put_strike(F, vol, T, ld)    # long put (higher K)
            K2 = otm_put_strike(F, vol, T, sd)    # short put (lower K)
            prem1 = black76_put(F, K1, T, vol, DF)
            prem2 = black76_put(F, K2, T, vol, DF)
            # Wing: OTM call (opposite direction)
            K3 = otm_call_strike(F, vol, T, wd)
            prem3 = black76_call(F, K3, T, vol, DF)

        spread_cost = prem1 - prem2
        wing_ratio = (spread_cost / prem3) if prem3 > 1e-8 else 0.0

        # Max loss: wing sold at wing_ratio, loss if deep OTM
        if is_call:
            # Short put K3 at wing_ratio: loss if spot → 0, but practically capped at K3 × wing_ratio
            max_loss = K3 * wing_ratio / spot
        else:
            # Short call K3 at wing_ratio: theoretically unbounded, show K3 × wing_ratio as reference
            max_loss = K3 * wing_ratio / spot

        payoff_pct = None
        if target is not None:
            if is_call:
                # Spread payoff
                sp_payoff = min(max(target - K1, 0.0), K2 - K1)
                # Wing payoff: short put expires worthless if target > K3 (bull view, K3 < fwd)
                wing_payoff = -max(K3 - target, 0.0) * wing_ratio if target < K3 else 0.0
            else:
                sp_payoff = min(max(K1 - target, 0.0), K1 - K2)
                wing_payoff = -max(target - K3, 0.0) * wing_ratio if target > K3 else 0.0
            payoff_pct = (sp_payoff + wing_payoff) / spot

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K1, K2, K3],
            barrier=None,
            net_premium_pct=0.0,
            breakeven=None,
            payoff_at_target_pct=payoff_pct,
            rr_at_target=None,
            max_loss_pct=max_loss,
            wing_ratio=round(wing_ratio, 2),
            is_zero_cost=True,
        ))
    return result


# ---------------------------------------------------------------------------
# European digital  (bisect for target premium; payout normalised to spot)
# ---------------------------------------------------------------------------

def _digital(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target
) -> list[PricedVariant]:
    result = []
    fn_price = digital_call if is_call else digital_put

    for v in variants:
        tgt_pct = v["target_prem_pct"]
        target_prem = tgt_pct * spot  # absolute premium in domestic per unit

        # Bisect: find K such that digital ≈ target_prem
        # K_lo near spot (high premium) → K_hi far OTM (near-zero premium)
        if is_call:
            K_lo = spot * 1.001
            K_hi = F * math.exp(3.0 * vol_sqrtT)
        else:
            K_lo = F * math.exp(-3.0 * vol_sqrtT)
            K_hi = spot * 0.999

        K = _bisect_strike(fn_price, spot, K_lo, K_hi, T, vol, r_d, r_f, target_prem, is_call)

        prem = fn_price(spot, K, T, vol, r_d, r_f, payout=spot)
        prem_pct = prem / spot

        payoff_pct, rr = None, None
        if target is not None:
            itm = (target > K) if is_call else (target < K)
            if itm:
                payoff_pct = 1.0   # payout = spot → payoff = 100% of spot
                rr = payoff_pct / prem_pct if prem_pct > 1e-8 else None
            else:
                payoff_pct = 0.0
                rr = 0.0

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K],
            barrier=None,
            net_premium_pct=prem_pct,
            breakeven=K,
            payoff_at_target_pct=payoff_pct,
            rr_at_target=rr,
            max_loss_pct=prem_pct,
            wing_ratio=None,
            is_zero_cost=False,
        ))
    return result


# ---------------------------------------------------------------------------
# Digital + RKO  (barrier at barrier_sigmas × σ√T from forward)
# ---------------------------------------------------------------------------

def _digital_rko(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target
) -> list[PricedVariant]:
    result = []
    fn_price = digital_rko_call if is_call else digital_rko_put

    for v in variants:
        tgt_pct = v["target_prem_pct"]
        b_sigmas = v.get("barrier_sigmas", 2.0)
        target_prem = tgt_pct * spot

        if is_call:
            barrier = F * math.exp(b_sigmas * vol_sqrtT)
            K_lo = spot * 1.001
            K_hi = barrier * 0.9999
        else:
            barrier = F * math.exp(-b_sigmas * vol_sqrtT)
            K_lo = barrier * 1.0001
            K_hi = spot * 0.999

        K = _bisect_strike_rko(fn_price, spot, K_lo, K_hi, barrier, T, vol, r_d, r_f, target_prem, is_call)

        prem = fn_price(spot, K, barrier, T, vol, r_d, r_f, payout=spot)
        prem_pct = prem / spot

        payoff_pct, rr = None, None
        if target is not None:
            itm = (target > K) if is_call else (target < K)
            not_ko = (target < barrier) if is_call else (target > barrier)
            if itm and not_ko:
                payoff_pct = 1.0
                rr = payoff_pct / prem_pct if prem_pct > 1e-8 else None
            else:
                payoff_pct = 0.0
                rr = 0.0

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K],
            barrier=barrier,
            net_premium_pct=prem_pct,
            breakeven=K,
            payoff_at_target_pct=payoff_pct,
            rr_at_target=rr,
            max_loss_pct=prem_pct,
            wing_ratio=None,
            is_zero_cost=False,
        ))
    return result


# ---------------------------------------------------------------------------
# Bisection helpers
# ---------------------------------------------------------------------------

def _bisect_strike(fn, spot, K_lo, K_hi, T, vol, r_d, r_f, target_prem, is_call, tol=1e-6) -> float:
    for _ in range(60):
        K_mid = 0.5 * (K_lo + K_hi)
        val = fn(spot, K_mid, T, vol, r_d, r_f, payout=spot)
        if abs(val - target_prem) < tol * spot:
            return K_mid
        # For calls: price decreases as K increases; for puts: price increases as K decreases
        if is_call:
            if val > target_prem:
                K_lo = K_mid   # need higher K
            else:
                K_hi = K_mid
        else:
            if val > target_prem:
                K_hi = K_mid   # for puts, price increases as K decreases
            else:
                K_lo = K_mid
    return 0.5 * (K_lo + K_hi)


def _bisect_strike_rko(fn, spot, K_lo, K_hi, barrier, T, vol, r_d, r_f, target_prem, is_call, tol=1e-6) -> float:
    for _ in range(60):
        K_mid = 0.5 * (K_lo + K_hi)
        val = fn(spot, K_mid, barrier, T, vol, r_d, r_f, payout=spot)
        if abs(val - target_prem) < tol * spot:
            return K_mid
        if is_call:
            if val > target_prem:
                K_lo = K_mid
            else:
                K_hi = K_mid
        else:
            if val > target_prem:
                K_hi = K_mid
            else:
                K_lo = K_mid
    return 0.5 * (K_lo + K_hi)
