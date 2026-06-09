"""Product pricer — `build_structure` + `price(structure, ms)`.

Phase B of the product-model refactor (see PRODUCT_MODEL_PLAN.md). Per-leg
model-appropriate pricing + uniform linear aggregation:

  value/greeks  = Σ signed_notionalᵢ × unitᵢ          (linear)
  payoff(target)= Σ signed_notionalᵢ × intrinsicᵢ(target)   (linear in legs at a point)

No structure-type dispatch in the aggregator. The only family-specific knowledge is
(a) how `build_structure` lays out the legs, and (b) the max-loss "stress point" and the
breakeven convention (both small, declarative).

Parity: reuses the exact legacy primitives (`black76_call/put`, `otm_*_strike`, `_VolModel`)
so the leg-sum reproduces `analytics.structure_pricer.price_variants` byte-for-byte on the
families implemented here. Guarded by `tests/test_product_model_parity.py`.

Implemented so far: vanilla, 1x1_spread. (1x1.5/1x2/seagull/digital/european_rko follow.)
"""

from __future__ import annotations

import math

from analytics.market_state import MarketState
from analytics.product_model import (
    Anchor,
    AnchorKind,
    Instrument,
    Leg,
    PricedLeg,
    PricedStructure,
    Right,
    Structure,
)
from analytics.strike_resolver import otm_call_strike, otm_put_strike
from analytics.structure_pricer import _VolModel
from pricing.black_scholes import black76_call, black76_put

_ZERO_COST_EPS = 0.0001   # × spot, matches legacy is_zero_cost test


# ---------------------------------------------------------------------------
# build_structure — JSON variant dict → Structure (market-independent)
# ---------------------------------------------------------------------------

def build_structure(family: str, variant: dict, is_call: bool) -> Structure | None:
    """Convert a curated variant dict into a Structure. Returns None for families
    not yet ported (caller falls back to the legacy pricer)."""
    right = Right.CALL if is_call else Right.PUT
    label = variant.get("label", "")

    if family == "vanilla":
        legs = (Leg(Instrument.VANILLA, right, +1.0, Anchor(AnchorKind.DELTA, variant["delta"])),)
        return Structure(family, legs, label)

    if family == "1x1_spread":
        legs = (
            Leg(Instrument.VANILLA, right, +1.0, Anchor(AnchorKind.DELTA, variant["long_delta"])),
            Leg(Instrument.VANILLA, right, -1.0, Anchor(AnchorKind.DELTA, variant["short_delta"])),
        )
        return Structure(family, legs, label)

    return None


# ---------------------------------------------------------------------------
# Leg resolution
# ---------------------------------------------------------------------------

def _resolve_leg(leg: Leg, F, vol, T, vol_sqrtT, vm, target) -> tuple[float, float]:
    """(strike, vol) for a leg, reproducing the legacy resolution exactly."""
    a = leg.anchor
    is_call = leg.right == Right.CALL
    if a.kind == AnchorKind.DELTA:
        v = vm.at_delta(a.value, is_call)
        K = otm_call_strike(F, v, T, a.value) if is_call else otm_put_strike(F, v, T, a.value)
        return K, v
    if a.kind == AnchorKind.ATMF:
        return F, vm.at_strike(F)
    if a.kind == AnchorKind.HALF_SIGMA:
        K = F * math.exp(0.5 * vol_sqrtT) if is_call else F * math.exp(-0.5 * vol_sqrtT)
        return K, vm.at_strike(K)
    if a.kind == AnchorKind.TARGET:
        return target, vm.at_strike(target)
    if a.kind == AnchorKind.STRIKE:
        return a.value, vm.at_strike(a.value)
    raise ValueError(f"unsupported anchor for vanilla leg: {a.kind}")


def _intrinsic(strike: float, target: float, is_call: bool) -> float:
    return max(target - strike, 0.0) if is_call else max(strike - target, 0.0)


# ---------------------------------------------------------------------------
# price
# ---------------------------------------------------------------------------

def price(
    structure: Structure,
    ms: MarketState,
    target: float | None = None,
    smile=None,
) -> PricedStructure:
    F, vol, T, r_d, r_f, spot = ms.fwd, ms.vol, ms.T, ms.r_d, ms.r_f, ms.spot
    DF = math.exp(-r_d * T)
    vol_sqrtT = vol * math.sqrt(T)
    vm = _VolModel(vol, smile=smile, F=F, horizon_days=round(T * 365))

    priced_legs: list[PricedLeg] = []
    for leg in structure.legs:
        K, v = _resolve_leg(leg, F, vol, T, vol_sqrtT, vm, target)
        is_call = leg.right == Right.CALL
        unit = black76_call(F, K, T, v, DF) if is_call else black76_put(F, K, T, v, DF)
        priced_legs.append(PricedLeg(leg, K, v, unit))

    # Linear aggregation: net premium = Σ signed_notional × unit.
    net_prem = sum(pl.leg.signed_notional * pl.unit_premium for pl in priced_legs)
    prem_pct = net_prem / spot
    is_zero_cost = abs(net_prem) < _ZERO_COST_EPS * spot

    payoff_pct = rr = None
    if target is not None:
        raw = sum(
            pl.leg.signed_notional * _intrinsic(pl.strike, target, pl.leg.right == Right.CALL)
            for pl in priced_legs
        )
        payoff_pct = raw / target
        rr = (payoff_pct / prem_pct) if (not is_zero_cost and prem_pct > 1e-8) else None

    # Breakeven: long strike ± net premium (debit only). Family-declarative.
    long_leg = next((pl for pl in priced_legs if pl.leg.signed_notional > 0), None)
    breakeven = None
    if long_leg is not None and net_prem > 0:
        is_call = long_leg.leg.right == Right.CALL
        breakeven = long_leg.strike + (net_prem if is_call else -net_prem)

    # Max loss: for vanilla & 1x1 (fully capped / single long), max loss = net premium.
    max_loss_pct = abs(prem_pct)

    return PricedStructure(
        structure=structure,
        priced_legs=priced_legs,
        net_premium_pct=prem_pct,
        payoff_at_target_pct=payoff_pct,
        rr_at_target=rr,
        max_loss_pct=max_loss_pct,
        breakeven=breakeven,
        is_zero_cost=is_zero_cost,
    )
