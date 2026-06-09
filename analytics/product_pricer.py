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

    if family in ("1x1.5_spread", "1x2_spread"):
        ratio = 1.5 if family == "1x1.5_spread" else 2.0
        if "long_delta" in variant and "short_delta" in variant:
            legs = (
                Leg(Instrument.VANILLA, right, +1.0, Anchor(AnchorKind.DELTA, variant["long_delta"])),
                Leg(Instrument.VANILLA, right, -ratio, Anchor(AnchorKind.DELTA, variant["short_delta"])),
            )
        elif variant.get("long_type") == "atmf":
            legs = (
                Leg(Instrument.VANILLA, right, +1.0, Anchor(AnchorKind.ATMF)),
                Leg(Instrument.VANILLA, right, -ratio, Anchor(AnchorKind.TARGET)),
            )
        elif variant.get("long_type") == "half_sigma":
            legs = (
                Leg(Instrument.VANILLA, right, +1.0, Anchor(AnchorKind.HALF_SIGMA)),
                Leg(Instrument.VANILLA, right, -ratio, Anchor(AnchorKind.TARGET)),
            )
        else:
            return None
        return Structure(family, legs, label)

    if family == "seagull":
        opp = Right.PUT if is_call else Right.CALL
        legs = (
            Leg(Instrument.VANILLA, right, +1.0, Anchor(AnchorKind.DELTA, variant["spread_long"])),
            Leg(Instrument.VANILLA, right, -1.0, Anchor(AnchorKind.DELTA, variant["spread_short"])),
            # Wing: opposite side, notional solved to fund zero cost (set at price time).
            Leg(Instrument.VANILLA, opp, -1.0, Anchor(AnchorKind.DELTA, variant["wing_delta"])),
        )
        return Structure(family, legs, label)

    if family == "european_digital":
        # Binary, base-ccy cash-or-nothing; strike solved for the target premium.
        legs = (Leg(Instrument.DIGITAL, right, +1.0, Anchor(AnchorKind.PREMIUM, variant["target_prem_pct"])),)
        return Structure(family, legs, label)

    if family == "european_rko":
        # Long vanilla + an expiry-only knock-out barrier (structure-level).
        legs = (Leg(Instrument.VANILLA, right, +1.0, Anchor(AnchorKind.DELTA, variant["long_delta"])),)
        return Structure(family, legs, label, barrier_anchor=Anchor(AnchorKind.DELTA, variant["barrier"]))

    return None


_RATIO_FAMILIES = ("1x1.5_spread", "1x2_spread")
# Binary/barrier families: the leg model is least natural (binary payoff, barrier), so these
# are priced by their own pricer via the legacy seam — wrapped into a PricedStructure for a
# uniform consumer interface. Parity is exact (same code path).
_WRAPPED_FAMILIES = ("european_digital", "european_rko")


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


def _variant_dict_from(structure: Structure) -> dict:
    """Reconstruct the legacy variant dict for a wrapped (binary/barrier) family."""
    legs = structure.legs
    if structure.family == "european_digital":
        return {"label": structure.label, "target_prem_pct": legs[0].anchor.value}
    if structure.family == "european_rko":
        return {
            "label": structure.label,
            "long_delta": legs[0].anchor.value,
            "barrier": structure.barrier_anchor.value,
        }
    raise ValueError(f"no wrapped variant dict for {structure.family}")


def _price_wrapped(structure, ms, target, smile, stop_price) -> PricedStructure | None:
    """Price a binary/barrier family via the legacy seam, wrapped into a PricedStructure.
    Parity is exact (same code path). Returns None if the legacy pricer drops the variant
    (e.g. a smile-arbitrage digital)."""
    from analytics.structure_pricer import price_variants

    is_call = structure.legs[0].right == Right.CALL
    warnings: list[str] = []
    pvs = price_variants(
        ms, structure.family, target=target, is_call=is_call, stop_price=stop_price,
        smile=smile, warnings=warnings, variants_override=[_variant_dict_from(structure)],
    )
    if not pvs:
        return None
    pv = pvs[0]
    priced_legs = [PricedLeg(structure.legs[0], pv.strikes[0], 0.0, 0.0)]
    return PricedStructure(
        structure=structure,
        priced_legs=priced_legs,
        net_premium_pct=pv.net_premium_pct,
        payoff_at_target_pct=pv.payoff_at_target_pct,
        rr_at_target=pv.rr_at_target,
        max_loss_pct=pv.max_loss_pct,
        breakeven=pv.breakeven,
        is_zero_cost=pv.is_zero_cost,
        barrier=pv.barrier,
        strikes_override=list(pv.strikes),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# price
# ---------------------------------------------------------------------------

def price(
    structure: Structure,
    ms: MarketState,
    target: float | None = None,
    smile=None,
    stop_price: float | None = None,
) -> PricedStructure:
    fam = structure.family
    if fam in _WRAPPED_FAMILIES:
        return _price_wrapped(structure, ms, target, smile, stop_price)

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

    # Solved leg notionals: the seagull wing is sized to fund the structure to zero cost.
    wing_ratio = None
    if fam == "seagull" and len(priced_legs) == 3:
        spread_cost = priced_legs[0].unit_premium - priced_legs[1].unit_premium
        prem_wing = priced_legs[2].unit_premium
        wing_ratio = (spread_cost / prem_wing) if prem_wing > 1e-8 else 0.0
        priced_legs[2].effective_notional = -wing_ratio

    # Linear aggregation: net premium = Σ notional × unit (using solved notionals).
    net_prem = sum(pl.notional * pl.unit_premium for pl in priced_legs)
    prem_pct = net_prem / spot
    is_zero_cost = abs(net_prem) < _ZERO_COST_EPS * spot

    long_leg = next((pl for pl in priced_legs if pl.notional > 0), None)
    long_is_call = long_leg.leg.right == Right.CALL if long_leg else True

    payoff_pct = rr = None
    if target is not None:
        if fam in _RATIO_FAMILIES:
            # Legacy quirk (preserved for behavior-parity): ratio payoff-at-target counts
            # the LONG leg intrinsic only (correct when the short sits at the target;
            # overstates for a deep delta-pair target). Faithful leg-sum is a deliberate
            # later correctness fix — see PRODUCT_MODEL_PLAN.
            raw = _intrinsic(long_leg.strike, target, long_is_call)
        else:
            raw = sum(
                pl.notional * _intrinsic(pl.strike, target, pl.leg.right == Right.CALL)
                for pl in priced_legs
            )
        payoff_pct = raw / target
        rr = (
            (payoff_pct / prem_pct)
            if (payoff_pct is not None and not is_zero_cost and prem_pct > 1e-8)
            else None
        )

    # Breakeven: long strike ± net premium (debit only; not for a zero-cost package).
    breakeven = None
    if long_leg is not None and not is_zero_cost and net_prem > 0:
        breakeven = long_leg.strike + (net_prem if long_is_call else -net_prem)

    # Max loss: vanilla / 1x1 = net premium. Ratio spreads = the package's stressed-forward
    # value (open tail beyond target), floored at the premium — reusing the legacy helper.
    strikes = [pl.strike for pl in priced_legs]
    if fam in _RATIO_FAMILIES:
        from analytics.structure_pricer import _today_package_value_pct
        stress_fwd = target if target is not None else strikes[1]
        ml = _today_package_value_pct(
            structure_id=fam, strikes=strikes, fwd_today=stress_fwd, T=T, vol=vol,
            r_d=r_d, r_f=r_f, spot=spot, is_call=long_is_call, surface=vm.surface,
        )
        max_loss_pct = max(ml, abs(prem_pct))
    elif fam == "seagull":
        if stop_price is not None:
            from analytics.structure_pricer import _today_package_value_pct
            max_loss_pct = _today_package_value_pct(
                structure_id="seagull", strikes=strikes, fwd_today=stop_price, T=T, vol=vol,
                r_d=r_d, r_f=r_f, spot=spot, is_call=long_is_call,
                wing_ratio=wing_ratio, surface=vm.surface,
            )
        else:
            max_loss_pct = 0.0
    else:
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
        wing_ratio=round(wing_ratio, 2) if wing_ratio is not None else None,
    )
