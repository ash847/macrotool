"""Product pricer — `build_structure` + `price(structure, ms)`.

Construction resolves anchors and funding constraints once. Fixed contracts
then drive aggregation, payoffs and revaluation:

  value/greeks  = Σ signed_notionalᵢ × unitᵢ          (linear)
  payoff(target)= Σ signed_notionalᵢ × intrinsicᵢ(target)   (linear in legs at a point)

Family-specific entry/display and sizing conventions live in product_policies.
These are compatibility rules, not generic risk guarantees.

Parity: reuses the exact legacy primitives (`black76_call/put`, `otm_*_strike`, `_VolModel`)
so the leg-sum reproduces `analytics.structure_pricer.price_variants` byte-for-byte on the
families implemented here. Guarded by `tests/test_product_model_parity.py`.

Entry construction supports vanilla-leg families plus specialised European wrappers.
All six supported vanilla-leg families retain immutable resolved contracts.
Specialised binary/barrier wrappers retain their existing consumer paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

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
    ResolvedContract,
    ResolvedLeg,
)
from analytics.strike_resolver import otm_call_strike, otm_put_strike
from analytics.structure_pricer import _VolModel
from pricing.black_scholes import black76_call, black76_put

RESOLVED_FAMILIES = frozenset(("vanilla", "1x1_spread", "1x1.5_spread", "1x2_spread",
                              "1x2x1_spread", "seagull", "ratio_spread"))


class InvalidButterflyWing(ValueError):
    """A derived butterfly wing cannot be represented by a positive strike."""


def freeze_contract(structure: Structure, strikes, tenor_years: float, *, notionals=None) -> ResolvedContract:
    """Retain already resolved strikes; never price or resolve an anchor here."""
    if structure.family not in RESOLVED_FAMILIES or structure.barrier_anchor is not None:
        raise ValueError("Family not yet migrated to the resolved vanilla contract")
    if len(strikes) != len(structure.legs):
        raise ValueError("Resolved strike count does not match the structure")
    if any(leg.instrument != Instrument.VANILLA for leg in structure.legs):
        raise ValueError("Resolved vanilla contract cannot contain binary legs")
    if notionals is None:
        if structure.family == "seagull":
            raise ValueError("Seagull requires solved leg quantities")
        notionals = [leg.signed_notional for leg in structure.legs]
    if len(notionals) != len(structure.legs):
        raise ValueError("Resolved quantity count does not match the structure")
    return ResolvedContract(
        structure.family, structure.label,
        tuple(ResolvedLeg(leg.right, strike, quantity)
              for leg, strike, quantity in zip(structure.legs, strikes, notionals)), tenor_years,
    )


# ---------------------------------------------------------------------------
# build_structure — JSON variant dict → Structure (market-independent)
# ---------------------------------------------------------------------------

def _anchor(variant: dict, delta_key: str, strike_key: str) -> Anchor:
    """A leg's anchor: the absolute strike under ``strike_key`` if present, else the
    delta under ``delta_key``."""
    if strike_key in variant:
        return Anchor(AnchorKind.STRIKE, variant[strike_key])
    return Anchor(AnchorKind.DELTA, variant[delta_key])


def _has_leg(variant: dict, delta_key: str, strike_key: str) -> bool:
    return delta_key in variant or strike_key in variant


def build_structure(family: str, variant: dict, is_call: bool) -> Structure | None:
    structure = _build_structure(family, variant, is_call)
    if structure is None:
        return None
    atmf = set(variant.get("atmf_legs", ()))
    legs = tuple(replace(leg, anchor=Anchor(AnchorKind.ATMF)) if i in atmf else leg
                 for i, leg in enumerate(structure.legs))
    barrier = (Anchor(AnchorKind.ATMF) if family == "european_rko" and 1 in atmf
               else structure.barrier_anchor)
    return replace(structure, legs=legs, barrier_anchor=barrier)


def _build_structure(family: str, variant: dict, is_call: bool) -> Structure | None:
    """Convert a curated variant dict into a Structure. Returns None for families
    not yet ported (caller falls back to the legacy pricer)."""
    right = Right.CALL if is_call else Right.PUT
    label = variant.get("label", "")

    if family == "vanilla":
        legs = (Leg(Instrument.VANILLA, right, +1.0, _anchor(variant, "delta", "strike")),)
        return Structure(family, legs, label)

    if family == "1x1_spread":
        legs = (
            Leg(Instrument.VANILLA, right, +1.0, _anchor(variant, "long_delta", "long_strike")),
            Leg(Instrument.VANILLA, right, -1.0, _anchor(variant, "short_delta", "short_strike")),
        )
        return Structure(family, legs, label)

    if family in ("1x1.5_spread", "1x2_spread", "ratio_spread"):
        ratio = variant["short_ratio"] if family == "ratio_spread" else (1.5 if family == "1x1.5_spread" else 2.0)
        if not math.isfinite(ratio) or ratio <= 1:
            raise ValueError("Ratio spread requires a finite short ratio greater than one")
        if _has_leg(variant, "long_delta", "long_strike") and _has_leg(variant, "short_delta", "short_strike"):
            legs = (
                Leg(Instrument.VANILLA, right, +1.0, _anchor(variant, "long_delta", "long_strike")),
                Leg(Instrument.VANILLA, right, -ratio, _anchor(variant, "short_delta", "short_strike")),
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

    if family == "1x2x1_spread":
        # Butterfly: the 1x2's two legs (+1 / −2) plus a long wing (+1) at
        # K3 = 2·K2 − K1. The wing's strike is derived from the resolved base
        # strikes at price time, so its anchor here is a placeholder (the short
        # leg's) that price() overrides.
        if _has_leg(variant, "long_delta", "long_strike") and _has_leg(variant, "short_delta", "short_strike"):
            long_anchor = _anchor(variant, "long_delta", "long_strike")
            short_anchor = _anchor(variant, "short_delta", "short_strike")
        elif variant.get("long_type") == "atmf":
            long_anchor, short_anchor = Anchor(AnchorKind.ATMF), Anchor(AnchorKind.TARGET)
        elif variant.get("long_type") == "half_sigma":
            long_anchor, short_anchor = Anchor(AnchorKind.HALF_SIGMA), Anchor(AnchorKind.TARGET)
        else:
            return None
        legs = (
            Leg(Instrument.VANILLA, right, +1.0, long_anchor),
            Leg(Instrument.VANILLA, right, -2.0, short_anchor),
            Leg(Instrument.VANILLA, right, +1.0, short_anchor),  # wing; strike set in price()
        )
        return Structure(family, legs, label)

    if family == "seagull":
        opp = Right.PUT if is_call else Right.CALL
        legs = (
            Leg(Instrument.VANILLA, right, +1.0, _anchor(variant, "spread_long", "spread_long_strike")),
            Leg(Instrument.VANILLA, right, -1.0, _anchor(variant, "spread_short", "spread_short_strike")),
            # Wing: opposite side, notional solved to fund zero cost (set at price time).
            Leg(Instrument.VANILLA, opp, -1.0, _anchor(variant, "wing_delta", "wing_strike")),
        )
        return Structure(family, legs, label)

    if family == "european_digital":
        # Binary, base-ccy cash-or-nothing; strike solved for the target premium.
        legs = (Leg(Instrument.DIGITAL, right, +1.0, Anchor(AnchorKind.PREMIUM, variant["target_prem_pct"])),)
        return Structure(family, legs, label)

    if family == "european_rko":
        # Long vanilla + an expiry-only knock-out barrier (structure-level).
        legs = (Leg(Instrument.VANILLA, right, +1.0, _anchor(variant, "long_delta", "long_strike")),)
        return Structure(family, legs, label, barrier_anchor=_anchor(variant, "barrier", "barrier_strike"))

    return None


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


def _variant_dict_from(structure: Structure) -> dict:
    variant = _wrapped_variant_dict(structure)
    atmf = [i for i, leg in enumerate(structure.legs) if leg.anchor.kind == AnchorKind.ATMF]
    if structure.barrier_anchor is not None and structure.barrier_anchor.kind == AnchorKind.ATMF:
        atmf.append(1)
    if atmf:
        variant["atmf_legs"] = atmf
    return variant


def _wrapped_variant_dict(structure: Structure) -> dict:
    """Reconstruct the legacy variant dict for a wrapped (binary/barrier) family."""
    legs = structure.legs
    if structure.family == "european_digital":
        return {"label": structure.label, "target_prem_pct": legs[0].anchor.value}
    if structure.family == "european_rko":
        def entry(anchor, delta_key, strike_key):
            return {strike_key if anchor.kind == AnchorKind.STRIKE else delta_key:
                    0.5 if anchor.kind == AnchorKind.ATMF else anchor.value}
        return {
            "label": structure.label,
            **entry(legs[0].anchor, "long_delta", "long_strike"),
            **entry(structure.barrier_anchor, "barrier", "barrier_strike"),
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
        breakevens=pv.breakevens,
        breakeven_has_zero_region=pv.breakeven_has_zero_region,
        is_zero_cost=pv.is_zero_cost,
        barrier=pv.barrier,
        strikes_override=list(pv.strikes),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# price
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstructedPackage:
    """Fixed contract plus inception quotes; no market anchors remain in contract."""
    contract: ResolvedContract
    priced_legs: tuple[PricedLeg, ...]
    wing_ratio: float | None


def construct_package(
    structure: Structure,
    ms: MarketState,
    target: float | None = None,
    smile=None,
) -> ConstructedPackage:
    """Resolve anchors and construction constraints once, before aggregation."""
    fam = structure.family
    if fam not in RESOLVED_FAMILIES:
        raise ValueError(f"Unsupported vanilla construction family: {fam}")

    F, vol, T, r_d = ms.fwd, ms.vol, ms.T, ms.r_d
    DF = math.exp(-r_d * T)
    vol_sqrtT = vol * math.sqrt(T)
    vm = _VolModel(vol, smile=smile, F=F, horizon_days=round(T * 365))

    priced_legs: list[PricedLeg] = []
    for leg in structure.legs:
        K, v = _resolve_leg(leg, F, vol, T, vol_sqrtT, vm, target)
        is_call = leg.right == Right.CALL
        unit = black76_call(F, K, T, v, DF) if is_call else black76_put(F, K, T, v, DF)
        priced_legs.append(PricedLeg(leg, K, v, unit))

    # Butterfly wing: derive K3 = 2·K2 − K1 from the resolved base strikes and
    # reprice the placeholder wing leg at its own (smile) vol.
    if fam == "1x2x1_spread" and len(priced_legs) == 3:
        K3 = 2.0 * priced_legs[1].strike - priced_legs[0].strike
        if K3 <= 0:
            raise InvalidButterflyWing("Butterfly wing strike must be positive")
        is_call3 = priced_legs[2].leg.right == Right.CALL
        v3 = vm.at_strike(K3)
        unit3 = black76_call(F, K3, T, v3, DF) if is_call3 else black76_put(F, K3, T, v3, DF)
        priced_legs[2] = PricedLeg(priced_legs[2].leg, K3, v3, unit3)

    # Solved leg notionals: the seagull wing is sized to fund the structure to zero cost.
    wing_ratio = None
    if fam == "seagull" and len(priced_legs) == 3:
        spread_cost = priced_legs[0].unit_premium - priced_legs[1].unit_premium
        prem_wing = priced_legs[2].unit_premium
        wing_ratio = (spread_cost / prem_wing) if prem_wing > 1e-8 else 0.0
        priced_legs[2].effective_notional = -wing_ratio

    contract = freeze_contract(structure, [leg.strike for leg in priced_legs], T,
                               notionals=[leg.notional for leg in priced_legs])
    return ConstructedPackage(contract, tuple(priced_legs), wing_ratio)


def price(
    structure: Structure,
    ms: MarketState,
    target: float | None = None,
    smile=None,
    stop_price: float | None = None,
) -> PricedStructure:
    """Construct once, aggregate fixed legs, then apply explicit entry policies."""
    if structure.family in _WRAPPED_FAMILIES:
        return _price_wrapped(structure, ms, target, smile, stop_price)
    package = construct_package(structure, ms, target, smile)
    return _price_constructed(structure, package, ms, target, smile, stop_price)


def _price_constructed(structure: Structure, package: ConstructedPackage, ms: MarketState,
                       target=None, smile=None, stop_price=None) -> PricedStructure:
    """Assemble inception output from fixed legs and retained entry quotes.

    This is not a market revaluation API; use value_resolved for market shocks.
    """
    from analytics.product_policies import entry_target_payoff, entry_zero_cost, sizing_loss_proxy

    contract = package.contract
    priced_legs = list(package.priced_legs)
    prem_pct = sum(leg.signed_notional * quote.unit_premium
                   for leg, quote in zip(contract.legs, priced_legs)) / ms.spot
    is_zero_cost = entry_zero_cost(contract, prem_pct)

    payoff_pct = rr = None
    if target is not None:
        payoff_pct = entry_target_payoff(contract, target)
        rr = (
            (payoff_pct / prem_pct)
            if (payoff_pct is not None and not is_zero_cost and prem_pct > 1e-8)
            else None
        )

    from analytics.expiry_breakeven import expiry_breakevens

    roots, zero_region = expiry_breakevens(
        contract.expiry_terms, prem_pct,
    )
    breakeven = roots[0] if roots is not None and len(roots) == 1 and not zero_region else None

    max_loss_pct = sizing_loss_proxy(contract, prem_pct, ms, stop_price=stop_price, surface=smile)

    return PricedStructure(
        structure=structure,
        priced_legs=priced_legs,
        net_premium_pct=prem_pct,
        payoff_at_target_pct=payoff_pct,
        rr_at_target=rr,
        max_loss_pct=max_loss_pct,
        breakeven=breakeven,
        breakevens=roots,
        breakeven_has_zero_region=zero_region,
        is_zero_cost=is_zero_cost,
        wing_ratio=package.wing_ratio,
        resolved_contract=contract,
    )
