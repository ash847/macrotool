"""
Market state computation for structure selection.

Derives quantitative metrics from raw market inputs (spot, fwd, vol, T, target).
All computations are deterministic; no domain judgment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pricing.black_scholes import call_value, put_value


@dataclass
class MarketState:
    """Fully-derived market state for a single trade horizon."""

    # --- raw inputs ---
    spot: float
    fwd: float
    vol: float    # ATM vol, annualised (e.g. 0.15 = 15%)
    T: float      # years to expiry
    r_d: float    # domestic continuously-compounded rate
    r_f: float    # foreign continuously-compounded rate

    # --- derived ---
    c: float             # ln(fwd/spot) / (σ√T) — normalised carry
    carry_regime: int    # 0 = noisy (<0.4), 1 = potential (0.4–0.8), 2 = high (>0.8)
    target_z: float | None      # ln(target/fwd) / (σ√T) — FORWARD-anchored; construction/eligibility/put_call. None if no target
    atmfsratio: float | None    # high-carry-ccy ATM-fwd/ATM-spot spread payout ratio (quote-ccy premium); None if spot==fwd
    put_call: str | None        # "Call" if target > fwd, "Put" if target < fwd; None if no target
    with_carry: bool            # True if view direction aligns with the carry (sign of c)

    # SPOT-anchored σ-distance of the target — the move from where spot is now.
    # Used by the scoring layer (affinity buckets/gates); `target_z` (forward) stays
    # for construction/eligibility/put_call. Identity: target_z = target_z_spot - c.
    # Default None so direct MarketState() constructions need not supply it.
    target_z_spot: float | None = None

    # Vol surface this state was priced against (None → flat ATM vol was used).
    # Carried so downstream vanilla pricing (scenario MtM, sizing) can re-use the
    # same surface rather than re-deriving it. Excluded from repr to keep logs clean.
    surface: object | None = field(default=None, repr=False, compare=False)


def compute_market_state(
    spot: float,
    fwd: float,
    vol: float,
    T: float,
    r_d: float,
    r_f: float,
    target: float | None = None,
    direction: str | None = None,
    carry_regime_cuts: list[float] | None = None,
    surface: object | None = None,
) -> MarketState:
    """
    Compute all derived market state metrics from raw inputs.

    atmfsratio is the payout ratio of the spread that captures the forward drift,
    built from options ON THE HIGHER-CARRY (higher-rate) CURRENCY, priced in the
    QUOTE currency on the original pair: a pair PUT when the quote ccy is
    high-carry (fwd > spot), a pair CALL when the base ccy is high-carry
    (fwd < spot). Long ATM-fwd, short ATM-spot, so it equals
    |fwd - spot| / (ATM-fwd premium - ATM-spot premium). None only if spot == fwd.

    Args:
        spot:   Spot rate.
        fwd:    Outright forward at expiry T.
        vol:    ATM implied vol, annualised.
        T:      Time to expiry in years.
        r_d:    Domestic continuously-compounded rate.
        r_f:    Foreign continuously-compounded rate.
        target: Optional target spot level.
        surface: Optional VolSurface. When supplied, the atmfsratio legs price at
                 the surface's interpolated vol per strike (vanilla options on the
                 high-carry ccy, so they MUST follow the smile). When None, a flat
                 ATM-vol surface is used, reproducing the legacy scalar-vol value.
    """
    from analytics.vol_surface import FlatSurface

    eff_surface = surface if surface is not None else FlatSurface(vol)
    horizon_days = max(round(T * 365), 1)
    vol_sqrt_T = vol * math.sqrt(T)

    c = math.log(fwd / spot) / vol_sqrt_T

    if carry_regime_cuts is None:
        from knowledge_engine.loader import load_affinity_scores
        carry_regime_cuts = load_affinity_scores()["thresholds"]["carry_regime"]
    cuts = carry_regime_cuts
    abs_c = abs(c)
    if abs_c < cuts[0]:
        carry_regime = 0
    elif abs_c < cuts[1]:
        carry_regime = 1
    else:
        carry_regime = 2

    target_z = math.log(target / fwd) / vol_sqrt_T if target is not None else None
    target_z_spot = math.log(target / spot) / vol_sqrt_T if target is not None else None
    put_call = ("Call" if target > fwd else "Put") if target is not None else None
    with_carry = (c > 0) == (direction == "base_lower") if direction else (c > 0)

    # atmfsratio: payout ratio of the spread that captures the forward drift,
    # built from options ON THE HIGHER-CARRY CURRENCY (the higher-rate leg, which
    # sits at a forward discount), priced in the QUOTE currency on the original
    # pair. A call on the high-carry ccy is a PAIR PUT when the QUOTE ccy is
    # high-carry (fwd > spot), and a PAIR CALL when the BASE ccy is high-carry
    # (fwd < spot). The spread is long ATM-fwd, short ATM-spot; its payoff is
    # capped at |fwd - spot|, so
    #   ratio = |fwd - spot| / (ATM-fwd premium - ATM-spot premium)  > 1
    # by no-arbitrage (the spread cost is the discounted, hence smaller, payoff).
    atmfsratio = None
    if fwd > 0 and spot > 0 and not math.isclose(fwd, spot):
        # Each leg is a vanilla on the high-carry ccy, so each prices at its own
        # strike's smile vol (vol_at_strike, forward = fwd). With a FlatSurface
        # both vols collapse to `vol` and the value is byte-identical to legacy.
        v_atmf = eff_surface.vol_at_strike(fwd, fwd, horizon_days)
        v_atms = eff_surface.vol_at_strike(spot, fwd, horizon_days)
        if fwd > spot:
            # quote ccy high-carry → call on quote = pair put
            atmf = put_value(spot, fwd, T, v_atmf, r_d, r_f)    # ATM-fwd put
            atms = put_value(spot, spot, T, v_atms, r_d, r_f)   # ATM-spot put
        else:
            # base ccy high-carry → call on base = pair call
            atmf = call_value(spot, fwd, T, v_atmf, r_d, r_f)   # ATM-fwd call
            atms = call_value(spot, spot, T, v_atms, r_d, r_f)  # ATM-spot call
        denom = atmf - atms
        if denom > 0:
            atmfsratio = abs(fwd - spot) / denom

    return MarketState(
        spot=spot,
        fwd=fwd,
        vol=vol,
        T=T,
        r_d=r_d,
        r_f=r_f,
        c=c,
        carry_regime=carry_regime,
        target_z=target_z,
        target_z_spot=target_z_spot,
        atmfsratio=atmfsratio,
        put_call=put_call,
        with_carry=with_carry,
        surface=surface,
    )
