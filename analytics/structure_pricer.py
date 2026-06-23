"""
Structure variant pricer.

Prices the concrete instances defined in structure_variants.json.
Premiums (inception cashflows) are expressed as a fraction of entry spot.
Payoff-at-target is base-ccy and converted at the PREVAILING spot at that scenario
— i.e. divided by `target` (spot equals target in that scenario), matching the
scenario grid's prevailing-spot convention. The European digital is recommended
as a base-ccy cash-or-nothing trade (fixed 1-unit-of-base payout), so its premium
is a base-ccy % and its payoff at target is exactly 100% — see `_digital`.

Smile vols
----------
By default every leg is priced at the ATM vol (ms.vol) — the flat-smile
approximation. When a VolSurface is passed to price_variants(smile=...), the
leg-based structures (vanilla, 1x1 / 1x1.5 / 1x2 spreads, seagull) instead price
each leg at its own interpolated smile vol (delta-defined legs at their
delta-pillar vol; strike-defined legs at vol_at_strike). The max-loss helper
``_today_package_value_pct`` likewise reprices its vanilla legs under a
sticky-delta smile when a surface is supplied.

The European package (european_digital and european_rko) is also smile-aware at
*entry*: each is threaded a strike→vol seam (``vol_fn``) so the digital prices at
the skew-consistent value ``DF·N(d2(σ(K))) − vega·σ′(K)`` and the ERKO's two
vanilla legs price at their own strike vol. A flat/None surface collapses the
seam to the scalar ATM vol, reproducing the legacy price byte-for-byte.

Still flat by design: the path-dependent RKO and digital+RKO pricers, and *all*
scenario-MtM legs for the digital / RKO / european-RKO family (barrier/binary
scenario skew is out of scope) — see analytics/scenario_pricer.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from analytics.market_state import MarketState
from analytics.strike_resolver import otm_call_strike, otm_put_strike
from pricing.black_scholes import black76_call, black76_put, call_mtm, call_value, put_mtm, put_value
from pricing.digital import SmileArbitrageError, digital_call, digital_put
from pricing.digital_rko import digital_rko_call, digital_rko_put
from pricing.european_rko import european_rko_call, european_rko_put

if TYPE_CHECKING:
    from analytics.vol_surface import SmileInterpolator


class _VolModel:
    """Resolves a per-leg implied vol.

    Flat (``smile is None``) returns the ATM vol for every strike/delta, which
    reproduces the legacy flat-smile behaviour byte-for-byte. With a smile, it
    delegates to the interpolator: delta-defined legs use ``vol_at_delta`` (the
    delta is the smile pillar coordinate), strike-defined legs use
    ``vol_at_strike``.
    """

    def __init__(self, flat_vol, smile=None, F=None, horizon_days=None):
        self._flat = flat_vol
        self._smile = smile
        self._F = F
        self._h = horizon_days

    @property
    def surface(self):
        """The underlying VolSurface, or None for the flat model."""
        return self._smile

    def at_strike(self, K: float) -> float:
        if self._smile is None:
            return self._flat
        return self._smile.vol_at_strike(K, self._F, self._h)

    def at_delta(self, delta: float, is_call: bool) -> float:
        if self._smile is None:
            return self._flat
        # vol_at_delta takes a call delta (positive) or put delta (negative).
        return self._smile.vol_at_delta(delta if is_call else -delta, self._h)

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
    # Sizing — populated when loss_budget passed to price_variants(), else None.
    # All amounts in BASE currency units (e.g. USD for USDBRL, EUR for EURPLN).
    structure_notional: float | None = None    # base ccy notional sized so max loss = loss_budget
    net_premium_ccy: float | None = None       # premium in base ccy
    payoff_at_target_ccy: float | None = None  # gross payoff at target in base ccy
    max_loss_ccy: float | None = None          # max loss in base ccy (= loss_budget by construction)


def price_variants(
    ms: MarketState,
    structure_id: str,
    target: float | None = None,
    is_call: bool = True,
    stop_price: float | None = None,
    loss_budget: float | None = None,
    linear_notional: float = 100.0,
    smile: "SmileInterpolator | None" = None,
    warnings: list[str] | None = None,
    variants_override: list[dict] | None = None,
) -> list[PricedVariant]:
    """
    Price all defined variants for a structure. Returns [] if no variants defined.

    If loss_budget is provided (in base ccy units), each variant is sized so its
    max loss equals the loss budget. The dollar-equivalent fields on PricedVariant
    (structure_notional, net_premium_ccy, payoff_at_target_ccy, max_loss_ccy) are
    populated. If loss_budget is None, those fields remain None.

    If a ``smile`` (SmileInterpolator) is supplied, the leg-based structures price
    each leg at its interpolated smile vol; otherwise every leg uses ms.vol (flat
    smile). See module docstring for the Phase 1 scope.

    Digital-bearing variants (``european_digital``, ``european_rko``) whose smile
    price triggers a ``SmileArbitrageError`` are dropped from the result rather
    than emitted with an unreliable mark. If a ``warnings`` list is supplied, a
    human-readable note is appended for each dropped variant.

    ``variants_override`` lets a caller price a *specific* synthetic variant (or
    variants) instead of the curated ``structure_variants.json`` menu — used by the
    agentic ``price_structure`` tool, which builds the variant dict from a parsed
    PM request. Each override dict must carry the same keys the curated menu uses
    for that ``structure_id``. When None (default), the curated menu is loaded, so
    every existing caller is unchanged.
    """
    if variants_override is not None:
        variants = variants_override
    else:
        cfg = _load_variants()
        if structure_id not in cfg:
            return []
        variants = cfg[structure_id]
    F, vol, T, r_d, r_f, spot = ms.fwd, ms.vol, ms.T, ms.r_d, ms.r_f, ms.spot
    DF = math.exp(-r_d * T)
    vol_sqrtT = vol * math.sqrt(T)
    vm = _VolModel(vol, smile=smile, F=F, horizon_days=round(T * 365))

    if structure_id == "vanilla":
        result = _vanilla(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm)
    elif structure_id == "1x1_spread":
        result = _spread(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm)
    elif structure_id == "1x1.5_spread":
        result = _1x1p5(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm)
    elif structure_id == "1x2_spread":
        result = _1x2(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm)
    elif structure_id == "1x2x1_spread":
        result = _1x2x1(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm)
    elif structure_id == "european_rko":
        result = _european_rko(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm, warnings)
    elif structure_id == "seagull":
        result = _seagull(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, stop_price, vm)
    elif structure_id == "european_digital":
        result = _digital(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm, warnings)
    elif structure_id == "european_digital_rko":
        result = _digital_rko(variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target)
    else:
        return []

    if loss_budget is not None and loss_budget > 0:
        for pv in result:
            _size_variant(pv, loss_budget, linear_notional)

    return result


def _size_variant(pv: PricedVariant, loss_budget: float, linear_notional: float = 100.0) -> None:
    """Populate dollar-equivalent fields on a PricedVariant given a loss budget.

    Sizing rule (premium-aware), with the structure's 1x-leg notional
    (``structure_notional``) capped/floored relative to the linear notional:

    - **Net credit** (``net_premium_pct < 0``): premium can't bound the loss, so
      loss-budget division is meaningless. Fix the 1x-leg notional at
      ``10 × linear_notional``.
    - **Net debit / zero-cost** (``net_premium_pct >= 0``): size so max loss
      equals the loss budget (``loss_budget / max_loss_pct``), but **cap** the
      notional at ``10 × linear_notional``. A ~zero max loss (the denominator
      would blow up) lands directly on the cap.

    All other dollar amounts are derived linearly from the resulting notional.
    """
    cap = 10.0 * linear_notional
    if pv.net_premium_pct < 0:
        notional = cap
    elif pv.max_loss_pct is None or pv.max_loss_pct < 1e-9:
        notional = cap
    else:
        notional = min(loss_budget / pv.max_loss_pct, cap)
    pv.structure_notional = notional
    pv.net_premium_ccy = pv.net_premium_pct * notional
    pv.max_loss_ccy = pv.max_loss_pct * notional if pv.max_loss_pct is not None else None
    if pv.payoff_at_target_pct is not None:
        pv.payoff_at_target_ccy = pv.payoff_at_target_pct * notional


def _spot_for_forward_today(fwd_today: float, T: float, r_d: float, r_f: float) -> float:
    return fwd_today * math.exp(-(r_d - r_f) * T)


def _spread_strikes_from_deltas(
    F: float,
    vol: float,
    T: float,
    is_call: bool,
    long_delta: float,
    short_delta: float,
) -> tuple[float, float]:
    if is_call:
        return (
            otm_call_strike(F, vol, T, long_delta),
            otm_call_strike(F, vol, T, short_delta),
        )
    return (
        otm_put_strike(F, vol, T, long_delta),
        otm_put_strike(F, vol, T, short_delta),
    )


def _delta_spread_legs(
    F: float,
    T: float,
    is_call: bool,
    long_delta: float,
    short_delta: float,
    vm: _VolModel,
) -> tuple[float, float, float, float]:
    """Resolve (K_long, K_short, vol_long, vol_short) for a delta-pair spread.

    Each leg's strike is resolved with that leg's own (delta-pillar) vol, and the
    same vol is returned for pricing — so under a flat vol model the result is
    identical to the legacy ATM-vol path.
    """
    v_long = vm.at_delta(long_delta, is_call)
    v_short = vm.at_delta(short_delta, is_call)
    resolve = otm_call_strike if is_call else otm_put_strike
    return (
        resolve(F, v_long, T, long_delta),
        resolve(F, v_short, T, short_delta),
        v_long,
        v_short,
    )


def _resolve_ratio_spread_strikes(
    variant: dict,
    *,
    F: float,
    vol: float,
    T: float,
    vol_sqrtT: float,
    is_call: bool,
    target: float | None,
    vm: _VolModel,
) -> tuple[float, float, float, float] | None:
    """Resolve (K1, K2, vol1, vol2) for a ratio-spread variant.

    Supports target-anchored variants (`long_type`) and delta-pair variants
    (`long_delta` / `short_delta`) that mirror the 1x1 spread family. Strike-
    defined legs (ATMF / half-sigma / target) take vol_at_strike; delta-defined
    legs take their delta-pillar vol.
    """
    if "long_delta" in variant and "short_delta" in variant:
        return _delta_spread_legs(
            F, T, is_call, variant["long_delta"], variant["short_delta"], vm
        )

    if target is None:
        return None

    target_z = abs(math.log(target / F) / vol_sqrtT) if vol_sqrtT > 0 else 0.0
    if target_z < variant.get("min_target_z", 0.0):
        return None

    if variant["long_type"] == "atmf":
        K1 = F
    else:  # half_sigma toward target
        K1 = F * math.exp(0.5 * vol_sqrtT) if is_call else F * math.exp(-0.5 * vol_sqrtT)

    return K1, target, vm.at_strike(K1), vm.at_strike(target)


def _today_package_value_pct(
    *,
    structure_id: str,
    strikes: list[float],
    fwd_today: float,
    T: float,
    vol: float,
    r_d: float,
    r_f: float,
    spot: float,
    is_call: bool,
    wing_ratio: float | None = None,
    surface: object | None = None,
) -> float:
    scenario_spot = _spot_for_forward_today(fwd_today, T, r_d, r_f)

    # Sticky-delta per-leg vol: the package is valued at the stressed forward
    # (fwd_today), so each fixed strike is re-deltaed there and picks up the
    # surface's skew. Flat/None surface → every leg uses `vol` (byte-identical).
    if surface is not None:
        from analytics.vol_surface import smile_skew_spread
        h_days = max(round(T * 365), 1)

        def lv(strike: float) -> float:
            return max(vol + smile_skew_spread(surface, strike, fwd_today, h_days), 0.01)
    else:
        def lv(strike: float) -> float:
            return vol

    if structure_id == "1x1.5_spread":
        if is_call:
            raw = call_mtm(scenario_spot, strikes[0], T, lv(strikes[0]), r_d, r_f) - 1.5 * call_mtm(scenario_spot, strikes[1], T, lv(strikes[1]), r_d, r_f)
        else:
            raw = put_mtm(scenario_spot, strikes[0], T, lv(strikes[0]), r_d, r_f) - 1.5 * put_mtm(scenario_spot, strikes[1], T, lv(strikes[1]), r_d, r_f)
    elif structure_id == "1x2_spread":
        if is_call:
            raw = call_mtm(scenario_spot, strikes[0], T, lv(strikes[0]), r_d, r_f) - 2.0 * call_mtm(scenario_spot, strikes[1], T, lv(strikes[1]), r_d, r_f)
        else:
            raw = put_mtm(scenario_spot, strikes[0], T, lv(strikes[0]), r_d, r_f) - 2.0 * put_mtm(scenario_spot, strikes[1], T, lv(strikes[1]), r_d, r_f)
    elif structure_id == "seagull":
        ratio = wing_ratio or 0.0
        if is_call:
            spread = call_mtm(scenario_spot, strikes[0], T, lv(strikes[0]), r_d, r_f) - call_mtm(scenario_spot, strikes[1], T, lv(strikes[1]), r_d, r_f)
            wing = put_mtm(scenario_spot, strikes[2], T, lv(strikes[2]), r_d, r_f)
        else:
            spread = put_mtm(scenario_spot, strikes[0], T, lv(strikes[0]), r_d, r_f) - put_mtm(scenario_spot, strikes[1], T, lv(strikes[1]), r_d, r_f)
            wing = call_mtm(scenario_spot, strikes[2], T, lv(strikes[2]), r_d, r_f)
        raw = spread - ratio * wing
    else:
        raise ValueError(f"Unsupported structure for today-value helper: {structure_id}")
    return abs(raw) / spot


# ---------------------------------------------------------------------------
# Vanilla
# ---------------------------------------------------------------------------

def _vanilla(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm
) -> list[PricedVariant]:
    result = []
    for v in variants:
        delta = v["delta"]
        leg_vol = vm.at_delta(delta, is_call)
        if is_call:
            K = otm_call_strike(F, leg_vol, T, delta)
            prem = black76_call(F, K, T, leg_vol, DF)
            be = K + prem
        else:
            K = otm_put_strike(F, leg_vol, T, delta)
            prem = black76_put(F, K, T, leg_vol, DF)
            be = K - prem

        prem_pct = prem / spot
        payoff_pct, rr = None, None
        if target is not None:
            raw = max(target - K, 0.0) if is_call else max(K - target, 0.0)
            payoff_pct = raw / target
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
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm
) -> list[PricedVariant]:
    result = []
    for v in variants:
        K_long, K_short, v_long, v_short = _delta_spread_legs(
            F, T, is_call, v["long_delta"], v["short_delta"], vm
        )
        if is_call:
            prem_long  = black76_call(F, K_long,  T, v_long,  DF)
            prem_short = black76_call(F, K_short, T, v_short, DF)
        else:
            prem_long  = black76_put(F, K_long,  T, v_long,  DF)
            prem_short = black76_put(F, K_short, T, v_short, DF)

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
            payoff_pct = raw / target
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
# 1x1.5 Spread  (long 1 × long leg, short 1.5 × short leg)
# ---------------------------------------------------------------------------

def _1x1p5(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm
) -> list[PricedVariant]:
    """Price 1x1.5 ratio spreads from target-based or delta-pair variants."""
    result = []
    for v in variants:
        resolved = _resolve_ratio_spread_strikes(
            v, F=F, vol=vol, T=T, vol_sqrtT=vol_sqrtT, is_call=is_call, target=target, vm=vm
        )
        if resolved is None:
            continue
        K1, K2, v1, v2 = resolved

        if is_call:
            prem1 = black76_call(F, K1, T, v1, DF)
            prem2 = black76_call(F, K2, T, v2, DF)
        else:
            prem1 = black76_put(F, K1, T, v1, DF)
            prem2 = black76_put(F, K2, T, v2, DF)

        net_prem = prem1 - 1.5 * prem2
        prem_pct = net_prem / spot
        is_zero_cost = abs(net_prem) < 0.0001 * spot

        breakeven = None
        if not is_zero_cost and net_prem > 0:
            breakeven = (K1 + net_prem) if is_call else (K1 - net_prem)

        payoff_pct = None
        if target is not None:
            gross_at_target = max(target - K1, 0.0) if is_call else max(K1 - target, 0.0)
            payoff_pct = gross_at_target / target
        rr = (
            payoff_pct / prem_pct
            if (payoff_pct is not None and not is_zero_cost and prem_pct > 1e-8)
            else None
        )

        # Max loss = net premium paid (the open tail beyond the short strike is not
        # capitalised into the sizing max-loss). Drives the notional via _size_variant.
        max_loss_pct = abs(prem_pct)

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K1, K2],
            barrier=None,
            net_premium_pct=prem_pct,
            breakeven=breakeven,
            payoff_at_target_pct=payoff_pct,
            rr_at_target=rr,
            max_loss_pct=max_loss_pct,
            wing_ratio=None,
            is_zero_cost=is_zero_cost,
        ))
    return result


# 1x2 Spread  (long 1 × long leg, short 2 × short leg)
# ---------------------------------------------------------------------------

def _1x2(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm
) -> list[PricedVariant]:
    result = []
    for v in variants:
        resolved = _resolve_ratio_spread_strikes(
            v, F=F, vol=vol, T=T, vol_sqrtT=vol_sqrtT, is_call=is_call, target=target, vm=vm
        )
        if resolved is None:
            continue
        K1, K2, v1, v2 = resolved

        if is_call:
            prem1 = black76_call(F, K1, T, v1, DF)
            prem2 = black76_call(F, K2, T, v2, DF)
        else:
            prem1 = black76_put(F, K1, T, v1, DF)
            prem2 = black76_put(F, K2, T, v2, DF)

        net_prem = prem1 - 2.0 * prem2
        prem_pct = net_prem / spot
        is_zero_cost = abs(net_prem) < 0.0001 * spot

        breakeven = None
        if not is_zero_cost and net_prem > 0:
            breakeven = (K1 + net_prem) if is_call else (K1 - net_prem)

        payoff_pct = None
        if target is not None:
            gross_at_target = max(target - K1, 0.0) if is_call else max(K1 - target, 0.0)
            payoff_pct = gross_at_target / target
        rr = (
            payoff_pct / prem_pct
            if (payoff_pct is not None and not is_zero_cost and prem_pct > 1e-8)
            else None
        )

        # Max loss = net premium paid (open tail beyond the short strike not capitalised).
        max_loss_pct = abs(prem_pct)

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K1, K2],
            barrier=None,
            net_premium_pct=prem_pct,
            breakeven=breakeven,
            payoff_at_target_pct=payoff_pct,
            rr_at_target=rr,
            max_loss_pct=max_loss_pct,
            wing_ratio=None,
            is_zero_cost=is_zero_cost,
        ))
    return result


# ---------------------------------------------------------------------------
# 1x2x1  (butterfly: the 1x2 with a long wing that caps the open tail)
# ---------------------------------------------------------------------------

def _1x2x1(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm
) -> list[PricedVariant]:
    """Butterfly built from the 1x2's two strikes plus an equidistant long wing.

    Strikes: long 1× K1, short 2× K2, long 1× K3 with K3 = 2·K2 − K1 (so the
    strike spacing is symmetric: K3−K2 = K2−K1). The third long leg caps the
    1x2's open tail beyond K2, so max loss = net premium paid (a long butterfly's
    expiry payoff is non-negative). K1/K2 come from the same resolver the 1x2
    uses, so every 1x2 variant has a matching 1x2x1.
    """
    result = []
    for v in variants:
        resolved = _resolve_ratio_spread_strikes(
            v, F=F, vol=vol, T=T, vol_sqrtT=vol_sqrtT, is_call=is_call, target=target, vm=vm
        )
        if resolved is None:
            continue
        K1, K2, v1, v2 = resolved
        K3 = 2.0 * K2 - K1          # equidistant long wing
        if K3 <= 0:
            continue
        v3 = vm.at_strike(K3)

        if is_call:
            prem1 = black76_call(F, K1, T, v1, DF)
            prem2 = black76_call(F, K2, T, v2, DF)
            prem3 = black76_call(F, K3, T, v3, DF)
        else:
            prem1 = black76_put(F, K1, T, v1, DF)
            prem2 = black76_put(F, K2, T, v2, DF)
            prem3 = black76_put(F, K3, T, v3, DF)

        net_prem = prem1 - 2.0 * prem2 + prem3
        prem_pct = net_prem / spot
        is_zero_cost = abs(net_prem) < 0.0001 * spot

        breakeven = None
        if not is_zero_cost and net_prem > 0:
            breakeven = (K1 + net_prem) if is_call else (K1 - net_prem)

        payoff_pct = None
        if target is not None:
            if is_call:
                gross = (max(target - K1, 0.0) - 2.0 * max(target - K2, 0.0)
                         + max(target - K3, 0.0))
            else:
                gross = (max(K1 - target, 0.0) - 2.0 * max(K2 - target, 0.0)
                         + max(K3 - target, 0.0))
            payoff_pct = gross / target
        rr = (
            payoff_pct / prem_pct
            if (payoff_pct is not None and not is_zero_cost and prem_pct > 1e-8)
            else None
        )

        # Long butterfly: expiry payoff is non-negative, so max loss = net premium
        # paid (the wing caps the 1x2's open tail). Clamp at 0 for a (rare) credit.
        max_loss_pct = max(prem_pct, 0.0)

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K1, K2, K3],
            barrier=None,
            net_premium_pct=prem_pct,
            breakeven=breakeven,
            payoff_at_target_pct=payoff_pct,
            rr_at_target=rr,
            max_loss_pct=max_loss_pct,
            wing_ratio=None,
            is_zero_cost=is_zero_cost,
        ))
    return result


# ---------------------------------------------------------------------------
# Seagull  (spread + zero-cost OTM wing on opposite side)
# ---------------------------------------------------------------------------

def _seagull(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, stop_price, vm
) -> list[PricedVariant]:
    result = []
    for v in variants:
        ld, sd, wd = v["spread_long"], v["spread_short"], v["wing_delta"]
        # Spread legs in the view direction; wing on the opposite side.
        v1 = vm.at_delta(ld, is_call)
        v2 = vm.at_delta(sd, is_call)
        v3 = vm.at_delta(wd, not is_call)

        if is_call:
            K1 = otm_call_strike(F, v1, T, ld)   # long call (lower K)
            K2 = otm_call_strike(F, v2, T, sd)   # short call (higher K)
            prem1 = black76_call(F, K1, T, v1, DF)
            prem2 = black76_call(F, K2, T, v2, DF)
            # Wing: OTM put (opposite direction)
            K3 = otm_put_strike(F, v3, T, wd)
            prem3 = black76_put(F, K3, T, v3, DF)
        else:
            K1 = otm_put_strike(F, v1, T, ld)    # long put (higher K)
            K2 = otm_put_strike(F, v2, T, sd)    # short put (lower K)
            prem1 = black76_put(F, K1, T, v1, DF)
            prem2 = black76_put(F, K2, T, v2, DF)
            # Wing: OTM call (opposite direction)
            K3 = otm_call_strike(F, v3, T, wd)
            prem3 = black76_call(F, K3, T, v3, DF)

        spread_cost = prem1 - prem2
        wing_ratio = (spread_cost / prem3) if prem3 > 1e-8 else 0.0

        if stop_price is not None:
            max_loss = _today_package_value_pct(
                structure_id="seagull",
                strikes=[K1, K2, K3],
                fwd_today=stop_price,
                T=T,
                vol=vol,
                r_d=r_d,
                r_f=r_f,
                spot=spot,
                is_call=is_call,
                wing_ratio=wing_ratio,
                surface=vm.surface,
            )
        else:
            max_loss = 0.0

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
            payoff_pct = (sp_payoff + wing_payoff) / target

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
# European digital — recommended as a BASE-CCY (e.g. USD) cash-or-nothing trade
# ---------------------------------------------------------------------------

def _digital(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm, warnings=None
) -> list[PricedVariant]:
    """European digital as a base-ccy cash-or-nothing structure.

    As a recommended trade the digital pays a FIXED 1 unit of base ccy (e.g. USD)
    if it finishes in the money, so the premium is a base-ccy % and the payoff at
    target is exactly **100%** — not spot/target. (The pricing module's quote-cash
    `digital_call/put` are unchanged; only this structure-level treatment differs.)

    Built from the asset-or-nothing identity, so it reprices in base-ccy terms and
    inherits the smile + SmileArbitrageError guard from the quote-cash digital leg:
        AON_call = call(K) + K·digital_call(payout=1 quote)   → /spot = USD digital call
        AON_put  = K·digital_put(payout=1 quote) − put(K)     → /spot = USD digital put
    Flat surface → vol_fn is None and every leg collapses to the scalar `vol`.
    """
    result = []
    vol_fn = vm.at_strike if vm.surface is not None else None

    def premium_usd(K: float) -> float:
        """Base-ccy (USD) cash-or-nothing premium as a fraction of base notional."""
        sig = vol_fn(K) if vol_fn is not None else vol
        if is_call:
            aon = (call_value(spot, K, T, sig, r_d, r_f)
                   + K * digital_call(spot, K, T, vol, r_d, r_f, payout=1.0, vol_fn=vol_fn))
        else:
            aon = (K * digital_put(spot, K, T, vol, r_d, r_f, payout=1.0, vol_fn=vol_fn)
                   - put_value(spot, K, T, sig, r_d, r_f))
        return aon / spot

    for v in variants:
        tgt_pct = v["target_prem_pct"]  # target premium, base-ccy fraction of notional

        # Bisect for K so the base-ccy premium ≈ tgt_pct. Bracket symmetrically
        # around the FORWARD (the digital's risk-neutral centre), not spot: on a
        # high-carry pair the forward sits far from spot, so a 10–30% strike can
        # land between spot and the forward (downside *to the forward*). Spot-
        # anchored bounds would clip those strikes and rail every variant to the
        # boundary. The span covers premium ≈ 0 → ≈ DF on either side, and the
        # (monotonic) bisection finds the strike. Symmetric for calls and puts.
        span = math.exp(4.0 * vol_sqrtT)
        K_lo, K_hi = F / span, F * span

        # Smile arbitrage anywhere across the strike search makes the digital
        # unreliable — drop the variant rather than emit a bogus mark.
        try:
            K = _bisect_digital_usd(premium_usd, K_lo, K_hi, tgt_pct, is_call)
            prem_pct = premium_usd(K)
        except SmileArbitrageError as e:
            if warnings is not None:
                warnings.append(f"variant '{v['label']}' not priced — {e}")
            continue

        payoff_pct, rr = None, None
        if target is not None:
            itm = (target > K) if is_call else (target < K)
            if itm:
                # Fixed base-ccy payout → payoff at target is exactly 100%.
                payoff_pct = 1.0
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
                # Fixed quote-ccy payout of entry spot → base ccy = spot/target.
                payoff_pct = spot / target
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
# European reverse knock-out  (expiry-only KO at delta-defined barrier)
# ---------------------------------------------------------------------------

def _european_rko(
    variants, F, vol, T, DF, r_d, r_f, spot, vol_sqrtT, is_call, target, vm, warnings=None
) -> list[PricedVariant]:
    if target is None:
        return []

    fn_price = european_rko_call if is_call else european_rko_put
    resolve = otm_call_strike if is_call else otm_put_strike
    # Smile seam: price each leg at its own strike vol. Flat surface → vol_fn is
    # None and every leg collapses to the scalar `vol`, reproducing the legacy
    # flat price byte-for-byte.
    vol_fn = vm.at_strike if vm.surface is not None else None
    result = []

    for v in variants:
        # Both legs are delta-defined: the long strike at long_delta and the KO
        # barrier at the (further-OTM) barrier delta. Each strike is placed at
        # its own delta-pillar vol (matching the spread family); flat surface →
        # at_delta returns the scalar vol so placement is unchanged.
        long_delta = v["long_delta"]
        barrier_delta = v["barrier"]
        K = resolve(F, vm.at_delta(long_delta, is_call), T, long_delta)
        barrier = resolve(F, vm.at_delta(barrier_delta, is_call), T, barrier_delta)

        # The digital strip can hit a smile arbitrage → drop the variant.
        try:
            prem = fn_price(spot, K, barrier, T, vol, r_d, r_f, vol_fn=vol_fn)
        except SmileArbitrageError as e:
            if warnings is not None:
                warnings.append(f"variant '{v['label']}' not priced — {e}")
            continue
        prem_pct = prem / spot

        breakeven = None
        if prem > 1e-8:
            breakeven = (K + prem) if is_call else (K - prem)

        payoff_pct, rr = None, None
        raw = max(target - K, 0.0) if is_call else max(K - target, 0.0)
        not_ko = (target < barrier) if is_call else (target > barrier)
        if raw > 0.0 and not_ko:
            payoff_pct = raw / target
            rr = payoff_pct / prem_pct if prem_pct > 1e-8 else None
        else:
            payoff_pct = 0.0
            rr = 0.0

        result.append(PricedVariant(
            variant_label=v["label"],
            strikes=[K, barrier],
            barrier=barrier,
            net_premium_pct=prem_pct,
            breakeven=breakeven,
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

def _bisect_digital_usd(premium_fn, K_lo, K_hi, tgt_pct, is_call, tol=1e-6) -> float:
    """Solve for the strike whose base-ccy digital premium ≈ tgt_pct.

    `premium_fn(K)` returns the base-ccy premium (fraction of notional). For a
    call it decreases as K rises; for a put it increases as K rises toward spot.
    May raise SmileArbitrageError (propagated to the caller to drop the variant).
    """
    for _ in range(60):
        K_mid = 0.5 * (K_lo + K_hi)
        val = premium_fn(K_mid)
        if abs(val - tgt_pct) < tol:
            return K_mid
        if is_call:
            if val > tgt_pct:
                K_lo = K_mid   # premium too high → push strike further OTM (higher)
            else:
                K_hi = K_mid
        else:
            if val > tgt_pct:
                K_hi = K_mid   # put premium too high → push strike further OTM (lower)
            else:
                K_lo = K_mid
    return 0.5 * (K_lo + K_hi)


def _bisect_strike(fn, spot, K_lo, K_hi, T, vol, r_d, r_f, target_prem, is_call, tol=1e-6, vol_fn=None) -> float:
    for _ in range(60):
        K_mid = 0.5 * (K_lo + K_hi)
        val = fn(spot, K_mid, T, vol, r_d, r_f, payout=spot, vol_fn=vol_fn)
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
