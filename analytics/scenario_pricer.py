"""
Scenario-based MtM pricer.

Takes a PricedVariant (strikes/barrier already fixed at trade entry) and a list
of generated scenarios, and prices the variant in each scenario using the
existing _mtm functions.

Invariants (enforced by caller):
  - r_d, r_f, strikes, barrier: unchanged across scenarios
  - scenario_spot and scenario_vol come from the scenario's derived block
  - At EXPIRY (remaining_time=0), pricing functions return intrinsic value
"""

from __future__ import annotations

import math

from analytics.structure_pricer import PricedVariant
from pricing.black_scholes import call_mtm, put_mtm
from pricing.digital import digital_call_mtm, digital_put_mtm
from pricing.digital_rko import digital_rko_call_mtm, digital_rko_put_mtm
from pricing.european_rko import european_rko_call_mtm, european_rko_put_mtm


def price_linear_scenarios(
    scenarios: list[dict],
    trade_inputs: dict,
    is_call: bool,
    notional: float,
    max_loss_ccy: float | None = None,
) -> list[dict]:
    """Price a delta-1 linear position through the scenario grid.

    The position is long the base currency when `is_call=True` and short the base
    currency when `is_call=False`. P&L is capped at the provided max loss, if any.

    A delta-1 expression of an FX view is a forward/NDF transacted at the entry
    forward, so its mark-to-market is the change in the forward-to-expiry vs the
    entry forward (PV'd over remaining time) — NOT the change in spot. Measuring
    against spot would book the carry roll-down as phantom P&L, which is large on
    high-carry pairs (USDTRY, USDBRL) and makes the "F" column non-zero.

    The quote-ccy MtM, N·(F_t − F0)·DF, is converted to base ccy at the PREVAILING
    spot at each checkpoint (scenario_spot) — i.e. the fixing spot at expiry — not
    at entry spot. At expiry this reproduces the NDF cash settlement
    (S_T − F0)/S_T per unit base notional.
    """
    entry_fwd: float = trade_inputs["forward"]
    r_d: float = trade_inputs["r_d"]
    direction = 1.0 if is_call else -1.0
    max_loss_pct = (max_loss_ccy / notional) if (max_loss_ccy is not None and notional > 0) else None

    results = []
    for sc in scenarios:
        d = sc["derived"]
        scenario_spot: float = d["scenario_spot"]
        scenario_fwd: float = d["scenario_fwd"]
        tau: float = d["remaining_time"]
        pv = math.exp(-r_d * tau)
        raw_pnl_pct = direction * ((scenario_fwd - entry_fwd) / scenario_spot) * pv
        pnl_pct = max(raw_pnl_pct, -max_loss_pct) if max_loss_pct is not None else raw_pnl_pct
        price_pct = pnl_pct
        pnl_ccy = pnl_pct * notional
        price_ccy = pnl_ccy

        results.append({
            "structure_id": "linear",
            "variant_label": "Delta 1 (max-loss capped)",
            "scenario_id": sc["id"],
            "row": sc["row"],
            "col": sc["col"],
            "time_fraction": sc["time_fraction"],
            "fwd_rule": sc["fwd_rule"],
            "vol_rule": sc["vol_rule"],
            "skew_rule": sc["skew_rule"],
            "tags": sc["tags"],
            "elapsed_time": d["elapsed_time"],
            "remaining_time": d["remaining_time"],
            "scenario_fwd": d["scenario_fwd"],
            "scenario_spot": scenario_spot,
            "vol_shift": "±4% vol" if sc.get("vol_shifts") else d["vol_shift"],
            "scenario_vol": trade_inputs["implied_vol"] if sc.get("vol_shifts") else d["scenario_vol"],
            "skew_multiplier": d["skew_multiplier"],
            "structure_notional": notional,
            "price_pct": price_pct,
            "pnl_pct": pnl_pct,
            "price_ccy": price_ccy,
            "pnl_ccy": pnl_ccy,
        })

    return results


def price_scenarios(
    variant: PricedVariant,
    structure_id: str,
    scenarios: list[dict],
    trade_inputs: dict,
    is_call: bool,
    surface: object | None = None,
) -> list[dict]:
    """
    Price `variant` in every scenario. Returns one row per scenario with
    price_pct (option value as fraction of entry spot) and
    pnl_pct (price_pct minus entry premium paid).

    trade_inputs must contain: spot, r_d, r_f.

    ``surface`` is an optional VolSurface. When supplied, the vanilla legs
    (vanilla / 1x1 / 1x1.5 / 1x2 / seagull) are repriced under a sticky-delta
    smile: each leg keeps the scenario's ATM vol level but picks up the surface's
    skew at the leg's delta under the scenario forward. Digital / digital_rko /
    european_rko legs stay on the flat scenario vol. When None (or a flat
    surface), every leg uses the scalar scenario vol — byte-identical to legacy.
    """
    entry_spot: float = trade_inputs["spot"]
    r_d: float = trade_inputs["r_d"]
    r_f: float = trade_inputs["r_f"]
    entry_premium_pct: float = variant.net_premium_pct
    notional: float | None = variant.structure_notional   # may be None

    results = []
    for sc in scenarios:
        d = sc["derived"]
        scenario_spot: float = d["scenario_spot"]
        tau: float = d["remaining_time"]
        vol_shifts = sc.get("vol_shifts")
        if vol_shifts:
            raws = []
            for shift in vol_shifts:
                try:
                    raw = _value_variant(
                        structure_id, variant,
                        scenario_spot, max(trade_inputs["implied_vol"] + shift, 0.01), tau,
                        r_d, r_f, entry_spot, is_call, surface,
                    )
                except Exception:
                    raw = 0.0
                raws.append(raw)
            raw = min(raws) if raws else 0.0
            vol_shift = "±4% vol"
            scenario_vol = trade_inputs["implied_vol"]
        else:
            scenario_vol = d["scenario_vol"]
            try:
                raw = _value_variant(
                    structure_id, variant,
                    scenario_spot, scenario_vol, tau,
                    r_d, r_f, entry_spot, is_call, surface,
                )
            except Exception:
                raw = 0.0
            vol_shift = d["vol_shift"]

        # Convert the quote-ccy option MtM to base ccy at the PREVAILING spot at
        # this checkpoint (matches the delta-1/NDF settlement convention). The
        # premium leg stays normalised to entry spot — it is an inception cashflow.
        price_pct = raw / scenario_spot
        pnl_pct = price_pct - entry_premium_pct
        price_ccy = (price_pct * notional) if notional is not None else None
        pnl_ccy = (pnl_pct * notional) if notional is not None else None

        results.append({
            "structure_id": structure_id,
            "variant_label": variant.variant_label,
            "scenario_id": sc["id"],
            "row": sc["row"],
            "col": sc["col"],
            "time_fraction": sc["time_fraction"],
            "fwd_rule": sc["fwd_rule"],
            "vol_rule": sc["vol_rule"],
            "skew_rule": sc["skew_rule"],
            "tags": sc["tags"],
            "elapsed_time": d["elapsed_time"],
            "remaining_time": tau,
            "scenario_fwd": d["scenario_fwd"],
            "scenario_spot": scenario_spot,
            "vol_shift": vol_shift,
            "scenario_vol": scenario_vol,
            "skew_multiplier": d["skew_multiplier"],
            "structure_notional": notional,
            "price_pct": price_pct,
            "pnl_pct": pnl_pct,
            "price_ccy": price_ccy,
            "pnl_ccy": pnl_ccy,
        })

    return results


# ---------------------------------------------------------------------------
# Per-structure valuation (locked strikes/barrier, varying market state)
# ---------------------------------------------------------------------------

def _value_variant(
    structure_id: str,
    variant: PricedVariant,
    sspot: float,
    svol: float,
    tau: float,
    r_d: float,
    r_f: float,
    entry_spot: float,
    is_call: bool,
    surface: object | None = None,
) -> float:
    """Return absolute MtM value (same currency units as entry_spot).

    Vanilla legs (vanilla / spreads / seagull) reprice under a sticky-delta
    smile when ``surface`` is supplied: each leg uses ``svol`` plus the surface's
    skew spread at the leg's strike under the scenario forward. Digital / RKO /
    european_rko legs stay on the flat ``svol`` (Phase 2 keeps them flat).
    """
    K = variant.strikes
    barrier = variant.barrier

    # Sticky-delta per-leg vol for vanilla legs. The scenario forward sets the
    # delta of each fixed strike; the surface supplies the skew at that delta.
    if surface is not None and tau > 0:
        from analytics.vol_surface import smile_skew_spread
        F_s = sspot * math.exp((r_d - r_f) * tau)
        h_days = max(round(tau * 365), 1)

        def leg_vol(strike: float) -> float:
            return max(svol + smile_skew_spread(surface, strike, F_s, h_days), 0.01)
    else:
        def leg_vol(strike: float) -> float:
            return svol

    if structure_id == "vanilla":
        if is_call:
            return call_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f)
        return put_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f)

    if structure_id == "1x1_spread":
        if is_call:
            return call_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f) - call_mtm(sspot, K[1], tau, leg_vol(K[1]), r_d, r_f)
        return put_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f) - put_mtm(sspot, K[1], tau, leg_vol(K[1]), r_d, r_f)

    if structure_id == "1x1.5_spread":
        if is_call:
            return call_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f) - 1.5 * call_mtm(sspot, K[1], tau, leg_vol(K[1]), r_d, r_f)
        return put_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f) - 1.5 * put_mtm(sspot, K[1], tau, leg_vol(K[1]), r_d, r_f)

    if structure_id == "1x2_spread":
        if is_call:
            return call_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f) - 2.0 * call_mtm(sspot, K[1], tau, leg_vol(K[1]), r_d, r_f)
        return put_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f) - 2.0 * put_mtm(sspot, K[1], tau, leg_vol(K[1]), r_d, r_f)

    if structure_id == "european_rko":
        if is_call:
            return european_rko_call_mtm(sspot, K[0], barrier, tau, svol, r_d, r_f)
        return european_rko_put_mtm(sspot, K[0], barrier, tau, svol, r_d, r_f)

    if structure_id == "seagull":
        wing_ratio = variant.wing_ratio or 0.0
        if is_call:
            # long call spread + short put wing
            spread = call_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f) - call_mtm(sspot, K[1], tau, leg_vol(K[1]), r_d, r_f)
            wing = put_mtm(sspot, K[2], tau, leg_vol(K[2]), r_d, r_f)
        else:
            # long put spread + short call wing
            spread = put_mtm(sspot, K[0], tau, leg_vol(K[0]), r_d, r_f) - put_mtm(sspot, K[1], tau, leg_vol(K[1]), r_d, r_f)
            wing = call_mtm(sspot, K[2], tau, leg_vol(K[2]), r_d, r_f)
        return spread - wing_ratio * wing

    if structure_id == "european_digital":
        # Base-ccy (USD) cash-or-nothing: pays a fixed 1 unit of base ccy if ITM.
        # Asset-or-nothing identity (flat scenario vol, per design):
        #   AON_call = call_mtm(K) + K·digital_call_mtm(K, payout=1)
        #   AON_put  = K·digital_put_mtm(K, payout=1) − put_mtm(K)
        # price_scenarios divides by scenario_spot → at expiry ITM this is exactly 1.0.
        if is_call:
            return call_mtm(sspot, K[0], tau, svol, r_d, r_f) + K[0] * digital_call_mtm(sspot, K[0], tau, svol, r_d, r_f, payout=1.0)
        return K[0] * digital_put_mtm(sspot, K[0], tau, svol, r_d, r_f, payout=1.0) - put_mtm(sspot, K[0], tau, svol, r_d, r_f)

    if structure_id == "european_digital_rko":
        if is_call:
            return digital_rko_call_mtm(sspot, K[0], barrier, tau, svol, r_d, r_f, payout=entry_spot)
        return digital_rko_put_mtm(sspot, K[0], barrier, tau, svol, r_d, r_f, payout=entry_spot)

    return 0.0
