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

from analytics.structure_pricer import PricedVariant
from pricing.black_scholes import call_mtm, put_mtm
from pricing.digital import digital_call_mtm, digital_put_mtm
from pricing.digital_rko import digital_rko_call_mtm, digital_rko_put_mtm


def price_scenarios(
    variant: PricedVariant,
    structure_id: str,
    scenarios: list[dict],
    trade_inputs: dict,
    is_call: bool,
) -> list[dict]:
    """
    Price `variant` in every scenario. Returns one row per scenario with
    price_pct (option value as fraction of entry spot) and
    pnl_pct (price_pct minus entry premium paid).

    trade_inputs must contain: spot, r_d, r_f.
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
        scenario_vol: float = d["scenario_vol"]
        tau: float = d["remaining_time"]

        try:
            raw = _value_variant(
                structure_id, variant,
                scenario_spot, scenario_vol, tau,
                r_d, r_f, entry_spot, is_call,
            )
        except Exception:
            raw = 0.0

        price_pct = raw / entry_spot
        pnl_pct = price_pct - entry_premium_pct
        price_ccy = (price_pct * notional) if notional is not None else None
        pnl_ccy = (pnl_pct * notional) if notional is not None else None

        results.append({
            "structure_id": structure_id,
            "variant_label": variant.variant_label,
            "scenario_id": sc["id"],
            "family": sc["family"],
            "time_fraction": sc["time_fraction"],
            "fwd_rule": sc["fwd_rule"],
            "vol_rule": sc["vol_rule"],
            "skew_rule": sc["skew_rule"],
            "tags": sc["tags"],
            "elapsed_time": d["elapsed_time"],
            "remaining_time": tau,
            "scenario_fwd": d["scenario_fwd"],
            "scenario_spot": scenario_spot,
            "vol_shift": d["vol_shift"],
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
) -> float:
    """Return absolute MtM value (same currency units as entry_spot)."""
    K = variant.strikes
    barrier = variant.barrier

    if structure_id == "vanilla":
        if is_call:
            return call_mtm(sspot, K[0], tau, svol, r_d, r_f)
        return put_mtm(sspot, K[0], tau, svol, r_d, r_f)

    if structure_id == "1x1_spread":
        if is_call:
            return call_mtm(sspot, K[0], tau, svol, r_d, r_f) - call_mtm(sspot, K[1], tau, svol, r_d, r_f)
        return put_mtm(sspot, K[0], tau, svol, r_d, r_f) - put_mtm(sspot, K[1], tau, svol, r_d, r_f)

    if structure_id == "1x2_spread":
        if is_call:
            return call_mtm(sspot, K[0], tau, svol, r_d, r_f) - 2.0 * call_mtm(sspot, K[1], tau, svol, r_d, r_f)
        return put_mtm(sspot, K[0], tau, svol, r_d, r_f) - 2.0 * put_mtm(sspot, K[1], tau, svol, r_d, r_f)

    if structure_id == "seagull":
        wing_ratio = variant.wing_ratio or 0.0
        if is_call:
            # long call spread + short put wing
            spread = call_mtm(sspot, K[0], tau, svol, r_d, r_f) - call_mtm(sspot, K[1], tau, svol, r_d, r_f)
            wing = put_mtm(sspot, K[2], tau, svol, r_d, r_f)
        else:
            # long put spread + short call wing
            spread = put_mtm(sspot, K[0], tau, svol, r_d, r_f) - put_mtm(sspot, K[1], tau, svol, r_d, r_f)
            wing = call_mtm(sspot, K[2], tau, svol, r_d, r_f)
        return spread - wing_ratio * wing

    if structure_id == "european_digital":
        if is_call:
            return digital_call_mtm(sspot, K[0], tau, svol, r_d, r_f, payout=entry_spot)
        return digital_put_mtm(sspot, K[0], tau, svol, r_d, r_f, payout=entry_spot)

    if structure_id == "european_digital_rko":
        if is_call:
            return digital_rko_call_mtm(sspot, K[0], barrier, tau, svol, r_d, r_f, payout=entry_spot)
        return digital_rko_put_mtm(sspot, K[0], barrier, tau, svol, r_d, r_f, payout=entry_spot)

    return 0.0
