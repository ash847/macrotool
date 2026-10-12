"""Concise expiry risk facts on the scenario engine's base-currency P&L basis."""

import math

from analytics.expiry_loss import maximum_expiry_loss


def payoff_risk_note(structure_id, variant, priced_structure, is_call):
    side = "above" if is_call else "below"
    opposite = "below" if is_call else "above"
    strikes = variant.strikes
    premium = variant.net_premium_pct
    if structure_id == "linear":
        return f"Loses as spot moves {'lower' if is_call else 'higher'}. The sizing reference is not a guaranteed stop."
    if not strikes or premium is None or not math.isfinite(premium):
        return "Payoff / risk unavailable — incomplete priced terms."
    strike = strikes[0]
    barrier = variant.barrier
    if structure_id == "european_digital":
        return f"At expiry, pays 100% of base-currency notional if spot finishes {side} {strike:.4f}; otherwise zero, before entry premium."
    if structure_id in {"rko", "european_rko", "european_digital_rko"}:
        if barrier is None:
            return "Payoff / risk unavailable — barrier not retained."
        if structure_id == "european_rko":
            return f"Pays zero if expiry spot is at or {side} {barrier:.4f}, or at or {opposite} {strike:.4f}. An earlier barrier touch does not knock it out."
        if structure_id == "european_digital_rko":
            return f"Pays only if expiry spot finishes {side} {strike:.4f} and spot has never touched the {barrier:.4f} knock-out barrier. Otherwise the payout is zero."
        return f"Knocks out if spot touches {barrier:.4f} during its life, even if spot subsequently reverses. Otherwise pays the {side}-strike option payoff at expiry (strike {strike:.4f})."
    legs = getattr(priced_structure, "priced_legs", None)
    if not legs:
        return "Payoff / risk unavailable — actual option legs not retained."
    terms = [(leg.notional, leg.strike, leg.leg.right.value == "call") for leg in legs]
    if any(not math.isfinite(weight) or not math.isfinite(level) or level <= 0 for weight, level, _ in terms):
        return "Payoff / risk unavailable — invalid option legs."
    knots = sorted({level for _, level, _ in terms})

    def intrinsic(spot):
        return sum(weight * max(spot - level if call else level - spot, 0.0)
                   for weight, level, call in terms)

    tails = []
    for upper in (False, True):
        edge = knots[-1] if upper else knots[0]
        slope = sum(weight if call else -weight for weight, _, call in terms if call == upper)
        if (upper and slope >= -1e-12) or (not upper and slope <= 1e-12):
            continue
        intercept = intrinsic(edge) - slope * edge
        direction = "above" if upper else "below"
        exposure = "short" if upper else "long"
        denominator = slope - premium
        crossing = -intercept / denominator if abs(denominator) > 1e-12 else None
        valid = crossing is not None and crossing > 0 and (crossing >= edge if upper else crossing <= edge)
        text = f"net {exposure} {direction} {edge:.4f}"
        if valid:
            text += f"; net losses {direction} {crossing:.4f} (net of entry premium)"
        else:
            probe = edge * 2 if upper else edge / 2
            if intrinsic(probe) / probe - premium < 0:
                text += "; net losses throughout that tail region"
            else:
                text += "; no net-loss crossing in that tail region"
        tails.append(text)
    if tails:
        note = "At expiry, " + "; ".join(tails) + "."
        if structure_id == "seagull" and len(strikes) >= 2:
            note += f" Directional option payoff capped beyond {strikes[1]:.4f}."
        return note
    if structure_id == "vanilla":
        return f"Can lose the {max(premium, 0):.2%} premium paid (of base-currency notional). Time decay reduces value if the move is slow."
    if structure_id == "1x1_spread":
        bound = maximum_expiry_loss(terms, premium)
        loss = f"{bound:.2%} of base-currency notional" if math.isfinite(bound) else "unbounded on the engine's base-currency basis"
        return f"Expiry option payoff capped {side} {strikes[1]:.4f}. Maximum net loss: {loss}."
    if structure_id == "1x2x1_spread":
        peak = max(knots, key=intrinsic)
        low_slope = sum(-weight for weight, _, call in terms if not call)
        high_slope = sum(weight for weight, _, call in terms if call)
        zero_outside = abs(low_slope) < 1e-12 and abs(high_slope) < 1e-12 and abs(intrinsic(knots[0])) < 1e-9 and abs(intrinsic(knots[-1])) < 1e-9
        note = f"Expiry option payoff peaks at {peak:.4f}."
        if zero_outside:
            note += f" Outside {knots[0]:.4f}–{knots[-1]:.4f}, payoff is zero; net P&L is {-premium:+.2%} of base-currency notional."
        else:
            note += " Unequal wings can leave a non-zero outer payoff."
        bound = maximum_expiry_loss(terms, premium)
        loss = f"{bound:.2%} of base-currency notional" if math.isfinite(bound) else "unbounded on the engine's base-currency basis"
        note += f" Maximum net loss: {loss}."
        return note
    return "Expiry payoff follows the retained option legs; no adverse outer-tail slope. Premium remains part of net P&L."
