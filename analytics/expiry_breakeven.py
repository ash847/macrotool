"""Exact expiry zero-P&L crossings in base currency, without premium accrual."""

import math


def variant_terms(family, strikes, is_call, wing_ratio=None):
    weights = {
        "vanilla": (1,), "1x1_spread": (1, -1),
        "1x1.5_spread": (1, -1.5), "1x2_spread": (1, -2),
        "1x2x1_spread": (1, -2, 1), "european_rko": (1,),
        "european_digital": (1,),
    }.get(family)
    if family == "seagull":
        if len(strikes) != 3 or wing_ratio is None:
            return None
        return [(1, strikes[0], is_call), (-1, strikes[1], is_call),
                (-wing_ratio, strikes[2], not is_call)]
    if weights is None or len(strikes) < len(weights):
        return None
    return [(weight, strike, is_call) for weight, strike in zip(weights, strikes)]


def expiry_breakevens(terms, premium, *, barrier=None, is_call=True, digital=False):
    if (not terms or not math.isfinite(premium)
            or any(not math.isfinite(weight) or not math.isfinite(strike) or strike <= 0
                   for weight, strike, _ in terms)
            or (barrier is not None and (not math.isfinite(barrier) or barrier <= 0))):
        return None, False
    knots = sorted({strike for _, strike, _ in terms} | ({barrier} if barrier is not None else set()))

    def coefficients(spot):
        if barrier is not None and (spot >= barrier if is_call else spot <= barrier):
            return 0.0, 0.0
        if digital:
            active = spot > terms[0][1] if is_call else spot < terms[0][1]
            return (1.0 if active else 0.0), 0.0
        active = [(weight, strike, call) for weight, strike, call in terms
                  if (spot > strike if call else spot < strike)]
        slope = math.fsum(weight if call else -weight for weight, _, call in active)
        intercept = math.fsum(-weight * strike if call else weight * strike for weight, strike, call in active)
        return slope, intercept

    boundaries = [0.0, *knots, math.inf]
    roots, flat_regions = [], []
    for lower, upper in zip(boundaries, boundaries[1:]):
        probe = lower * 2 if math.isinf(upper) else (lower + upper) / 2
        slope, intercept = coefficients(probe)
        denominator = slope - premium
        if abs(denominator) < 1e-14:
            if abs(intercept) < 1e-14:
                flat_regions.append((lower, upper))
            continue
        root = -intercept / denominator
        if root <= 0 or not math.isfinite(root) or root < lower or root > upper:
            continue
        actual_slope, actual_intercept = coefficients(root)
        if abs(actual_slope + actual_intercept / root - premium) < 1e-10:
            roots.append(root)
    isolated = []
    for root in sorted(roots):
        if any(lower <= root <= upper for lower, upper in flat_regions):
            continue
        if not isolated or not math.isclose(root, isolated[-1], rel_tol=1e-10):
            isolated.append(root)
    return isolated, bool(flat_regions)


def breakeven_text(value):
    roots = getattr(value, "breakevens", None)
    if roots is None:
        return "Unavailable"
    text = "; ".join(f"{root:.4f}" for root in roots)
    if getattr(value, "breakeven_has_zero_region", False):
        return (text + "; " if text else "") + "zero-P&L region(s), not a single breakeven"
    return text or "None (no zero-P&L crossing)"
