"""Expiry loss extrema on the engine's base-currency P&L basis."""

import math


def maximum_expiry_loss(terms, premium):
    if not terms or not math.isfinite(premium):
        return None
    if any(not math.isfinite(weight) or not math.isfinite(strike) or strike <= 0
           for weight, strike, _ in terms):
        return None
    knots = sorted({strike for _, strike, _ in terms})

    def intrinsic(spot):
        return sum(weight * max(spot - strike if call else strike - spot, 0.0)
                   for weight, strike, call in terms)

    low_intercept = intrinsic(0.0)
    low_slope = sum(-weight for weight, _, call in terms if not call)
    high_slope = sum(weight for weight, _, call in terms if call)
    low_limit = (low_slope - premium if abs(low_intercept) < 1e-12
                 else math.copysign(math.inf, low_intercept))
    minimum = min([intrinsic(strike) / strike - premium for strike in knots]
                  + [low_limit, high_slope - premium])
    return max(-minimum, 0.0)
