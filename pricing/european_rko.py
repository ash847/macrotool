"""
European reverse knock-out (ERKO) pricer.

Unlike the path-dependent RKO, the knock-out condition is checked only at
expiry. The option pays like a vanilla only if terminal spot finishes between
the strike and the barrier in the profit direction:

  - Call ERKO: payoff = max(S_T - K, 0) * 1_{S_T < H}, with H > K
  - Put  ERKO: payoff = max(K - S_T, 0) * 1_{S_T > H}, with H < K

This is a terminal-state truncation, not a barrier-touch product.
"""

from __future__ import annotations

from pricing.black_scholes import call_value, put_value
from pricing.digital import digital_call, digital_put


def european_rko_call(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """Expiry-only reverse knock-out call."""
    if barrier <= strike:
        return 0.0
    vanilla_k = call_value(spot, strike, T, sigma, r_d, r_f, notional)
    vanilla_h = call_value(spot, barrier, T, sigma, r_d, r_f, notional)
    strip = (barrier - strike) * digital_call(spot, barrier, T, sigma, r_d, r_f, payout=1.0, notional=notional)
    return max(vanilla_k - vanilla_h - strip, 0.0)


def european_rko_put(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """Expiry-only reverse knock-out put."""
    if barrier >= strike:
        return 0.0
    vanilla_k = put_value(spot, strike, T, sigma, r_d, r_f, notional)
    vanilla_h = put_value(spot, barrier, T, sigma, r_d, r_f, notional)
    strip = (strike - barrier) * digital_put(spot, barrier, T, sigma, r_d, r_f, payout=1.0, notional=notional)
    return max(vanilla_k - vanilla_h - strip, 0.0)


def european_rko_call_mtm(
    spot: float,
    strike: float,
    barrier: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """MtM of a long expiry-only reverse knock-out call."""
    if T_remaining <= 0:
        if spot >= barrier:
            return 0.0
        return max(spot - strike, 0.0) * notional
    return european_rko_call(spot, strike, barrier, T_remaining, sigma, r_d, r_f, notional)


def european_rko_put_mtm(
    spot: float,
    strike: float,
    barrier: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """MtM of a long expiry-only reverse knock-out put."""
    if T_remaining <= 0:
        if spot <= barrier:
            return 0.0
        return max(strike - spot, 0.0) * notional
    return european_rko_put(spot, strike, barrier, T_remaining, sigma, r_d, r_f, notional)
