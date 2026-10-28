"""
Exact unadjusted forward-delta inversion, matching the smile's N(d1) convention.
ATM-forward is a separate anchor (K=F), not a synonym for 50-delta.
"""

from __future__ import annotations

import math
from statistics import NormalDist

# Public request grammar range; the exact mathematical resolver accepts (0, 1).
# ATMF remains K=F, distinct from a 50-delta strike.
MIN_DELTA, MAX_DELTA = 0.10, 0.50

def _strike(fwd: float, vol: float, T: float, delta: float, is_call: bool) -> float:
    if not all(math.isfinite(x) for x in (fwd, vol, T, delta)):
        raise ValueError("Strike inputs must be finite")
    if fwd <= 0 or vol <= 0 or T <= 0 or not 0 < delta < 1:
        raise ValueError("Forward, vol and tenor must be positive; delta must be in (0, 1)")
    z = NormalDist().inv_cdf(delta)
    sign = -1.0 if is_call else 1.0
    return fwd * math.exp(0.5 * vol * vol * T + sign * z * vol * math.sqrt(T))


def otm_call_strike(fwd: float, vol: float, T: float, delta: float) -> float:
    """Strike at the given call forward delta; historical function name retained."""
    return _strike(fwd, vol, T, delta, True)


def otm_put_strike(fwd: float, vol: float, T: float, delta: float) -> float:
    """Strike at the given absolute put forward delta (not necessarily below F)."""
    return _strike(fwd, vol, T, delta, False)
