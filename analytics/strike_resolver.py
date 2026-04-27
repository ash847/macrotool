"""
σ-approximation for delta-to-strike conversion.

Avoids full GK delta inversion; uses N^{-1}(Δ_call) as the z-score.
Accurate to < 1% vs exact inversion for typical FX tenors and vols.

Supported deltas: 0.10, 0.15, 0.25, 0.50. Linearly interpolated between.
"""

from __future__ import annotations

import math

# N^{-1}(Δ_call) for standard call deltas
_DELTA_Z: dict[float, float] = {
    0.50: 0.0000,
    0.25: 0.6745,
    0.15: 1.0364,
    0.10: 1.2816,
}


def _z_for_delta(delta: float) -> float:
    if delta in _DELTA_Z:
        return _DELTA_Z[delta]
    keys = sorted(_DELTA_Z.keys(), reverse=True)  # [0.50, 0.25, 0.15, 0.10]
    for i in range(len(keys) - 1):
        d_hi, d_lo = keys[i], keys[i + 1]
        if d_lo <= delta <= d_hi:
            t = (delta - d_lo) / (d_hi - d_lo)
            return _DELTA_Z[d_lo] * (1 - t) + _DELTA_Z[d_hi] * t
    raise ValueError(f"delta {delta} outside supported range [0.10, 0.50]")


def otm_call_strike(fwd: float, vol: float, T: float, delta: float) -> float:
    """Strike for an OTM call at the given delta (above the forward)."""
    return fwd * math.exp(_z_for_delta(delta) * vol * math.sqrt(T))


def otm_put_strike(fwd: float, vol: float, T: float, delta: float) -> float:
    """Strike for an OTM put at the given delta (below the forward)."""
    return fwd * math.exp(-_z_for_delta(delta) * vol * math.sqrt(T))
