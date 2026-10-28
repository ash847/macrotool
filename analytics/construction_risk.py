"""Construction risk: derived from the legs, cross-checked against the catalog."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


def declared_additional_loss(construction: Mapping) -> bool | None:
    value = construction.get("can_lose_beyond_premium")
    return value if type(value) is bool else None


def passes_no_tails(construction: Mapping) -> bool:
    return declared_additional_loss(construction) is False


@dataclass(frozen=True)
class LegRisk:
    lower_spot_tail: bool
    higher_spot_tail: bool
    can_lose_beyond_premium: bool


def leg_risk(terms) -> LegRisk | None:
    """Expiry loss geometry of an option package from its legs alone — the same for
    strike- and delta-specified legs. ``terms`` are ``(signed notional, strike, is_call)``.

    A side has a tail when the package is net short notional beyond its outermost strike
    on that side: calls set the higher-spot side, puts the lower-spot side. It can lose
    beyond the premium when it has a tail, or when its expiry payoff dips below zero
    between strikes (a spread entered the wrong way round).
    """
    if not terms or any(not math.isfinite(w) or not math.isfinite(k) or k <= 0 for w, k, _ in terms):
        return None
    eps = 1e-12
    # bool(): strikes may be numpy floats, and callers test identity (`is True`).
    higher = bool(sum(w for w, _, call in terms if call) < -eps)
    lower = bool(sum(w for w, _, call in terms if not call) < -eps)

    def intrinsic(spot):
        return sum(w * max(spot - k if call else k - spot, 0.0) for w, k, call in terms)

    scale = max(k for _, k, _ in terms)
    dips = bool(min(intrinsic(k) for _, k, _ in terms) < -eps * scale)
    return LegRisk(lower, higher, lower or higher or dips)


def reconcile(declared: bool | None, derived: bool | None) -> bool | None:
    """Leg geometry is the answer; a catalog declaration, where one matches, must agree.
    A disagreement is unknown — never a silent pick of either."""
    if derived is None:
        return declared
    if declared is None or declared is derived:
        return derived
    return None
