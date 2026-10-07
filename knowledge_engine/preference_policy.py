"""Hard PM eligibility checks, separate from sizing proxies and directional tails."""

import math


def preference_exclusion_reason(family, variant, constraint):
    if constraint == "Avoid capped structures" and family != "vanilla":
        return "Vanilla-only preference excludes this family"
    if constraint == "Avoid tail-risky structures":
        economics = getattr(variant, "economics", None)
        bound = getattr(economics, "contractual_max_loss_pct", None)
        if (getattr(economics, "contractual_loss_status", None) != "bounded"
                or bound is None or not math.isfinite(bound) or bound < 0):
            return "Defined-loss preference requires a verified finite maximum loss; unknown or unbounded loss is excluded"
    return None
