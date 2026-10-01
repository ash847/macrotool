"""Approved directional tail facts and construction-level eligibility."""

from knowledge_engine.loader import load_directional_tail_risk

TAIL_CONSTRAINTS = ("none", "lower_spot", "higher_spot", "both", "against_view", "with_view")


def resolved_tail_constraint(constraint, direction):
    if constraint not in TAIL_CONSTRAINTS:
        raise ValueError("Unknown directional tail constraint")
    if direction not in ("base_higher", "base_lower"):
        raise ValueError("Unknown view direction")
    if constraint == "against_view":
        return "lower_spot" if direction == "base_higher" else "higher_spot"
    if constraint == "with_view":
        return "higher_spot" if direction == "base_higher" else "lower_spot"
    return constraint


def tail_constraint_label(constraint):
    return load_directional_tail_risk()["preferences"][constraint]


def _declared_tails(family, is_call):
    values = load_directional_tail_risk()["families"].get(family, {}).get("call" if is_call else "put")
    if not isinstance(values, dict):
        return None, None
    lower, higher = values.get("lower_spot_tail"), values.get("higher_spot_tail")
    return (lower if type(lower) is bool else None, higher if type(higher) is bool else None)


def construction_tails(family, construction, is_call):
    from analytics.structure_pricer import _load_variants

    ignored = {"label", "can_lose_beyond_premium"}
    terms = {key: value for key, value in construction.items() if key not in ignored}
    matched = any({key: value for key, value in item.items() if key not in ignored} == terms
                  for item in _load_variants().get(family, []))
    return _declared_tails(family, is_call) if matched else (None, None)


def assign_variant_tails(family, variant, is_call, construction=None):
    from analytics.structure_pricer import _load_variants

    if family == "linear":
        lower, higher = _declared_tails(family, is_call)
    else:
        if construction is None:
            construction = next((item for item in _load_variants().get(family, [])
                                 if item["label"] == variant.variant_label), None)
        lower, higher = construction_tails(family, construction, is_call) if construction is not None else (None, None)
    if family == "seagull" and variant.wing_ratio is not None and variant.wing_ratio <= 0:
        lower, higher = (False, False) if lower is not None else (None, None)
    variant.lower_spot_tail, variant.higher_spot_tail = lower, higher


def tail_exclusion_reason(variant, constraint, direction):
    resolved = resolved_tail_constraint(constraint, direction)
    sides = ("lower_spot", "higher_spot") if resolved == "both" else (() if resolved == "none" else (resolved,))
    reasons = []
    for side in sides:
        value = getattr(variant, side + "_tail", None)
        if value is not False:
            reasons.append(f"{side.replace('_', '-')} tail {'present' if value is True else 'unknown (not treated as safe)'}")
    return "; ".join(reasons) or None


def tail_risk_text(variant):
    def label(value):
        return "Yes" if value is True else "No" if value is False else "Unknown"
    return (f"Lower-spot tail: {label(getattr(variant, 'lower_spot_tail', None))}; "
            f"Higher-spot tail: {label(getattr(variant, 'higher_spot_tail', None))}")
