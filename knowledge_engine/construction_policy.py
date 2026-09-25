"""Construction-level eligibility using the same catalog as entry pricing."""

from analytics.construction_risk import declared_additional_loss, passes_no_tails
from analytics.structure_pricer import _load_variants


def family_has_no_tail_construction(structure_id: str) -> bool:
    return any(passes_no_tails(item) for item in _load_variants().get(structure_id, []))


def configured_additional_loss(structure_id: str, construction: dict) -> bool | None:
    """Match custom terms, not display labels; unclassified terms remain unknown."""
    ignored = {"label", "can_lose_beyond_premium"}
    terms = {key: value for key, value in construction.items() if key not in ignored}
    matches = [
        declared_additional_loss(item)
        for item in _load_variants().get(structure_id, [])
        if {key: value for key, value in item.items() if key not in ignored} == terms
    ]
    if not matches or any(value is None for value in matches):
        return None
    return matches[0] if all(value is matches[0] for value in matches) else None
