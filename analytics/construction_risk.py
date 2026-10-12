"""Interpret declared construction risk without inferring payoff geometry."""

from collections.abc import Mapping


def declared_additional_loss(construction: Mapping) -> bool | None:
    value = construction.get("can_lose_beyond_premium")
    return value if type(value) is bool else None


def passes_no_tails(construction: Mapping) -> bool:
    return declared_additional_loss(construction) is False
