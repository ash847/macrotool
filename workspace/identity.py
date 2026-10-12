"""The one rule for how a new evaluation relates to the ideas a conversation holds.

An idea is identified by **pair + direction**. Within that, a change of target,
expiry, mode or PM preferences is a new *version* of the same idea. Phase 1 uses this
only to decide what to record (the active idea just follows the latest pack); option C
reuses it unchanged to decide whether a pack belongs to an existing idea.
"""

from __future__ import annotations

from typing import Iterable, Literal

from workspace.models import IdeaVersion, TradeIdea, ViewSpec

Change = Literal["same", "revision", "new_idea"]


def idea_key(pair: str, direction: str) -> tuple[str, str]:
    return (pair, direction)


def classify_view_change(
    latest: IdeaVersion | None, spec: ViewSpec, prefs: dict
) -> Change:
    """Compare a fresh evaluation with an idea's latest version."""
    if latest is None or idea_key(latest.spec.pair, latest.spec.direction) != idea_key(
        spec.pair, spec.direction
    ):
        return "new_idea"
    if latest.spec == spec and latest.prefs == prefs:
        return "same"
    return "revision"


def find_idea(ideas: Iterable[TradeIdea], spec: ViewSpec) -> TradeIdea | None:
    """The idea among ``ideas`` with the spec's pair + direction, if any."""
    key = idea_key(spec.pair, spec.direction)
    return next((i for i in ideas if idea_key(i.pair, i.direction) == key), None)
