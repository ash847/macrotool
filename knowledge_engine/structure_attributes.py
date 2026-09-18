"""Qualitative, IP-clean characterization of a priced structure.

Derives a small set of attribute *tags* from numbers already computed (the P&L
driver split + the scenario aggregates + the structure's profile), then DISCARDS
the numbers. The LLM composes prose over these tags; it never sees a score, a
weight, a threshold, or a dimension name.

This is the whole authoring surface: ``ATTRIBUTES`` (the phrasebook) + the
thresholds below. Adding a structure costs nothing (it inherits the vocabulary);
adding a dimension is one tag + one condition. No pairwise templates exist — a
contrast between two structures is the *difference* of their tag sets, articulated
by the LLM, plus the single ``deciding_axis``.

Pure module: logic + glosses only, no IO, no rendering of app objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from knowledge_engine.comparator import ScenarioAggregates
from knowledge_engine.scenario_scorer import ScoreResult, driver_contribs

Polarity = Literal["edge", "caveat", "neutral"]


@dataclass(frozen=True)
class Attribute:
    gloss: str          # PM-facing, neutral — a *finding*, not finished prose
    polarity: Polarity


# --- the phrasebook (the only hand-authored surface) -----------------------
ATTRIBUTES: dict[str, Attribute] = {
    "carry_led":        Attribute("edge comes mainly from carry / roll-down, not the spot move", "edge"),
    "direction_led":    Attribute("edge depends on the spot move developing", "edge"),
    "vol_sensitive":    Attribute("P&L is materially sensitive to a vol shock", "neutral"),
    "holds_on_slow_path": Attribute("holds up if the move is slow or noisy", "edge"),
    "needs_the_move":   Attribute("needs the move to actually happen to pay", "caveat"),
    "resilient_wrong_way": Attribute("limited bleed if the market goes the wrong way", "edge"),
    "exposed_wrong_way": Attribute("loses meaningfully if the market goes against it", "caveat"),
    "rewards_overshoot": Attribute("keeps paying if the move overshoots the target", "edge"),
    "capped_upside":    Attribute("upside is capped beyond the target", "caveat"),
    "tail_beyond_short_strike": Attribute("turns increasingly negative on a large move through the short strike", "caveat"),
    "binary_expiry":    Attribute("outcome hinges on where spot finishes at expiry", "caveat"),
    "barrier_path_risk": Attribute("can be disrupted by the path, not just the final level", "caveat"),
}

# --- tuning surface (cutoffs; could move to JSON later, per the house pattern)
_LED_SHARE = 0.35       # a driver "leads" if it owns >= this share of the gross split
_VEGA_SHARE = 0.20      # vol matters if |Vega| owns >= this share
_OVERSHOOT_EDGE = 1.05  # overshoot "rewards" if it beats the correct-path avg by >= 5%
_WRONG_WAY_FLOOR = -0.01  # avg P&L (spot fraction) above which wrong-way bleed is "limited"

_CAPPED = {"1x1_spread", "1x1.5_spread", "1x2_spread", "seagull"}
_RATIO = {"1x1.5_spread", "1x2_spread"}
_BINARY = {"european_digital", "european_digital_rko"}
# Path-dependent barriers only. european_rko is EXPIRY-ONLY (the knock-out is tested at
# expiry, not on the path — see pricing/european_rko.py) so it must NOT carry
# barrier_path_risk; its profile flag is path_dependent=False. Only genuinely
# path-terminating barriers (rko, european_digital_rko — both enabled=false) belong here.
_BARRIER = {"rko", "european_digital_rko"}


def attributes(
    structure_id: str,
    pm_score: ScoreResult,
    aggregates: ScenarioAggregates,
) -> frozenset[str]:
    """Pure: (structure id + its scored cells + its scenario aggregates) -> tags.

    Called from both pack-build (recommended structures) and price_structure
    (a PM-named structure), against the same frozen inputs, so any structure —
    recommended or off-menu — lands in the identical vocabulary.
    """
    tags: set[str] = set()

    # --- edge source (off the driver split; the numbers never leave here) ---
    d = driver_contribs(pm_score)              # {Carry, Directional, Adverse, Vega}
    gross = sum(abs(v) for v in d.values()) or 1.0
    share = {k: v / gross for k, v in d.items()}
    if share.get("Carry", 0.0) >= _LED_SHARE and d.get("Carry", 0.0) > 0:
        tags.add("carry_led")
    if share.get("Directional", 0.0) >= _LED_SHARE and d.get("Directional", 0.0) > 0:
        tags.add("direction_led")
    if abs(share.get("Vega", 0.0)) >= _VEGA_SHARE:
        tags.add("vol_sensitive")

    # --- path behaviour (off the scenario aggregates) ---
    slow = aggregates.slow_path.avg_pnl_pct
    if slow is not None:
        tags.add("holds_on_slow_path" if slow >= 0.0 else "needs_the_move")
    ww = aggregates.wrong_way.avg_pnl_pct
    if ww is not None:
        tags.add("resilient_wrong_way" if ww >= _WRONG_WAY_FLOOR else "exposed_wrong_way")
    over = aggregates.overshoot.avg_pnl_pct
    corr = aggregates.correct_path.avg_pnl_pct
    rewards_overshoot = over is not None and corr is not None and over > corr * _OVERSHOOT_EDGE

    if rewards_overshoot:
        tags.add("rewards_overshoot")

    # --- structural geometry (off the profile/id — no scores at all) ---
    if structure_id in _CAPPED and not rewards_overshoot:
        tags.add("capped_upside")
    if structure_id in _RATIO:
        tags.add("tail_beyond_short_strike")
    if structure_id in _BINARY:
        tags.add("binary_expiry")
    if structure_id in _BARRIER:
        tags.add("barrier_path_risk")

    return frozenset(tags)


# --- the deciding axis (replaces the O(N^2) pairwise grid) ------------------
_AXIS_GLOSS: dict[str, str] = {
    "slow_path":    "it holds the slow / grinding scenarios better",
    "correct_path": "it pays more if the move develops as expected",
    "wrong_way":    "it bleeds less if the market goes against the view",
    "overshoot":    "it keeps paying if the move overshoots",
    "vol":          "it behaves better under a vol shock",
}


def deciding_axis(chosen: ScenarioAggregates, runner: ScenarioAggregates) -> str:
    """The single scenario bucket that most separates the top pick from the
    runner-up. Returns the axis *gloss* (IP-safe) — never the numbers."""
    diffs = {
        "slow_path":    _delta(chosen.slow_path, runner.slow_path),
        "correct_path": _delta(chosen.correct_path, runner.correct_path),
        "wrong_way":    _delta(chosen.wrong_way, runner.wrong_way),
        "overshoot":    _delta(chosen.overshoot, runner.overshoot),
        "vol":          _delta(chosen.vol_sensitivity, runner.vol_sensitivity),
    }
    axis = max(diffs, key=lambda k: diffs[k])
    return _AXIS_GLOSS[axis]


def _delta(a, b) -> float:
    return (a.avg_pnl_pct or 0.0) - (b.avg_pnl_pct or 0.0)


# --- IP-clean text rendering (shared by the agent pack + the admin preview) -
def render_findings(
    rows: list[tuple[str, frozenset[str]]],
    *,
    deciding: str | None = None,
    header: str = "FINDINGS (qualitative — synthesize, do not list; state no score)",
) -> str:
    """Render labelled (structure -> tag set) rows as an IP-clean findings block.

    ``rows`` is an ordered list of ``(label, tags)``. Emits edges / caveats per
    label using the phrasebook glosses; ``deciding`` is the axis gloss line.
    Contains no scores, weights, thresholds, or dimension names by construction.
    """
    if not rows:
        return ""
    lines = [header + ":"]
    for label, tags in rows:
        lines.append(f"  {label}:")
        edges = [ATTRIBUTES[t].gloss for t in _ordered(tags) if ATTRIBUTES[t].polarity == "edge"]
        caveats = [ATTRIBUTES[t].gloss for t in _ordered(tags)
                   if ATTRIBUTES[t].polarity in ("caveat", "neutral")]
        if edges:
            lines.append("    edges:   " + "; ".join(edges))
        if caveats:
            lines.append("    caveats: " + "; ".join(caveats))
        if not edges and not caveats:
            lines.append("    (no distinguishing findings)")
    if deciding:
        lines.append(f"  decided by: {deciding}")
    return "\n".join(lines)


def _ordered(tags: frozenset[str]) -> list[str]:
    """Stable, vocabulary order so the rendered text is deterministic."""
    return [t for t in ATTRIBUTES if t in tags]
