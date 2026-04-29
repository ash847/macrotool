"""
Scenario scorer — combines family weights (from scenario_weighter) with
per-scenario P&L rows (from scenario_pricer) into a single weighted-P&L
score per structure.

Pure function. No MarketState input, no shortlist input — just the rows
priced for one structure and a weight vector.

Score formula:
    family_pnl[f] = mean(pnl_pct over scenarios in family f)
    score_pct     = Σ (weight[f] × family_pnl[f])
    score_ccy     = score_pct × structure_notional   (if notional present)

If a family has rows priced but no weight (or vice versa), the weight
defaults to 0 contribution. Families with priced rows but no weight in
the dict are still surfaced in the breakdown so the PM can see what
was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyBreakdown:
    family: str
    n_scenarios: int
    avg_pnl_pct: float
    avg_pnl_ccy: float | None
    weight: float
    contrib_pct: float           # weight × avg_pnl_pct
    contrib_ccy: float | None    # weight × avg_pnl_ccy   (if ccy values present)


@dataclass(frozen=True)
class ScoreResult:
    score_pct: float
    score_ccy: float | None
    families: list[FamilyBreakdown]


def score_structure(
    scenario_rows: list[dict],
    weights: dict[str, float],
) -> ScoreResult:
    """
    Compute weighted P&L score for one structure.

    Args:
        scenario_rows: output of `analytics.scenario_pricer.price_scenarios`,
                       one row per scenario for a single structure variant.
                       Required keys per row: 'family', 'pnl_pct'. Optional:
                       'pnl_ccy' (used to compute score_ccy when present).
        weights:       {family: weight} from `compute_family_weights`. Should
                       sum to ~1.0 but the function does not enforce this —
                       any normalisation is the caller's responsibility.

    Returns:
        ScoreResult with the aggregate score (in % of spot, and base ccy if
        available) plus a per-family breakdown for UI display.
    """
    if not scenario_rows:
        return ScoreResult(score_pct=0.0, score_ccy=None, families=[])

    by_family: dict[str, list[dict]] = {}
    for row in scenario_rows:
        by_family.setdefault(row["family"], []).append(row)

    breakdowns: list[FamilyBreakdown] = []
    score_pct = 0.0
    score_ccy_acc: float = 0.0
    any_ccy = False

    for family, rows in by_family.items():
        n = len(rows)
        avg_pct = sum(r["pnl_pct"] for r in rows) / n

        ccy_values = [r.get("pnl_ccy") for r in rows if r.get("pnl_ccy") is not None]
        if ccy_values:
            avg_ccy: float | None = sum(ccy_values) / len(ccy_values)
            any_ccy = True
        else:
            avg_ccy = None

        w = weights.get(family, 0.0)
        contrib_pct = w * avg_pct
        contrib_ccy = (w * avg_ccy) if avg_ccy is not None else None

        score_pct += contrib_pct
        if contrib_ccy is not None:
            score_ccy_acc += contrib_ccy

        breakdowns.append(FamilyBreakdown(
            family=family,
            n_scenarios=n,
            avg_pnl_pct=avg_pct,
            avg_pnl_ccy=avg_ccy,
            weight=w,
            contrib_pct=contrib_pct,
            contrib_ccy=contrib_ccy,
        ))

    return ScoreResult(
        score_pct=score_pct,
        score_ccy=(score_ccy_acc if any_ccy else None),
        families=breakdowns,
    )
