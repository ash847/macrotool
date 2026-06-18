"""
Scenario-grid scorer.

Combines explicit scenario-cell multipliers with per-scenario P&L rows into a
single weighted-average score for one structure variant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellBreakdown:
    scenario_id: str
    row: str
    col: str
    pnl_pct: float
    pnl_ccy: float | None
    multiplier: float
    normalized_weight: float
    contrib_pct: float
    contrib_ccy: float | None


@dataclass(frozen=True)
class ScoreResult:
    score_pct: float
    score_ccy: float | None
    cells: list[CellBreakdown]


# P&L-driver buckets over the scenario-grid columns. Every GRID_COL maps to exactly
# one bucket, so a variant's driver contributions sum back to its weighted P&L (see
# driver_contribs). "Carry" = the no-move (decay / roll-down) cell; "Adverse" = the
# spot-anchored downside cells; "Vega" = the vol-shock column. Lives in the engine so
# both the UI (structure_eval) and the agent pack (render) can share it.
DRIVER_BUCKETS: dict[str, list[str]] = {
    "Carry":       ["S"],
    "Directional": ["t%→K", "K−½σ", "K", "K+½σ"],
    "Adverse":     ["−½σ", "−1σ"],
    "Vega":        ["Δvol"],
}


def driver_contribs(score: "ScoreResult") -> dict[str, float]:
    """Decompose a ScoreResult's weighted P&L into driver buckets — the sum of
    per-cell ``contrib_pct`` grouped by grid column. Exhaustive: the bucket totals
    sum back to ``score.score_pct``."""
    by_col: dict[str, float] = {}
    for c in score.cells:
        by_col[c.col] = by_col.get(c.col, 0.0) + c.contrib_pct
    return {
        bucket: sum(by_col.get(col, 0.0) for col in cols)
        for bucket, cols in DRIVER_BUCKETS.items()
    }


def score_structure(
    scenario_rows: list[dict],
    multipliers: dict[str, float],
) -> ScoreResult:
    if not scenario_rows:
        return ScoreResult(score_pct=0.0, score_ccy=None, cells=[])

    weighted_rows: list[tuple[dict, float]] = []
    for row in scenario_rows:
        m = multipliers.get(row["scenario_id"], 0.0)
        weighted_rows.append((row, m))

    total_weight = sum(m for _, m in weighted_rows if m > 0)
    if total_weight <= 0:
        return ScoreResult(score_pct=0.0, score_ccy=None, cells=[])

    score_pct = 0.0
    score_ccy_acc = 0.0
    any_ccy = False
    cells: list[CellBreakdown] = []

    for row, multiplier in weighted_rows:
        norm_w = (multiplier / total_weight) if multiplier > 0 else 0.0
        contrib_pct = norm_w * row["pnl_pct"]
        pnl_ccy = row.get("pnl_ccy")
        contrib_ccy = (norm_w * pnl_ccy) if pnl_ccy is not None else None

        score_pct += contrib_pct
        if contrib_ccy is not None:
            any_ccy = True
            score_ccy_acc += contrib_ccy

        cells.append(CellBreakdown(
            scenario_id=row["scenario_id"],
            row=row["row"],
            col=row["col"],
            pnl_pct=row["pnl_pct"],
            pnl_ccy=pnl_ccy,
            multiplier=multiplier,
            normalized_weight=norm_w,
            contrib_pct=contrib_pct,
            contrib_ccy=contrib_ccy,
        ))

    return ScoreResult(
        score_pct=score_pct,
        score_ccy=(score_ccy_acc if any_ccy else None),
        cells=cells,
    )
