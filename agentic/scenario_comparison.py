"""Retained scenario evidence and public attribution for custom comparisons.

Raw scores and weights stay server-side. Public output contains engine-computed
ordering and signed shares, using every cell in the absolute-share denominator.
"""

from dataclasses import dataclass
import math

from knowledge_engine.scenario_scorer import ScoreResult, cell_label, contribution_share
from knowledge_engine.loader import load_contribution_display


@dataclass(frozen=True)
class ScenarioEvidence:
    rows: tuple[dict, ...]
    score: ScoreResult


def validate_evidence(evidence):
    if evidence is None or not evidence.rows or not evidence.score.cells:
        raise ValueError("Scenario evidence is unavailable; a usable target and ranked pack are required.")
    cells = evidence.score.cells
    ids = [cell.scenario_id for cell in cells]
    if (len(set(ids)) != len(ids) or len(evidence.rows) != len(ids)
            or set(ids) != {row["scenario_id"] for row in evidence.rows}):
        raise ValueError("Scenario evidence has incomplete or duplicate cells.")
    if any(row.get("pricing_error") for row in evidence.rows):
        raise ValueError("Scenario pricing failed for one or more cells; comparison is unavailable.")
    values = [evidence.score.score_pct]
    if evidence.score.score_ccy is not None:
        values.append(evidence.score.score_ccy)
    for cell in cells:
        values.extend((cell.pnl_pct, cell.contrib_pct, cell.normalized_weight))
        if cell.pnl_ccy is not None:
            values.append(cell.pnl_ccy)
        if cell.contrib_ccy is not None:
            values.append(cell.contrib_ccy)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Scenario evidence contains non-finite values.")
    if any(cell.normalized_weight < 0 for cell in cells):
        raise ValueError("Scenario weights must be non-negative.")
    if not math.isclose(sum(cell.normalized_weight for cell in cells), 1.0, abs_tol=1e-10):
        raise ValueError("Scenario weights are unavailable or invalid.")


def evaluate_custom(result, pack):
    from analytics.scenario_pricer import price_scenarios
    from knowledge_engine.scenario_scorer import score_structure

    if not pack.variants_ranked or not pack.scenario_grid or not pack.scenario_weights:
        raise ValueError("Scenario scoring unavailable; establish a ranked view with a usable target first.")
    ms = pack.market_state
    inputs = dict(spot=ms.spot, forward=ms.fwd, implied_vol=ms.vol, tenor_years=ms.T,
                  target=pack.target, r_d=ms.r_d, r_f=ms.r_f)
    rows = price_scenarios(result.variant, result.request.family, pack.scenario_grid,
                           inputs, pack.is_call, surface=ms.surface)
    evidence = ScenarioEvidence(tuple(rows), score_structure(rows, pack.scenario_weights))
    validate_evidence(evidence)
    return evidence


def _ordering(value):
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "equal"
    return "higher" if value > 0 else "lower"


def _drivers(cells, values):
    minimum = load_contribution_display()["minimum_absolute_total_pct"]
    denominator = math.fsum(abs(value) for value in values)
    items = [(cell_label(cell), value) for cell, value in zip(cells, values)]
    def render(rows):
        return [{"scenario": label, "signed_share_of_total_absolute_contribution":
                 contribution_share(value, denominator, minimum)} for label, value in rows]
    return {
        "contributors": render(sorted((item for item in items if item[1] > 0), key=lambda item: -item[1])[:3]),
        "detractors": render(sorted((item for item in items if item[1] < 0), key=lambda item: item[1])[:3]),
    }


def public_attribution(evidence):
    validate_evidence(evidence)
    return _drivers(evidence.score.cells, [cell.contrib_pct for cell in evidence.score.cells])


def compare_evidence(custom, reference, custom_notional, reference_notional):
    validate_evidence(custom)
    validate_evidence(reference)
    cells = custom.score.cells
    other = {cell.scenario_id: cell for cell in reference.score.cells}
    if set(other) != {cell.scenario_id for cell in cells} or any(
        cell.normalized_weight != other[cell.scenario_id].normalized_weight for cell in cells
    ):
        raise ValueError("Trades do not share the same scenario cells and weights.")
    gaps = [cell.contrib_pct - other[cell.scenario_id].contrib_pct for cell in cells]
    result = {
        "per_unit_performance": _ordering(custom.score.score_pct - reference.score.score_pct),
        "per_unit_gap_attribution": _drivers(cells, gaps),
        "actual_sized_performance": "unavailable",
    }
    if custom_notional is None or reference_notional is None:
        return result
    if not all(math.isfinite(n) and n >= 0 for n in (custom_notional, reference_notional)):
        raise ValueError("Sized comparison requires finite non-negative notionals.")
    # Exact bridge at the comparator's notional, not a causal claim:
    # Nc*Pc - Nr*Pr = Nr*(Pc-Pr) + (Nc-Nr)*Pc.
    structure_effect = reference_notional * (custom.score.score_pct - reference.score.score_pct)
    size_effect = (custom_notional - reference_notional) * custom.score.score_pct
    result.update(
        actual_sized_performance=_ordering(structure_effect + size_effect),
        structure_effect_at_comparator_notional=_ordering(structure_effect),
        size_effect_using_custom_per_unit_performance=_ordering(size_effect),
    )
    # Divide all sized cell differences by a common positive notional for a
    # stable dimensionless near-zero threshold, without changing share ratios.
    scale = max(custom_notional, reference_notional)
    sized_gaps = [(cell.contrib_pct * custom_notional - other[cell.scenario_id].contrib_pct * reference_notional)
                  / scale if scale else 0.0 for cell in cells]
    result["actual_sized_gap_attribution"] = _drivers(cells, sized_gaps)
    return result
