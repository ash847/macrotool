"""max_possible_score() — the affinity ceiling used to express a structure's total
as a % of the maximum possible on the tester Shortlisted-structures block."""

from __future__ import annotations

from knowledge_engine.loader import load_affinity_scores
from knowledge_engine.structure_scorer import (
    _SCORED_DIMS,
    max_possible_score,
)


def test_ceiling_equals_sum_of_per_dimension_maxima():
    cfg = load_affinity_scores()["structures"]
    expected = 0.0
    for dim in _SCORED_DIMS:
        dim_max = 0.0
        for sc in cfg.values():
            for v in (sc.get(dim) or {}).values():
                if isinstance(v, (int, float)):
                    dim_max = max(dim_max, float(v))
        expected += dim_max
    assert max_possible_score() == expected
    assert max_possible_score() > 0


def test_ceiling_is_a_true_upper_bound_for_every_structure():
    ceiling = max_possible_score()
    cfg = load_affinity_scores()["structures"]
    for sc in cfg.values():
        best = 0.0
        for dim in _SCORED_DIMS:
            vals = [v for v in (sc.get(dim) or {}).values() if isinstance(v, (int, float))]
            best += max(vals) if vals else 0.0
        # No structure's best-case total can exceed the global ceiling.
        assert best <= ceiling + 1e-9


def test_percent_of_max_preserves_ranking():
    # Dividing by a fixed ceiling is monotonic → % order == raw-score order.
    ceiling = max_possible_score()
    raw = [5.5, 4.0, 4.0, 1.0, -2.0]
    pct = [100.0 * r / ceiling for r in raw]
    assert [p for _, p in sorted(zip(raw, pct), reverse=True)] == sorted(pct, reverse=True)
