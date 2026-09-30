import pytest

from agentic.render import _cell_driver_lines
from knowledge_engine.scenario_scorer import contribution_share, CellBreakdown, ScoreResult


@pytest.mark.parametrize("total", [None, 0, -0.02, 0.000001, 0.0001, float("nan"), float("inf")])
def test_unusable_totals_are_na(total):
    assert contribution_share(0.002, total, 0.0001) is None
    assert contribution_share(-0.002, total, 0.0001) is None


def test_signed_absolute_shares_are_bounded():
    contributions = [0.03, 0.01, -0.02]
    shares = [contribution_share(value, sum(map(abs, contributions)), 0.0001) for value in contributions]
    assert shares == pytest.approx([0.5, 1 / 6, -1 / 3])
    assert sum(map(abs, shares)) == pytest.approx(1)


def test_renderer_uses_full_total_not_displayed_subset():
    cell = CellBreakdown("test", "Expiry", "K", 0.02, None, 1, 0.1, 0.002, None)
    text = "\n".join(_cell_driver_lines(([cell], []), total_pct=0.008))
    assert "+25.0% share of total absolute contribution" in text
    assert "scoring notional" not in text
    assert "of trade notional" in text
    assert "original contribution +0.20%" in text
    unavailable = "\n".join(_cell_driver_lines(([cell], []), total_pct=0))
    assert "N/A" in unavailable and "original contribution +0.20%" in unavailable


def test_each_variant_has_its_own_denominator():
    assert contribution_share(0.002, 0.004, 0.0001) == pytest.approx(0.5)
    assert contribution_share(0.002, 0.008, 0.0001) == pytest.approx(0.25)


@pytest.mark.parametrize("values", [[0.006, -0.005], [0.005, -0.006], [0.006, -0.006]])
def test_cancellation_and_negative_net_scores(values):
    cells = [CellBreakdown(str(index), "Expiry", "K", value, None, 1, 1, value, None)
             for index, value in enumerate(values)]
    score = ScoreResult(sum(values), None, cells)
    shares = [contribution_share(value, score.absolute_contribution_total_pct, 0.0001) for value in values]
    assert sum(map(abs, shares)) == pytest.approx(1)
    assert all(abs(share) <= 1 for share in shares)
    assert score.score_pct == sum(values)
    if values == [0.006, -0.005]:
        assert shares == pytest.approx([6 / 11, -5 / 11])


def test_absolute_total_includes_cells_outside_top_bottom():
    cells = [CellBreakdown(str(index), "Expiry", "K", 0.001, None, 1, 0.1, 0.001, None)
             for index in range(10)]
    score = ScoreResult(0.01, None, cells)
    text = "\n".join(_cell_driver_lines((cells[:3], []), total_pct=score.absolute_contribution_total_pct))
    assert text.count("+10.0% share") == 3
