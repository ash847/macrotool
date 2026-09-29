import pytest

from agentic.render import _cell_driver_lines
from knowledge_engine.scenario_scorer import contribution_share, CellBreakdown


@pytest.mark.parametrize("total", [None, 0, -0.02, 0.000001, 0.0001, float("nan"), float("inf")])
def test_unusable_totals_are_na(total):
    assert contribution_share(0.002, total, 0.0001) is None
    assert contribution_share(-0.002, total, 0.0001) is None


def test_signed_shares_can_exceed_one_and_sum_to_one():
    contributions = [0.03, 0.01, -0.02]
    shares = [contribution_share(value, sum(contributions), 0.0001) for value in contributions]
    assert shares == pytest.approx([1.5, 0.5, -1])
    assert sum(shares) == pytest.approx(1)


def test_renderer_uses_full_total_not_displayed_subset():
    cell = CellBreakdown("test", "Expiry", "K", 0.02, None, 1, 0.1, 0.002, None)
    text = "\n".join(_cell_driver_lines(([cell], []), total_pct=0.008))
    assert "+25.0% of net P&L score" in text
    assert "original contribution +0.20%" in text
    unavailable = "\n".join(_cell_driver_lines(([cell], []), total_pct=0))
    assert "N/A" in unavailable and "original contribution +0.20%" in unavailable


def test_each_variant_has_its_own_denominator():
    assert contribution_share(0.002, 0.004, 0.0001) == pytest.approx(0.5)
    assert contribution_share(0.002, 0.008, 0.0001) == pytest.approx(0.25)
