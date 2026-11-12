from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from interface import tester_view


@pytest.mark.parametrize("kwargs", [{}, {"is_admin": False}])
def test_ranking_is_hidden_without_admin_access(monkeypatch, kwargs):
    def unexpected(*args, **kw):
        pytest.fail("Admin ranking rendered without admin access")

    monkeypatch.setattr(tester_view.st, "subheader", unexpected)
    monkeypatch.setattr(tester_view.st, "caption", unexpected)
    monkeypatch.setattr(tester_view.st, "dataframe", unexpected)
    monkeypatch.setattr(tester_view, "_render_scoring_grid", unexpected)
    tester_view.render_admin_ranking(SimpleNamespace(), **kwargs)


@pytest.mark.parametrize("count", [3, 20])
def test_ranking_preserves_engine_order_scores_and_caps_at_15(monkeypatch, count):
    tables = []
    inspected = []
    monkeypatch.setattr(tester_view, "_render_scoring_grid",
                        lambda ve, rank, ccy: inspected.append((ve, rank, ccy)))
    monkeypatch.setattr(tester_view.st, "subheader", lambda *a: None)
    monkeypatch.setattr(tester_view.st, "caption", lambda *a: None)
    monkeypatch.setattr(tester_view.st, "dataframe", lambda df, **kw: tables.append(df))
    variants = [SimpleNamespace(
        structure_id="vanilla", struct_label="Vanilla",
        variant_label=f"Call {i}",
        pv=SimpleNamespace(variant_label=f"Call {i}", strikes=[5 + i / 100],
                           structure_notional=1_000_000, net_premium_pct=0.01),
        score=SimpleNamespace(score_ccy=100 - i, score_pct=(100 - i) / 1_000_000),
        score_base=SimpleNamespace(score_ccy=-i, score_pct=-i / 1_000_000),
    ) for i in range(count)]
    # A missing currency score must stay missing, rather than displaying zero.
    variants[1].score.score_ccy = None
    ev = SimpleNamespace(variants=variants, base_ccy="USD", active_ctx="Classic Carry")
    tester_view.render_admin_ranking(ev, is_admin=True)
    table, = tables
    assert list(table.index) == list(range(1, min(count, 15) + 1))
    assert list(table["Variant"]) == [v.variant_label for v in variants[:15]]
    assert table.loc[1, "Engine score (USD)"] == 100
    assert table["Engine score (USD)"].isna().iloc[1]
    assert list(table["Engine score (%)"]) == [v.score.score_pct for v in variants[:15]]
    assert list(table["Baseline score (USD)"]) == [v.score_base.score_ccy for v in variants[:15]]
    assert inspected == [(ve, rank, "USD") for rank, ve in enumerate(variants[:15], 1)]


@pytest.mark.parametrize("scaled", [True, False])
def test_scoring_grid_matches_engine_contributions_including_zero_weights(monkeypatch, scaled):
    from knowledge_engine.scenario_scorer import score_structure

    tables, expanders = [], []
    monkeypatch.setattr(tester_view.st, "expander",
                        lambda title, **kw: (expanders.append((title, kw)) or nullcontext()))
    monkeypatch.setattr(tester_view.st, "caption", lambda *a: None)
    monkeypatch.setattr(tester_view.st, "markdown", lambda *a: None)
    monkeypatch.setattr(tester_view.st, "dataframe", lambda df, **kw: tables.append(df))
    rows = [
        {"scenario_id": "2w|K", "row": "2w", "col": "K", "pnl_pct": .12,
         "pnl_ccy": 120 if scaled else None},
        {"scenario_id": "Expiry|K", "row": "Expiry", "col": "K", "pnl_pct": -.04,
         "pnl_ccy": -40 if scaled else None},
        {"scenario_id": "2w|S", "row": "2w", "col": "S", "pnl_pct": -.08,
         "pnl_ccy": -80 if scaled else None},
    ]
    score = score_structure(rows, {"2w|K": 3, "Expiry|K": 1, "2w|S": 0})
    ve = SimpleNamespace(
        structure_id="vanilla", struct_label="Vanilla", score=score,
        pv=SimpleNamespace(variant_label="Put", strikes=[350]),
    )
    tester_view._render_scoring_grid(ve, 2, "EUR")
    grid, details = tables
    assert expanders[0][0].startswith("#2 Vanilla")
    assert expanders[0][1]["expanded"] is False
    assert list(grid.index) == ["2w", "Expiry"]
    assert grid.loc["2w", "target"] == pytest.approx(.09)
    assert grid.loc["Expiry", "target"] == pytest.approx(-.01)
    assert grid.loc["2w", "No move"] == 0
    assert grid.isna().loc["Expiry", "No move"]
    assert grid.sum().sum() == pytest.approx(score.score_pct)
    assert list(details["P&L × weight (%)"]) == [c.contrib_pct for c in score.cells]
    assert list(details["Weight"]) == [.75, .25, 0]
    if scaled:
        assert details["P&L × weight (EUR)"].sum() == pytest.approx(score.score_ccy)
    else:
        assert details["P&L × weight (EUR)"].isna().all()
