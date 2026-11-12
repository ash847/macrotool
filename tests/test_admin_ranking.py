from types import SimpleNamespace

import pytest

from interface import tester_view


@pytest.mark.parametrize("kwargs", [{}, {"is_admin": False}])
def test_ranking_is_hidden_without_admin_access(monkeypatch, kwargs):
    def unexpected(*args, **kw):
        pytest.fail("Admin ranking rendered without admin access")

    monkeypatch.setattr(tester_view.st, "subheader", unexpected)
    monkeypatch.setattr(tester_view.st, "caption", unexpected)
    monkeypatch.setattr(tester_view.st, "dataframe", unexpected)
    tester_view.render_admin_ranking(SimpleNamespace(), **kwargs)


@pytest.mark.parametrize("count", [3, 20])
def test_ranking_preserves_engine_order_scores_and_caps_at_15(monkeypatch, count):
    tables = []
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
