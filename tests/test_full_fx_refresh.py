import pytest

from data.snapshot_loader import load_snapshot
from knowledge_engine.conventions import resolve


@pytest.mark.parametrize("pair,spot,forward", [
    ("EURPLN", 4.3615, 4.422296),
    ("EURHUF", 365.39, 373.7727),
    ("EURUSD", 1.1465, 1.163291),
    ("GBPUSD", 1.3381, 1.338679),
    ("USDJPY", 156.26, 151.5066),
])
def test_refreshed_quotes_and_pair_support(pair, spot, forward):
    snapshot = load_snapshot().get(pair)
    assert str(snapshot.as_of) == "2026-09-16"
    assert snapshot.spot == pytest.approx(spot)
    assert snapshot.get_forward("1Y").outright == pytest.approx(forward)
    assert len(snapshot.vol_surface) == 20
    assert resolve(pair).instrument_type == snapshot.instrument_type


def test_eur_gbp_discount_curves_retained():
    snapshot = load_snapshot()
    tenors = ["1W", "1M", "2M", "3M", "6M", "1Y"]
    eur = [0.9995, 0.9979, 0.9959, 0.9938, 0.9876, 0.9753]
    gbp = [0.9991, 0.9961, 0.9922, 0.9882, 0.9766, 0.9536]
    for pair in ("EURUSD", "EURPLN", "EURHUF"):
        curve = snapshot.get(pair).eur_df_curve
        assert [node.tenor for node in curve] == tenors
        assert [node.df for node in curve] == eur
    curve = snapshot.get("GBPUSD").gbp_df_curve
    assert [node.tenor for node in curve] == tenors
    assert [node.df for node in curve] == gbp
