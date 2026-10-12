import math

import pytest

from agentic.standard_pack import build_pack
from analytics.distributions import interpolate_atm_vol
from analytics.vol_surface import build_vol_surface
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.conventions import resolve
from knowledge_engine.models import TradeView
from pricing.forwards import rate_context_for_snapshot


PAIRS = ("USDMXN", "USDBRL", "USDTRY", "USDZAR", "USDCNH", "USDKRW", "USDSGD", "USDINR")


@pytest.mark.parametrize("pair", PAIRS)
def test_latest_snapshot_inputs(pair):
    snapshot = load_snapshot().get(pair)
    assert str(snapshot.as_of) == "2026-09-16"
    assert resolve(pair).instrument_type == snapshot.instrument_type
    assert len(snapshot.forwards) == 4
    assert len(snapshot.vol_surface) == 20
    multiplier = 0.01 if pair in ("USDKRW", "USDINR") else 0.0001
    for forward in snapshot.forwards:
        assert forward.outright == pytest.approx(snapshot.spot + forward.points * multiplier)
    assert all(0 < node.df < 1 for node in snapshot.usd_df_curve)
    assert all(math.isfinite(node.vol) and node.vol > 0 for node in snapshot.vol_surface)
    build_vol_surface(snapshot)


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("horizon", (30, 91, 182, 365))
@pytest.mark.parametrize("direction", ("base_higher", "base_lower"))
def test_latest_pairs_full_engine(pair, horizon, direction):
    snapshot = load_snapshot().get(pair)
    rates = rate_context_for_snapshot(snapshot, horizon / 365)
    vol = interpolate_atm_vol(snapshot, horizon)
    magnitude = 100 * vol * math.sqrt(horizon / 365)
    view = TradeView(pair=pair, direction=direction, direction_conviction="high",
                     horizon_days=horizon, magnitude_pct=magnitude)
    pack = build_pack(view, snapshot, load_config(), linear_notional=1_000_000, target_rr=3)
    assert pack.recommended
    assert pack.market_state.surface is not None
    assert rates.forward == pytest.approx(snapshot.spot * math.exp((rates.r_d - rates.r_f) * horizon / 365))
    assert pack.loss_budget == pytest.approx(1_000_000 * magnitude / 100 / 3)
    for recommendation in pack.recommended:
        variant = recommendation.variant
        assert math.isfinite(recommendation.score_ccy)
        assert math.isfinite(variant.net_premium_ccy)
        assert 0 < variant.structure_notional <= 10_000_000
        assert all(math.isfinite(strike) and strike > 0 for strike in variant.strikes)
    if horizon == 91 and direction == "base_higher":
        best = max(pack.recommended, key=lambda item: item.score_ccy)
        print(f"RESULT {pair} spot={snapshot.spot:.6f} fwd={rates.forward:.6f} "
              f"vol={vol:.6f} target={pack.target:.6f} budget={pack.loss_budget:.2f} "
              f"best={best.structure_id}/{best.variant.variant_label} "
              f"premium={best.variant.net_premium_ccy:.2f} "
              f"notional={best.variant.structure_notional:.2f} score={best.score_ccy:.2f}")
