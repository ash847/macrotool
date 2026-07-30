from __future__ import annotations

import pytest
from pydantic import ValidationError

from data.snapshot_loader import load_snapshot
from data.snapshot_overrides import apply_overrides


class TestApplyOverrides:
    def test_empty_overrides_returns_equal_content_copy(self):
        base = load_snapshot()
        out = apply_overrides(base, {})
        assert out is not base
        assert out.model_dump() == base.model_dump()

    def test_forward_override_touches_only_one_tenor(self):
        base = load_snapshot()
        out = apply_overrides(base, {"USDBRL": {"forwards": {"1M": 5.9000}}})
        assert out.get("USDBRL").get_forward("1M").outright == pytest.approx(5.9000)
        assert out.get("USDBRL").get_forward("3M").outright == pytest.approx(
            base.get("USDBRL").get_forward("3M").outright
        )
        assert base.get("USDBRL").get_forward("1M").outright == pytest.approx(5.157)

    def test_forward_points_recomputed_from_spot(self):
        base = load_snapshot()
        out = apply_overrides(base, {"GBPUSD": {"forwards": {"1W": 1.2715}}})
        fwd = out.get("GBPUSD").get_forward("1W")
        assert fwd.points == pytest.approx((1.2715 - out.get("GBPUSD").spot) * 10000)

    def test_atm_override_preserves_rr_and_bf(self):
        base = load_snapshot()
        ccy0 = base.get("USDBRL")
        base_call = ccy0.get_vol("1M", "25DC")
        base_put = ccy0.get_vol("1M", "25DP")
        base_rr = base_call - base_put
        base_bf = 0.5 * (base_call + base_put) - ccy0.get_vol("1M", "ATM")

        out = apply_overrides(base, {"USDBRL": {"atm_vols": {"1M": 0.190}}})
        ccy1 = out.get("USDBRL")
        assert ccy1.get_vol("1M", "ATM") == pytest.approx(0.190)
        assert ccy1.get_vol("1M", "25DC") - ccy1.get_vol("1M", "25DP") == pytest.approx(base_rr)
        assert 0.5 * (ccy1.get_vol("1M", "25DC") + ccy1.get_vol("1M", "25DP")) - ccy1.get_vol("1M", "ATM") == pytest.approx(base_bf)

    def test_rr_and_bf_override_reconstruct_call_put(self):
        base = load_snapshot()
        out = apply_overrides(base, {
            "USDBRL": {
                "atm_vols": {"1M": 0.180},
                "risk_reversals": {"1M": {"25": 0.030}},
                "butterflies": {"1M": {"25": 0.010}},
            }
        })
        ccy = out.get("USDBRL")
        assert ccy.get_vol("1M", "25DC") == pytest.approx(0.205)
        assert ccy.get_vol("1M", "25DP") == pytest.approx(0.175)

    def test_unknown_pair_raises(self):
        base = load_snapshot()
        with pytest.raises(ValueError, match="Unknown pair override"):
            apply_overrides(base, {"USDXXX": {"forwards": {"1M": 5.0}}})

    def test_unknown_tenor_raises(self):
        base = load_snapshot()
        with pytest.raises(ValueError, match="Unknown forward tenor"):
            apply_overrides(base, {"USDBRL": {"forwards": {"4M": 5.9}}})

    def test_invalid_negative_vol_raises_validation_error(self):
        base = load_snapshot()
        with pytest.raises(ValidationError):
            apply_overrides(base, {"USDBRL": {"atm_vols": {"1M": -0.01}}})
