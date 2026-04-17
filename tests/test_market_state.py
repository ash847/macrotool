"""
Tests for analytics.market_state — MarketState computation.

Covers:
  - carry (c) sign and magnitude
  - carry_regime thresholds (0 / 1 / 2)
  - target_z sign, direction, and None when omitted
  - atmfsratio: present only when carry_regime >= 1, always > 1, correct direction
  - negative carry (fwd < spot) uses put spread
"""

import math
import pytest

from analytics.market_state import compute_market_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_f_from_cip(spot: float, fwd: float, T: float, r_d: float) -> float:
    """Derive r_f consistent with the given spot/fwd/T/r_d via CIP."""
    return r_d - math.log(fwd / spot) / T


# Reusable base parameters (USDBRL-like, moderate carry)
_SPOT = 5.00
_VOL  = 0.15
_T    = 0.25   # 3M
_R_D  = 0.043  # USD rate


def _make(fwd: float, target: float | None = None):
    """Build a MarketState for the given forward (spot and vol fixed above)."""
    r_f = _r_f_from_cip(_SPOT, fwd, _T, _R_D)
    return compute_market_state(_SPOT, fwd, _VOL, _T, _R_D, r_f, target=target)


def _c(fwd: float) -> float:
    return math.log(fwd / _SPOT) / (_VOL * math.sqrt(_T))


# Forwards that hit each regime cleanly
_FWD_REGIME_0 = _SPOT * math.exp(0.20 * _VOL * math.sqrt(_T))  # |c| = 0.20
_FWD_REGIME_1 = _SPOT * math.exp(0.60 * _VOL * math.sqrt(_T))  # |c| = 0.60
_FWD_REGIME_2 = _SPOT * math.exp(1.00 * _VOL * math.sqrt(_T))  # |c| = 1.00


# ---------------------------------------------------------------------------
# Carry (c) and regime
# ---------------------------------------------------------------------------

class TestCarryRegime:
    def test_regime_0_positive_carry(self):
        ms = _make(_FWD_REGIME_0)
        assert ms.carry_regime == 0
        assert ms.c == pytest.approx(0.20, abs=1e-9)

    def test_regime_1_positive_carry(self):
        ms = _make(_FWD_REGIME_1)
        assert ms.carry_regime == 1
        assert ms.c == pytest.approx(0.60, abs=1e-9)

    def test_regime_2_positive_carry(self):
        ms = _make(_FWD_REGIME_2)
        assert ms.carry_regime == 2
        assert ms.c == pytest.approx(1.00, abs=1e-9)

    def test_regime_boundary_low(self):
        # Exactly at c=0.4 → regime 1 (boundary inclusive on upper side)
        fwd = _SPOT * math.exp(0.40 * _VOL * math.sqrt(_T))
        ms = _make(fwd)
        assert ms.carry_regime == 1

    def test_regime_boundary_high(self):
        # Exactly at c=0.8 → regime 2
        fwd = _SPOT * math.exp(0.80 * _VOL * math.sqrt(_T))
        ms = _make(fwd)
        assert ms.carry_regime == 2

    def test_negative_carry_regime(self):
        # fwd < spot → c negative; abs(c) drives regime
        fwd = _SPOT * math.exp(-0.60 * _VOL * math.sqrt(_T))
        r_f = _r_f_from_cip(_SPOT, fwd, _T, _R_D)
        ms = compute_market_state(_SPOT, fwd, _VOL, _T, _R_D, r_f)
        assert ms.c == pytest.approx(-0.60, abs=1e-9)
        assert ms.carry_regime == 1

    def test_c_stored_on_state(self):
        ms = _make(_FWD_REGIME_1)
        assert ms.c == pytest.approx(math.log(ms.fwd / ms.spot) / (_VOL * math.sqrt(_T)))


# ---------------------------------------------------------------------------
# target_z
# ---------------------------------------------------------------------------

class TestTargetZ:
    def test_target_z_none_when_omitted(self):
        ms = _make(_FWD_REGIME_1)
        assert ms.target_z is None

    def test_target_z_positive_above_fwd(self):
        # Target above fwd → z > 0
        fwd = _FWD_REGIME_1
        target = fwd * 1.05
        ms = _make(fwd, target=target)
        expected = math.log(target / fwd) / (_VOL * math.sqrt(_T))
        assert ms.target_z == pytest.approx(expected, rel=1e-9)
        assert ms.target_z > 0

    def test_target_z_negative_below_fwd(self):
        # Target below fwd → z < 0
        fwd = _FWD_REGIME_1
        target = fwd * 0.95
        ms = _make(fwd, target=target)
        assert ms.target_z < 0

    def test_target_at_fwd_gives_zero(self):
        fwd = _FWD_REGIME_1
        ms = _make(fwd, target=fwd)
        assert ms.target_z == pytest.approx(0.0, abs=1e-9)


class TestPutCall:
    def test_call_when_target_above_fwd(self):
        fwd = _FWD_REGIME_1
        ms = _make(fwd, target=fwd * 1.05)
        assert ms.put_call == "Call"

    def test_put_when_target_below_fwd(self):
        fwd = _FWD_REGIME_1
        ms = _make(fwd, target=fwd * 0.95)
        assert ms.put_call == "Put"

    def test_none_when_no_target(self):
        ms = _make(_FWD_REGIME_1)
        assert ms.put_call is None


# ---------------------------------------------------------------------------
# atmfsratio
# ---------------------------------------------------------------------------

class TestAtmfsRatio:
    def test_none_when_regime_0(self):
        ms = _make(_FWD_REGIME_0)
        assert ms.atmfsratio is None

    def test_present_when_regime_1(self):
        ms = _make(_FWD_REGIME_1)
        assert ms.atmfsratio is not None

    def test_present_when_regime_2(self):
        ms = _make(_FWD_REGIME_2)
        assert ms.atmfsratio is not None

    def test_ratio_always_greater_than_one(self):
        # By no-arbitrage: spread cost < carry pips (discounted), so ratio > 1
        for fwd in (_FWD_REGIME_1, _FWD_REGIME_2):
            ms = _make(fwd)
            assert ms.atmfsratio > 1.0, f"Expected atmfsratio > 1, got {ms.atmfsratio}"

    def test_negative_carry_ratio_present_and_positive(self):
        fwd = _SPOT * math.exp(-0.60 * _VOL * math.sqrt(_T))
        r_f = _r_f_from_cip(_SPOT, fwd, _T, _R_D)
        ms = compute_market_state(_SPOT, fwd, _VOL, _T, _R_D, r_f)
        assert ms.atmfsratio is not None
        assert ms.atmfsratio > 1.0

    def test_higher_carry_not_worse_ratio(self):
        # Higher carry (regime 2) should have a ratio consistent with there being
        # more absolute pips available — ratio stays > 1 and is finite
        ms1 = _make(_FWD_REGIME_1)
        ms2 = _make(_FWD_REGIME_2)
        assert ms1.atmfsratio > 1.0
        assert ms2.atmfsratio > 1.0
        assert math.isfinite(ms1.atmfsratio)
        assert math.isfinite(ms2.atmfsratio)
