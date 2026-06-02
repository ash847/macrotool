"""
Premium currency-basis regression guard.

The engine quotes option premiums as a fraction of *base-currency (USD)*
notional via ``net_premium = black76(...) / spot``. Black-76 returns a premium
in *quote* currency (e.g. TRY) discounted at the *quote* rate ``r_d``. Dividing
that quote-ccy premium by spot algebraically collapses the quote-rate discount
into the *base* rate ``r_f``:

    black76(F,K,T,σ,DF_d) / spot  ==  DF_f * (forward_value / forward)

i.e. the displayed premium is base-ccy-notional, effectively discounted at the
base (USD) rate. This is the correct, intended basis.

These tests pin that identity on a deliberately *high-carry* pair (TRY-like),
where DF_d (~0.81) and DF_f (~0.99) differ a lot. If anyone ever reintroduces
the basis bug — dividing the quote-discounted value by *forward* instead of
spot, flipping /spot to /fwd, or dropping the discount factor — the premium
would land ~19% (the quote DF) away from the forward basis and these fail.

See CLAUDE.md "JSON vs Python" / pricing notes; the bug class this guards
against was diagnosed live during the USDTRY 1x2 trace.
"""

from __future__ import annotations

import math

import pytest

from analytics.market_state import compute_market_state
from analytics.structure_pricer import price_variants
from pricing.black_scholes import _N


# High-carry, TRY-like market: spot 38.45, fwd 46.76, USD r_f ~4.5%, TRY r_d ~84%.
_SPOT = 38.45
_FWD = 46.7597
_VOL = 0.2121
_T = 90 / 365.0
_R_F = 0.0452
_R_D = _R_F + math.log(_FWD / _SPOT) / _T
_TARGET = 55.0
_DF_F = math.exp(-_R_F * _T)
_DF_D = math.exp(-_R_D * _T)


def _fwd_call_value(K: float) -> float:
    """Undiscounted Black-76 forward value (quote ccy) of a call struck at K."""
    d1 = (math.log(_FWD / K) + 0.5 * _VOL * _VOL * _T) / (_VOL * math.sqrt(_T))
    d2 = d1 - _VOL * math.sqrt(_T)
    return _FWD * _N(d1) - K * _N(d2)


def _market_state():
    return compute_market_state(
        _SPOT, _FWD, _VOL, _T, _R_D, _R_F, target=_TARGET, direction="base_higher"
    )


def test_high_carry_market_actually_stresses_the_basis():
    """Sanity: the two discount factors must differ enough to expose a bug."""
    # If DF_d ~ DF_f the test would be vacuous; TRY-like carry guarantees a gap.
    assert _DF_D < 0.85
    assert _DF_F > 0.98
    assert _DF_F / _DF_D > 1.15


@pytest.mark.parametrize(
    "structure_id, expected_value",
    [
        ("vanilla", lambda K: _fwd_call_value(K[0])),
        ("1x1_spread", lambda K: _fwd_call_value(K[0]) - _fwd_call_value(K[1])),
        ("1x1.5_spread", lambda K: _fwd_call_value(K[0]) - 1.5 * _fwd_call_value(K[1])),
        ("1x2_spread", lambda K: _fwd_call_value(K[0]) - 2.0 * _fwd_call_value(K[1])),
    ],
)
def test_premium_is_base_ccy_notional_on_forward_basis(structure_id, expected_value):
    """net_premium_pct == DF_f * (undiscounted forward value / forward).

    This is the load-bearing invariant: it confirms the premium is base-ccy %
    discounted at the *base* rate, NOT the quote rate over forward (the bug).
    """
    ms = _market_state()
    variants = price_variants(ms, structure_id, target=_TARGET, is_call=True, stop_price=None)
    assert variants, f"no variants priced for {structure_id}"

    for pv in variants:
        fwd_value = expected_value(pv.strikes)
        expected_pct = _DF_F * fwd_value / _FWD
        assert pv.net_premium_pct == pytest.approx(expected_pct, abs=1e-4), (
            f"{structure_id} {pv.variant_label}: premium basis drifted — "
            f"got {pv.net_premium_pct:.5f}, forward-basis {expected_pct:.5f}"
        )


def test_quote_rate_discount_over_forward_is_the_wrong_answer():
    """Guard the specific bug: quote-discounted value / forward is ~19% low.

    Confirms the engine does NOT produce the buggy number, so the assertion in
    the test above is discriminating rather than coincidental.
    """
    ms = _market_state()
    pv = price_variants(ms, "vanilla", target=_TARGET, is_call=True, stop_price=None)[0]
    K = pv.strikes[0]
    buggy_pct = _DF_D * _fwd_call_value(K) / _FWD  # quote-disc value over forward
    # The buggy basis must be materially different from what the engine returns.
    assert abs(pv.net_premium_pct - buggy_pct) > 5e-3
