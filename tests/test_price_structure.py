"""Phase 2 tests — the price_structure tool (Tier 2).

Prices PM-style requests against a synthetic high-carry market state. Asserts the
parser→pricer bridge works, that the session (not the LLM) supplies direction /
target / loss budget, and that the three tagged outcomes behave correctly.
"""

import pytest

from analytics.market_state import compute_market_state
from agentic.price_structure import (
    PricedStructure,
    PricingUnavailable,
    price_structure,
)
from agentic.structure_request import ClarificationNeeded, StructureRequestError


def _ms(target=36.0):
    # USDTRY-like: high carry (fwd > spot), topside target (call).
    return compute_market_state(
        spot=30.0,
        fwd=33.0,
        vol=0.20,
        T=0.25,
        r_d=0.40,
        r_f=0.05,
        target=target,
        direction="base_higher",
        surface=None,
    )


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

def test_vanilla_prices():
    res = price_structure("vanilla 25Δ", _ms(), is_call=True, target=36.0)
    assert isinstance(res, PricedStructure)
    assert len(res.variant.strikes) == 1
    assert res.variant.net_premium_pct > 0
    assert res.request.family == "vanilla"


def test_ratio_delta_pair_prices():
    res = price_structure("34 vs 25 1x1.5", _ms(), is_call=True, target=36.0)
    assert isinstance(res, PricedStructure)
    assert len(res.variant.strikes) == 2


def test_ratio_anchored_prices():
    res = price_structure("1x1.5 ATMF vs target", _ms(), is_call=True, target=36.0)
    assert isinstance(res, PricedStructure)
    assert res.variant.payoff_at_target_pct is not None


def test_digital_prices_near_target_premium():
    # A cheap 10% digital solves to a strike well OTM (above a 36 target), so at a
    # 36 target it is still out-of-the-money → payoff 0 (binary, correct).
    res_otm = price_structure("digital 10%", _ms(), is_call=True, target=36.0)
    assert isinstance(res_otm, PricedStructure)
    assert abs(res_otm.variant.net_premium_pct - 0.10) < 0.02
    assert res_otm.variant.payoff_at_target_pct == 0.0
    assert res_otm.variant.strikes[0] > 36.0  # strike sits beyond the target

    # With a target beyond that strike, the base-ccy cash-or-nothing pays 100%.
    res_itm = price_structure("digital 10%", _ms(target=42.0), is_call=True, target=42.0)
    assert isinstance(res_itm, PricedStructure)
    assert res_itm.variant.payoff_at_target_pct == 1.0


def test_loss_budget_populates_ccy_fields():
    res = price_structure("vanilla 25Δ", _ms(), is_call=True, target=36.0, loss_budget=10.0)
    assert isinstance(res, PricedStructure)
    assert res.variant.structure_notional is not None
    assert res.variant.max_loss_ccy is not None


def test_direction_from_session_not_request():
    # A put-side trade (base_lower): direction comes from is_call, not the string.
    ms = _ms(target=30.0)  # target below forward → put
    res = price_structure("vanilla 25Δ", ms, is_call=False, target=30.0)
    assert isinstance(res, PricedStructure)
    assert res.variant.net_premium_pct > 0


# ---------------------------------------------------------------------------
# Tagged non-success outcomes
# ---------------------------------------------------------------------------

def test_clarification_passthrough():
    res = price_structure("34 vs 25", _ms(), is_call=True, target=36.0)
    assert isinstance(res, ClarificationNeeded)


def test_unavailable_when_target_missing():
    # european_rko needs a target → price_variants returns [] → PricingUnavailable.
    res = price_structure("erko 40Δ/20Δ", _ms(target=None), is_call=True, target=None)
    assert isinstance(res, PricingUnavailable)
    assert "target" in res.detail.lower()


def test_malformed_request_raises():
    with pytest.raises(StructureRequestError):
        price_structure("vanilla 120Δ", _ms(), is_call=True, target=36.0)
