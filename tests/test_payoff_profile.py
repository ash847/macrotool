"""Guards for the engine-authored payoff geometry (agent narration).

These pin the exact facts the agent used to get wrong (feedback items #1/#5/#8/#13):
which side the tail is on, where the value region starts/ends, premium direction,
and expiry-vs-path nature. The geometry is computed from the legs, so the guards are
directional-correctness guards, not string-format guards.
"""

from __future__ import annotations

import pytest

from knowledge_engine.payoff_profile import payoff_profile, render_payoff


# --- 1x2 call: the #8 case — short ABOVE the upper breakeven, worthless below K1 ---
def test_1x2_call_tail_is_upside_not_below_k2():
    legs = [(1.0, 5.5694, True), (-2.0, 5.9000, True)]
    p = payoff_profile("1x2_spread", legs, net_premium_pct=0.008, is_zero_cost=False, is_call=True)
    assert p.tail == "loss_upside"
    # upper breakeven = 2*K2 - K1 = 6.2306; the tail opens ABOVE it, never below K1
    assert p.breakevens == (6.2306,)
    assert "above 6.2306" in p.tail_where
    assert p.value_region == "between 5.5694 and 6.2306"     # NOT extending below K1
    assert p.max_payoff_where == "at 5.9000"                  # peak at the short strike


def test_1x2_put_tail_is_downside():
    legs = [(1.0, 5.60, False), (-2.0, 5.30, False)]
    p = payoff_profile("1x2_spread", legs, net_premium_pct=0.008, is_zero_cost=False, is_call=False)
    assert p.tail == "loss_downside"
    assert "below 5.0000" in p.tail_where          # lower BE = 2*5.30 - 5.60 = 5.00
    assert p.value_region == "between 5.0000 and 5.6000"


def test_put_call_symmetry_flips_tail_side():
    call = payoff_profile("1x2_spread", [(1.0, 5.57, True), (-2.0, 5.90, True)],
                          net_premium_pct=0.0, is_zero_cost=False, is_call=True)
    put = payoff_profile("1x2_spread", [(1.0, 5.90, False), (-2.0, 5.57, False)],
                         net_premium_pct=0.0, is_zero_cost=False, is_call=False)
    assert call.tail == "loss_upside"
    assert put.tail == "loss_downside"


# --- capped families ---
def test_vanilla_call_capped_loss_uncapped_upside():
    p = payoff_profile("vanilla", [(1.0, 5.60, True)], net_premium_pct=0.02, is_zero_cost=False, is_call=True)
    assert p.tail == "capped"
    assert p.value_region == "above 5.6000"
    assert p.max_payoff_where == "uncapped on a further move up"


def test_call_spread_capped_both_sides():
    p = payoff_profile("1x1_spread", [(1.0, 5.60, True), (-1.0, 5.90, True)],
                       net_premium_pct=0.01, is_zero_cost=False, is_call=True)
    assert p.tail == "capped"
    assert p.value_region == "between 5.6000 and 5.9000"


def test_1x2x1_butterfly_caps_the_1x2_tail():
    legs = [(1.0, 5.5694, True), (-2.0, 5.9000, True), (1.0, 6.2306, True)]
    p = payoff_profile("1x2x1_spread", legs, net_premium_pct=0.012, is_zero_cost=False, is_call=True)
    assert p.tail == "capped"                                # the wing closes the tail
    assert p.value_region == "between 5.5694 and 6.2306"


# --- ERKO: the #13 case — barrier is a distinct role from the strike, expiry-only ---
def test_erko_states_barrier_distinct_from_strike_and_is_expiry_only():
    p = payoff_profile("european_rko", [], net_premium_pct=0.015, is_zero_cost=False,
                       is_call=False, strikes=[4.2100], barrier=4.0283)
    assert p.product_nature == "expiry_only"                 # NOT path_dependent (kills #12)
    assert "4.2100" in p.value_region and "4.0283" in p.value_region
    assert "nothing" in p.value_region.lower()               # pays nothing beyond the barrier
    line = render_payoff(p)
    assert "not path-dependent" in line


def test_rko_is_path_dependent():
    p = payoff_profile("rko", [], net_premium_pct=0.02, is_zero_cost=False,
                       is_call=True, strikes=[5.80], barrier=6.20)
    assert p.product_nature == "path_dependent"


# --- digital: binary, no breakeven, fixed payout ---
def test_digital_is_binary_with_no_breakeven():
    p = payoff_profile("european_digital", [], net_premium_pct=0.02, is_zero_cost=False,
                       is_call=True, strikes=[5.80])
    assert p.product_nature == "binary_expiry"
    assert p.breakevens == ()
    assert "above 5.8000" in p.value_region


# --- premium direction (the #9/#10/#11 root) ---
@pytest.mark.parametrize("prem,zc,expected", [
    (0.02, False, "debit"),
    (-0.02, False, "credit"),
    (0.0, True, "zero_cost"),
])
def test_premium_flow_direction(prem, zc, expected):
    p = payoff_profile("vanilla", [(1.0, 5.60, True)], net_premium_pct=prem, is_zero_cost=zc, is_call=True)
    assert p.premium_flow == expected


def test_render_debit_says_you_pay_credit_says_you_receive():
    debit = payoff_profile("vanilla", [(1.0, 5.60, True)], net_premium_pct=0.02, is_zero_cost=False, is_call=True)
    credit = payoff_profile("1x2_spread", [(1.0, 5.60, True), (-2.0, 5.90, True)],
                            net_premium_pct=-0.01, is_zero_cost=False, is_call=True)
    assert "you pay the premium" in render_payoff(debit)
    assert "you receive the premium" in render_payoff(credit)


# --- consistency: the rendered line mints no number that isn't a strike/breakeven ---
def test_rendered_numbers_are_only_strikes_or_breakevens():
    import re
    legs = [(1.0, 5.5694, True), (-2.0, 5.9000, True)]
    p = payoff_profile("1x2_spread", legs, net_premium_pct=0.0, is_zero_cost=False, is_call=True)
    line = render_payoff(p)
    allowed = {"5.5694", "5.9000", "6.2306"}
    nums = set(re.findall(r"\d+\.\d{4}", line))
    assert nums <= allowed


# --- fork flex: pre_expiry_note renders only when populated ---
def test_pre_expiry_note_absent_by_default_and_appended_when_set():
    import dataclasses
    p = payoff_profile("vanilla", [(1.0, 5.60, True)], net_premium_pct=0.02, is_zero_cost=False, is_call=True)
    assert p.pre_expiry_note is None
    assert "MtM" not in render_payoff(p)
    p2 = dataclasses.replace(p, pre_expiry_note="pre-expiry MtM can differ")
    assert "pre-expiry MtM can differ" in render_payoff(p2)
