from types import SimpleNamespace

from interface.structure_eval import variant_display_label, variant_label_with_strikes


def test_seagull_display_label_includes_call_spread_wing_ratio():
    pv = SimpleNamespace(
        variant_label="ATMF/25D + 25D wing",
        strikes=[5.0, 5.25, 4.75],
        wing_ratio=0.63,
    )

    assert variant_display_label("seagull", pv) == "1x1 Call spread + 0.63x put wing"
    assert (
        variant_label_with_strikes("seagull", pv)
        == "1x1 Call spread + 0.63x put wing  ·  Strikes: 5.0000 (ATMF) / 5.2500 (25D) / 4.7500 (25D)"
    )


def test_seagull_display_label_includes_put_spread_wing_ratio():
    pv = SimpleNamespace(
        variant_label="ATMF/25D + 25D wing",
        strikes=[5.0, 4.75, 5.25],
        wing_ratio=1.18,
    )

    assert variant_display_label("seagull", pv) == "1x1 Put spread + 1.18x call wing"


def test_vanilla_label_is_only_the_strike_in_title():
    pv = SimpleNamespace(
        variant_label="ATMF (50D)",
        strikes=[5.0],
        wing_ratio=None,
    )

    assert variant_display_label("vanilla", pv) == "Vanilla option"
    assert variant_label_with_strikes("vanilla", pv) == "5.0000"


def test_one_by_one_spread_label_uses_product_name_and_strikes():
    pv = SimpleNamespace(
        variant_label="ATMF / 25D",
        strikes=[5.0, 5.25],
        wing_ratio=None,
    )

    assert variant_display_label("1x1_spread", pv) == "1x1 Call spread"
    assert (
        variant_label_with_strikes("1x1_spread", pv)
        == "1x1 Call spread  ·  Strikes: 5.0000 (ATMF) / 5.2500 (25D)"
    )


def test_ratio_spread_label_uses_product_name_and_strikes():
    pv = SimpleNamespace(
        variant_label="ATMF / 1.5x target",
        strikes=[5.0, 5.25],
        wing_ratio=None,
    )

    assert variant_display_label("1x1.5_spread", pv) == "1x1.5 call Ratio spread"
    assert (
        variant_label_with_strikes("1x1.5_spread", pv)
        == "1x1.5 call Ratio spread  ·  Strikes: 5.0000 (ATMF) / 5.2500"
    )


def test_ratio_spread_label_uses_put_direction():
    pv = SimpleNamespace(
        variant_label="25D / 10D",
        strikes=[5.0, 4.75],
        wing_ratio=None,
    )

    assert variant_display_label("1x2_spread", pv) == "1x2 put Ratio spread"
    assert (
        variant_label_with_strikes("1x2_spread", pv)
        == "1x2 put Ratio spread  ·  Strikes: 5.0000 (25D) / 4.7500 (10D)"
    )


def test_european_digital_label_uses_barrier_level():
    pv = SimpleNamespace(
        variant_label="~20% prem",
        strikes=[5.25],
        barrier=None,
        wing_ratio=None,
    )

    assert variant_display_label("european_digital", pv) == "European digital"
    assert (
        variant_label_with_strikes("european_digital", pv)
        == "European digital  ·  Barrier: 5.2500"
    )


def test_european_rko_label_uses_strike_and_barrier():
    pv = SimpleNamespace(
        variant_label="ATMF / 25D",
        strikes=[5.0],
        barrier=5.40,
        wing_ratio=None,
    )

    assert variant_display_label("european_rko", pv) == "European RKO"
    assert (
        variant_label_with_strikes("european_rko", pv)
        == "European RKO  ·  Strike: 5.0000  ·  Barrier: 5.4000"
    )
