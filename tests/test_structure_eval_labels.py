from types import SimpleNamespace

from interface.structure_eval import variant_display_label, variant_label_with_strikes


def test_seagull_display_label_includes_call_spread_wing_ratio():
    pv = SimpleNamespace(
        variant_label="ATMF/25D + 25D wing",
        strikes=[5.0, 5.25, 4.75],
        wing_ratio=0.63,
    )

    assert variant_display_label("seagull", pv) == "1x1 call spread + 0.63x put wing"
    assert (
        variant_label_with_strikes("seagull", pv)
        == "1x1 call spread + 0.63x put wing  ·  Strikes: 5.0000 / 5.2500 / 4.7500"
    )


def test_seagull_display_label_includes_put_spread_wing_ratio():
    pv = SimpleNamespace(
        variant_label="ATMF/25D + 25D wing",
        strikes=[5.0, 4.75, 5.25],
        wing_ratio=1.18,
    )

    assert variant_display_label("seagull", pv) == "1x1 put spread + 1.18x call wing"


def test_non_seagull_display_label_uses_variant_label():
    pv = SimpleNamespace(
        variant_label="ATMF / 25D",
        strikes=[5.0, 5.25],
        wing_ratio=None,
    )

    assert variant_display_label("1x1_spread", pv) == "ATMF / 25D"
