from conversation.flow import target_from_reference


def test_target_from_reference_base_higher_uses_reference_anchor():
    assert abs(target_from_reference(1.10, "base_higher", 5.0) - 1.155) < 1e-9


def test_target_from_reference_base_lower_uses_reference_anchor():
    assert abs(target_from_reference(1.10, "base_lower", 5.0) - 1.045) < 1e-9


def test_target_from_reference_none_magnitude_returns_none():
    assert target_from_reference(1.10, "base_higher", None) is None
