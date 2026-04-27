"""
Tests for knowledge_engine.structure_scorer.

Covers:
  - Gating: structures requiring a target are excluded when no target supplied
  - Gating: structures with target_z_abs_min gate excluded when target is too near fwd
  - Ranking: vanilla wins when no target + low carry
  - Ranking: 1x1_spread wins when target at extended distance + high carry
  - Ranking: risk_reversal competitive when no target + moderate carry
  - Overlay structures appear after primary structures in the shortlist
  - Overlay structures never occupy primary slots
  - Score notes recorded in rules_fired
"""

import math
import pytest

from analytics.market_state import compute_market_state
from knowledge_engine.structure_scorer import score_structures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_f(spot, fwd, T, r_d):
    return r_d - math.log(fwd / spot) / T

_SPOT = 5.00
_VOL  = 0.15
_T    = 0.25
_R_D  = 0.043

def _fwd_for_c(c: float) -> float:
    return _SPOT * math.exp(c * _VOL * math.sqrt(_T))

def _ms(c: float = 0.20, target_z: float | None = None):
    fwd = _fwd_for_c(c)
    r_f = _r_f(_SPOT, fwd, _T, _R_D)
    target = None
    if target_z is not None:
        target = fwd * math.exp(target_z * _VOL * math.sqrt(_T))
    return compute_market_state(_SPOT, fwd, _VOL, _T, _R_D, r_f, target=target)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

class TestGating:
    def test_spread_excluded_when_no_target(self):
        ms = _ms(c=0.60)  # regime 1, no target
        result = score_structures(ms)
        ids = [s.structure_id for s in result.shortlist]
        assert "1x1_spread" not in ids
        assert "seagull" not in ids
        assert "1x2_spread" not in ids

    def test_spread_excluded_when_target_too_near(self):
        # target_z = 0.3 < 0.5 gate threshold
        ms = _ms(c=0.60, target_z=0.3)
        result = score_structures(ms)
        ids = [s.structure_id for s in result.shortlist]
        assert "1x1_spread" not in ids

    def test_spread_included_when_target_at_gate_boundary(self):
        # target_z exactly at 0.5 — should pass the >= 0.5 gate
        ms = _ms(c=0.60, target_z=0.5)
        result = score_structures(ms)
        ids = [s.structure_id for s in result.shortlist]
        assert "1x1_spread" in ids

    def test_vanilla_never_gated_out(self):
        for c in (0.1, 0.6, 1.0):
            ms = _ms(c=c)
            result = score_structures(ms)
            ids = [s.structure_id for s in result.shortlist]
            assert "vanilla" in ids


# ---------------------------------------------------------------------------
# Ranking — primary structures
# ---------------------------------------------------------------------------

class TestRanking:
    def test_vanilla_wins_no_target_low_carry(self):
        # carry regime 0, no target → vanilla tops (affinity tuned to prefer simplicity
        # when carry signal is noisy and no target is given)
        ms = _ms(c=0.20)  # regime 0
        result = score_structures(ms)
        primaries = [s for s in result.shortlist if not s.is_exotic]
        assert primaries[0].structure_id == "vanilla"

    def test_vanilla_wins_far_target_high_carry(self):
        # carry regime 2 (c=1.0), target at 2.0σ (far) → vanilla tops due to far=3.0
        # plus carry_regime and carry_alignment scores added in user tuning
        ms = _ms(c=1.0, target_z=2.0)
        result = score_structures(ms)
        primaries = [s for s in result.shortlist if not s.is_exotic]
        assert primaries[0].structure_id == "vanilla"

    def test_risk_reversal_permanently_gated(self):
        # risk_reversal has target_z_abs_min=999 gate — never eligible
        ms = _ms(c=0.60)
        result = score_structures(ms)
        ids = [s.structure_id for s in result.shortlist]
        assert "risk_reversal" not in ids

    def test_primary_count_capped_at_three(self):
        ms = _ms(c=0.60, target_z=1.5)
        result = score_structures(ms, max_primary=3)
        primaries = [s for s in result.shortlist if not s.is_exotic]
        assert len(primaries) <= 3

    def test_ranks_are_sequential_from_one(self):
        ms = _ms(c=0.60, target_z=1.5)
        result = score_structures(ms)
        ranks = [s.rank for s in result.shortlist]
        assert ranks == list(range(1, len(ranks) + 1))


# ---------------------------------------------------------------------------
# Overlay structures
# ---------------------------------------------------------------------------

class TestOverlays:
    def test_overlays_appear_after_primaries(self):
        ms = _ms(c=0.60, target_z=1.0)
        result = score_structures(ms)
        seen_overlay = False
        for item in result.shortlist:
            if item.is_exotic:
                seen_overlay = True
            else:
                assert not seen_overlay, "Primary appeared after overlay"

    def test_overlays_present_in_shortlist(self):
        ms = _ms(c=0.60, target_z=1.0)
        result = score_structures(ms)
        overlay_ids = {s.structure_id for s in result.shortlist if s.is_exotic}
        assert len(overlay_ids) > 0

    def test_rko_is_overlay(self):
        ms = _ms(c=0.60, target_z=1.0)
        result = score_structures(ms)
        rko_items = [s for s in result.shortlist if s.structure_id == "rko"]
        assert len(rko_items) == 1
        assert rko_items[0].is_exotic is True


# ---------------------------------------------------------------------------
# Score notes
# ---------------------------------------------------------------------------

class TestScoreNotes:
    def test_rules_fired_contains_score_notes(self):
        ms = _ms(c=0.60, target_z=1.5)
        result = score_structures(ms)
        assert len(result.rules_fired) == len(result.shortlist)
        for note in result.rules_fired:
            assert "=" in note  # format: "struct_id=score"
