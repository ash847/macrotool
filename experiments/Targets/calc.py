"""
FX View Distribution Tool — pure calculation module.
No Streamlit imports. No side effects.
"""
from __future__ import annotations

import math
from scipy.stats import norm


# ── Core ────────────────────────────────────────────────────────────────────

def horizon_vol(vol: float, T: float) -> float:
    return vol * math.sqrt(T)


def _ln_cdf(x: float, F: float, sigma_T: float) -> float:
    """P(S_T ≤ x) under ln(S_T/F) ~ N(0, sigma_T²)."""
    if x <= 0:
        return 0.0
    return norm.cdf(math.log(x / F) / sigma_T)


def _bucket_q(lo: float | None, hi: float | None, F: float, sigma_T: float) -> float:
    cdf_hi = 1.0 if hi is None else _ln_cdf(hi, F, sigma_T)
    cdf_lo = 0.0 if lo is None else _ln_cdf(lo, F, sigma_T)
    return max(0.0, cdf_hi - cdf_lo)


# ── Bucket construction ─────────────────────────────────────────────────────

def generate_boundaries(F: float, sigma_T: float) -> list[float]:
    """
    Seven interior boundaries at z = -2, -1, -0.5, 0, +0.5, +1, +2,
    giving 8 equal-sigma buckets. Target plays no role here.
    """
    z_levels = [-2, -1, -0.5, 0, 0.5, 1, 2]
    return [F * math.exp(z * sigma_T) for z in z_levels]


def _label(lo: float | None, hi: float | None) -> str:
    if lo is None:
        return f"< {hi:.4f}"
    if hi is None:
        return f"> {lo:.4f}"
    return f"{lo:.4f} – {hi:.4f}"


def _short_label(lo: float | None, hi: float | None) -> str:
    fmt = lambda x: f"{x:.3f}"
    if lo is None:
        return f"< {fmt(hi)}"
    if hi is None:
        return f"> {fmt(lo)}"
    return f"{fmt(lo)} – {fmt(hi)}"


def make_buckets(F: float, sigma_T: float) -> list[dict]:
    """
    Returns list of 8 bucket dicts:
      lower: float | None  (None = -inf)
      upper: float | None  (None = +inf)
      label: str           (full 4dp)
      short: str           (3dp, for sliders / chart)
      q:     float         (market probability, 0–100)
    """
    bounds = generate_boundaries(F, sigma_T)
    edges: list[float | None] = [None] + bounds + [None]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = _bucket_q(lo, hi, F, sigma_T) * 100.0
        out.append({
            'lower': lo,
            'upper': hi,
            'label': _label(lo, hi),
            'short': _short_label(lo, hi),
            'q': round(q, 2),
        })
    return out


# ── Probability initialisation ──────────────────────────────────────────────

def make_zone_buckets(parent: dict, F: float, sigma_T: float, n: int = 6) -> list[dict]:
    """
    Split a finite bucket into n equal-width (spot space) sub-buckets.
    Only valid for middle buckets where both lower and upper are finite.
    """
    lo, hi = parent["lower"], parent["upper"]
    width = (hi - lo) / n
    out = []
    for i in range(n):
        sub_lo = lo + i * width
        sub_hi = lo + (i + 1) * width
        q = _bucket_q(sub_lo, sub_hi, F, sigma_T) * 100
        out.append({
            "lower": sub_lo,
            "upper": sub_hi,
            "label": _label(sub_lo, sub_hi),
            "short": f"{sub_lo:.4f} – {sub_hi:.4f}",
            "q": round(q, 3),
        })
    return out


def init_zone_p(zone_buckets: list[dict], parent_p: int) -> list[int]:
    """
    Initialise sub-bucket P proportional to Q within the zone, summing to parent_p.
    Falls back to uniform if Q is zero throughout.
    """
    sub_qs = [b["q"] for b in zone_buckets]
    total_q = sum(sub_qs)
    n = len(zone_buckets)
    if total_q <= 0:
        base = parent_p // n
        p = [base] * n
        p[n // 2] += parent_p - sum(p)
        return p
    p = [round(q / total_q * parent_p) for q in sub_qs]
    diff = parent_p - sum(p)
    if diff:
        p[p.index(max(p))] += diff
    return p


def init_p_from_q(buckets: list[dict]) -> list[int]:
    """Round Q probabilities to integers summing to exactly 100."""
    p = [round(b['q']) for b in buckets]
    diff = 100 - sum(p)
    if diff:
        p[p.index(max(p))] += diff
    return p


# ── Target zone ─────────────────────────────────────────────────────────────

PRECISION_SIGMA = {'Tight': 0.25, 'Medium': 0.5, 'Loose': 1.0}


def target_zone(K: float, sigma_T: float, precision: str) -> tuple[float, float]:
    s = PRECISION_SIGMA[precision]
    return K * math.exp(-s * sigma_T), K * math.exp(s * sigma_T)


# ── Derived metrics ─────────────────────────────────────────────────────────

def _p_above(
    level: float,
    buckets: list[dict],
    p_vals: list[int],
    F: float,
    sigma_T: float,
) -> float:
    """
    User probability (0–100) above `level`.
    Within-bucket interpolation uses the Q (lognormal) shape as a reference.
    """
    lev_cdf = _ln_cdf(level, F, sigma_T)
    total = 0.0
    for b, p in zip(buckets, p_vals):
        lo_cdf = 0.0 if b['lower'] is None else _ln_cdf(b['lower'], F, sigma_T)
        hi_cdf = 1.0 if b['upper'] is None else _ln_cdf(b['upper'], F, sigma_T)
        q = hi_cdf - lo_cdf
        if lo_cdf >= lev_cdf:
            total += p
        elif hi_cdf > lev_cdf and q > 0:
            total += p * (hi_cdf - lev_cdf) / q
    return total


def compute_metrics(
    buckets: list[dict],
    p_vals: list[int],
    F: float,
    K: float,
    sigma_T: float,
    precision: str,
) -> dict:
    p_dir = _p_above(F, buckets, p_vals, F, sigma_T)
    q_dir = 50.0  # lognormal centred at F → P(S_T > F) = 50% by construction

    p_target = _p_above(K, buckets, p_vals, F, sigma_T)
    q_target = (1 - _ln_cdf(K, F, sigma_T)) * 100

    z_lo, z_hi = target_zone(K, sigma_T, precision)
    p_zone = (
        _p_above(z_lo, buckets, p_vals, F, sigma_T)
        - _p_above(z_hi, buckets, p_vals, F, sigma_T)
    )
    q_zone = (_ln_cdf(z_hi, F, sigma_T) - _ln_cdf(z_lo, F, sigma_T)) * 100

    if K >= F:
        ov_lv = K * math.exp(0.5 * sigma_T)
        p_ov = _p_above(ov_lv, buckets, p_vals, F, sigma_T)
        q_ov = (1 - _ln_cdf(ov_lv, F, sigma_T)) * 100
        p_tail = float(p_vals[0])
        q_tail = buckets[0]['q']
    else:
        ov_lv = K * math.exp(-0.5 * sigma_T)
        p_ov = 100.0 - _p_above(ov_lv, buckets, p_vals, F, sigma_T)
        q_ov = _ln_cdf(ov_lv, F, sigma_T) * 100
        p_tail = float(p_vals[-1])
        q_tail = buckets[-1]['q']

    delta = [p_vals[i] - buckets[i]['q'] for i in range(len(buckets))]

    return {
        'p_dir': p_dir,       'q_dir': q_dir,
        'p_target': p_target, 'q_target': q_target,
        'p_zone': p_zone,     'q_zone': q_zone,
        'zone_lo': z_lo,      'zone_hi': z_hi,
        'p_overshoot': p_ov,  'q_overshoot': q_ov,
        'overshoot_level': ov_lv,
        'p_tail': p_tail,     'q_tail': q_tail,
        'delta': delta,
    }
