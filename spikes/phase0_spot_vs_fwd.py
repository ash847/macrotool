"""Phase 0 spike (non-destructive): spot- vs forward-anchored evaluation.

Two layers compared side by side, across the 4 active pairs:
  [A] Selection — target_z (σ-distance of target) under spot vs forward anchor,
      with affinity bucket + gate classification. put_call (forward) unchanged.
  [B] Evaluation — scenario-grid terminal spot at the key cells under a
      forward-centered grid (today) vs a spot-centered grid (proposed).

Key identity: target_z_fwd = target_z_spot - c, where c = ln(fwd/spot)/(σ√T).
So the whole shift between anchors is exactly the normalised carry c.
"""
import math
from data.snapshot_loader import load_snapshot
from pricing.forwards import rate_context_for_snapshot, tenor_to_years
from analytics.scenario_generator import generate_scenarios

PAIRS = ["USDBRL", "USDTRY", "EURPLN", "GBPUSD"]
TENOR = "6M"
TZ = [0.5, 1.25, 1.75]          # target_z_abs bucket boundaries
CARRY_CUTS = [0.25, 0.65]


def bucket(z):
    a = abs(z)
    if a < TZ[0]: return "near"
    if a < TZ[1]: return "moderate"
    if a < TZ[2]: return "extended"
    return "far"


def regime(c):
    a = abs(c)
    return 0 if a < CARRY_CUTS[0] else (1 if a < CARRY_CUTS[1] else 2)


snap = load_snapshot()
T = tenor_to_years(TENOR)
print(f"===== Phase 0 spike — {TENOR} (T={T:.3f}) =====\n")
print("Carry shift c (= z_fwd - z_spot offset):")
ctx = {}
for p in PAIRS:
    ccy = snap.get(p); rc = rate_context_for_snapshot(ccy, T)
    spot, fwd = rc.spot, rc.forward; vol = ccy.get_atm_vol(TENOR); sT = vol * math.sqrt(T)
    c = math.log(fwd / spot) / sT
    ctx[p] = (ccy, rc, spot, fwd, vol, sT, c)
    print(f"  {p}: spot={spot:.4f} fwd={fwd:.4f} vol={vol:.1%} σ_T={sT:.4f} "
          f"→ c={c:+.3f} (regime {regime(c)}; fwd {fwd/spot-1:+.1%} vs spot)")

print("\n[A] SELECTION — target at m σ from SPOT (z_spot=m); z_fwd = m - c.")
for p in PAIRS:
    ccy, rc, spot, fwd, vol, sT, c = ctx[p]
    print(f"\n  {p} (c={c:+.3f}):")
    print(f"    {'m σ/spot':>9}{'target':>10}{'z_spot':>8}{'z_fwd':>8}"
          f"{'bkt_spot':>10}{'bkt_fwd':>10}{'p/c':>5}{'1x1≥.5':>8}{'erko≤1.5':>9}  flips")
    for m in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
        target = spot * math.exp(m * sT)
        z_spot, z_fwd = m, math.log(target / fwd) / sT
        pc = "C" if target > fwd else "P"
        g1 = ("Y" if abs(z_spot) >= 0.5 else "N") + "/" + ("Y" if abs(z_fwd) >= 0.5 else "N")
        ge = ("Y" if abs(z_spot) <= 1.5 else "N") + "/" + ("Y" if abs(z_fwd) <= 1.5 else "N")
        flips = []
        if bucket(z_spot) != bucket(z_fwd): flips.append("bucket")
        if (abs(z_spot) >= 0.5) != (abs(z_fwd) >= 0.5): flips.append("1x1-gate")
        if (abs(z_spot) <= 1.5) != (abs(z_fwd) <= 1.5): flips.append("erko-gate")
        print(f"    {m:>9.1f}{target:>10.4f}{z_spot:>8.2f}{z_fwd:>8.2f}"
              f"{bucket(z_spot):>10}{bucket(z_fwd):>10}{pc:>5}{g1:>8}{ge:>9}  {','.join(flips)}")

print("\n[B] EVALUATION — terminal spot at EXPIRY, fwd-centered (today) vs spot-centered.")
for p in PAIRS:
    ccy, rc, spot, fwd, vol, sT, c = ctx[p]
    print(f"\n  {p}: fwd-centered 'no-move' → terminal spot {fwd:.4f} ({fwd/spot-1:+.1%} vs spot); "
          f"spot-centered 'no-move' → {spot:.4f} (0%)")
    for lbl, mm in [("-1σ", -1), ("no-move", 0), ("+1σ", 1)]:
        f_term, s_term = fwd * math.exp(mm * sT), spot * math.exp(mm * sT)
        print(f"    {lbl:>8}: fwd-grid {f_term:>10.4f} ({f_term/spot-1:+6.1%})   "
              f"spot-grid {s_term:>10.4f} ({s_term/spot-1:+6.1%})")

# Empirical validation against the live generator for one pair.
p = "USDTRY"; ccy, rc, spot, fwd, vol, sT, c = ctx[p]
ti = {"spot": spot, "forward": fwd, "target": spot * 1.10, "tenor_years": T,
      "implied_vol": vol, "r_d": rc.r_d, "r_f": rc.r_f}
rows = generate_scenarios(ti)
exp_F = [r for r in rows if r["row"] == "Expiry" and r["fwd_rule"] == "F"]
ss = exp_F[0]["derived"]["scenario_spot"]
print(f"\n  [B-validate] {p} live generate_scenarios Expiry/F ('no-move') scenario_spot = "
      f"{ss:.4f} = fwd {fwd:.4f} ({ss/spot-1:+.1%} vs spot {spot:.4f})")
