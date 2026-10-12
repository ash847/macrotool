# Full FX quote refresh

All 13 workbook pairs use the latest observation, 2026-09-16, for spot,
four forward tenors and five smile nodes per tenor. EURHUF is newly enabled.
Existing EUR and GBP discount curves are unchanged; EURHUF uses a copy of
the existing EURUSD EUR curve. Their vintage remains older than the FX quotes.
USDJPY now uses the same-date USD SOFR curve already used by the other USD pairs.

Confirmed point multipliers: 0.01 for JPY/KRW/INR/HUF; 0.0001 for all other
workbook pairs, including EURPLN. Forward = spot + points times multiplier.
Vol, RR and BF source values are percentages. Calls/puts use ATM + BF +/- RR/2.
Only observed 1M/3M/6M/1Y pillars are stored; the engine handles interpolation.

This refresh changes market fixtures and adds EURHUF conventions only. No
pricing logic or historical backtest inputs/results are changed. No workbook
or backtest-lab runtime dependency is introduced. Quote provenance is recorded
in latest_fx_snapshot_provenance.json. Contractual conventions require normal
execution checks; these remain research fixtures rather than executable quotes.
