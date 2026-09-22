# FX pair support update

Eight USD-base pairs use 2026-09-16 observations from Fx vols.xlsx: USDMXN,
USDBRL, USDTRY, USDZAR, USDCNH, USDKRW, USDSGD and USDINR. New convention
files enable ZAR, SGD, KRW and INR. The agent prompt, tool validation and UI
pair lists use the loaded snapshot, so no agent orchestration changes are needed.

GBPUSD, EURUSD, EURPLN and USDJPY retain the branch's prior snapshot entries
without changes. Per-pair as-of dates remain authoritative for this mixed-date
research snapshot. EURHUF remains unavailable without a base discount curve.

Forward multipliers are 0.0001 for MXN/BRL/TRY/ZAR/CNH/SGD and 0.01 for
KRW/INR. KRW's negligible forward carry was explicitly confirmed by the user.
Only observed 1M/3M/6M/1Y pillars are stored; existing engine interpolation
handles other horizons. Source vols and RR/BF are percentage values.

USD DFs are copied from the validated historical SOFR bootstrap on the same
date. All four pillars were complete; no missing-data fills were needed.
There is no runtime dependency on the workbook or the independent backtest lab.
See latest_fx_snapshot_provenance.json for conversion rules and source details.

New convention files use the engine's forward-delta/USD-premium modelling
defaults. Contract-specific option cuts and NDF fixing times are explicitly
unconfirmed. This is a research snapshot, not executable market quotations.

The pair regression suite exercises both directions at 30/91/182/365 days,
with smile construction, finite pricing/ranking, forward-rate consistency and
fixed-loss sizing checks. No historical backtest or tilt pipeline changes are
included in this update.
