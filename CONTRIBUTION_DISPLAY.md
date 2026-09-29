# Contributions as shares of the variant score

Chat and Trade View's contributor/detractor display use each cell's weighted
percentage P&L contribution divided by that exact variant's full `score_pct`.
The denominator includes all scenario cells, not just the top/bottom rows shown.
Currency scores and sized currency P&L are not mixed into this calculation.

Shares retain their signs, can exceed 100%, and sum to 100% only across the full
scenario set. They are not scenario probabilities or returns on premium.
Original contributions as percentages of scoring notional remain available,
including when the normalized share is N/A.

`knowledge/defaults/contribution_display.json` sets the minimum positive total.
The initial cutoff is 0.0001 in fractional units (0.01%, one basis point).
Totals at or below this cutoff, including negative totals, display N/A; missing
or non-finite inputs also display N/A. Restart after editing the cached config.

The pack retains the original engine total for each ranked variant, including
the linear benchmark. This change affects presentation only, not scenario
weights, ranking, pricing or sizing. Version: 0.2.23.
