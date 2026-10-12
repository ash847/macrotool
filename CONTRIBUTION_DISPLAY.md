# Shares of total absolute contribution

Chat and Trade View's contributor/detractor display use each cell's weighted
percentage P&L contribution divided by that variant's sum of absolute contributions.
The denominator includes all scenario cells, not just the top/bottom rows shown.
Currency scores and sized currency P&L are not mixed into this calculation.

Shares retain their signs and lie between -100% and +100%. Their absolute values
sum to 100% across the full scenario set, subject to rounding. Contributions of
+6 and -5 yield +54.5% and -45.5%, not +600% and -500%. Shares measure relative
influence, not trade quality, shares of net profit, probabilities or returns on premium.
Original contributions as percentages of trade notional remain available,
including when the normalized share is N/A.

`knowledge/defaults/contribution_display.json` sets `minimum_absolute_total_pct`.
The initial cutoff is 0.0001 in fractional units (0.01%, one basis point).
Absolute totals at or below this cutoff display N/A; missing
or non-finite inputs also display N/A. Restart after editing the cached config.

Zero or negative net scores remain valid when the absolute total exceeds the cutoff.
There is no separate scoring notional. Saved historical replies are not rewritten.

The pack retains the full absolute contribution total for each ranked variant, including
the linear benchmark. This change affects presentation only, not scenario
weights, ranking, pricing or sizing. Version: 0.2.24.
