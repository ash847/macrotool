# PM preference filters — 0.2.43

- The menu no longer offers early monetisation. Saved chats retain their old preference until the user chooses a replacement and applies it; Settings labels this explicitly.
- Avoid capped upside is a hard vanilla-only filter, not just a score penalty. The linear comparator is not an eligible recommendation.
- Avoid tails requires a computed, finite, non-negative contractual maximum loss. Unknown and unbounded loss are excluded. Finite loss may exceed premium; no protective legs are added.
- This policy is checked after pricing for ranked and fallback recommendations, Trade View evaluations and custom pricing. The former premium-only construction declaration is not used as proof of a finite maximum loss.
- Direction-specific tail restrictions remain separate and can further restrict eligibility.
- Existing engine preference keys and management overlay mappings are retained for compatibility; no Supabase config changes are required.
