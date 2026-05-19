# Snapshot fixtures

Each `.json` file is a v1 baseline snapshot consumed by `baseline.load_snapshot()`.

## Schema

```json
{
  "schema_version": 1,
  "pair": "USDBRL",
  "forward": 5.00,
  "tenor_years": 0.25,
  "source": "synthetic_lognormal_v1 | smile_implied_v1 | ...",
  "bins": [<float>, ...],
  "probs": [<float>, ...]
}
```

`bins` are strictly increasing price levels (bin centres).
`probs` are non-negative, sum to 1 within float tolerance, and align 1-to-1 with `bins`.
`source` documents where the distribution came from — synthetic, smile-implied, etc.

## Adding a real smile-implied snapshot

To plug in MacroTool's actual smile distribution: from the main app, export
the result of `compute_smile_distribution(...)` to the schema above (write
the bins and probs arrays directly, set `source` to e.g. `"smile_implied_v1"`).
The Kelly v2 loader doesn't care how the distribution was generated — only
that it conforms to the schema.

## Files

- `synthetic_usdbrl_3m.json` — synthetic lognormal stand-in for USDBRL 3M,
  forward 5.00, sigma 0.10 (annualised), 200 bins. Used for development and
  smoke testing; **not** a real market baseline.
