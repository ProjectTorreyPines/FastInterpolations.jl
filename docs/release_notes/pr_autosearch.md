## Summary

Introduce `AutoSearch` — an adaptive default search policy that resolves to `Binary()` for scalar queries and `LinearBinary()` for vector queries at call time. Also optimizes `LinearBinary` internals (branchless binary, window default 2→ optimal sweet spot) and ships comprehensive tests and documentation.

## Motivation

Previously, `Binary()` was the unconditional default. This meant vector queries over sorted data — the most common batch use case — silently used O(log n) binary search instead of the much faster O(1)-amortized `LinearBinary`. Users had to opt in manually, and many didn't know to.

The new `AutoSearch` default removes this decision entirely for 95% of users:

| Query type | Old behavior | New behavior | Effect |
|:-----------|:-------------|:-------------|:-------|
| Scalar (e.g., `itp(0.5)`) | `Binary()` (was default) | `Binary()` via AutoSearch | Unchanged |
| Vector (e.g., `itp(Vector)`) | `Binary()` (was default) | `LinearBinary()` via AutoSearch | Up to ~5× faster on sorted data |

## Key Changes

### `AutoSearch` (new type)

```julia
itp = linear_interp(x, y)     # stores AutoSearch — the new default
itp(0.5)                       # → Binary() (scalar)
itp([0.1, 0.5, 0.9])           # → LinearBinary() (vector)
itp(0.5; search=Binary())      # explicit override still works
```

Resolution happens once per call, at the eval entry point, with negligible overhead.

### `LinearBinary` optimization

- **Branchless binary core**: replaced `while` loop with `for _ in 1:iters` where `iters = 64 - leading_zeros(...)`. Precomputed trip count → constant-iteration loop + `ifelse` → ARM64 `csel` (branch-free). ~25–55% faster than the old loop on random queries.
- **Default window changed**: `LinearBinary()` now defaults to `linear_window=2` (was 8). Window=2 minimizes overhead for mixed/unknown patterns while still exploiting locality for sorted sequences.

### Safety fix

`_search_linear_binary!` now clamps the hint before first use (`ix = clamp(ix, 1, n-1)`), guarding against user-provided out-of-range hints (`Ref(0)`, stale hints from a different grid).

## New Export

```julia
export AutoSearch
```

## Impact

- **Zero breaking changes** for existing code: explicit `search=Binary()` or `search=LinearBinary()` at constructor or call site is honored as-is.
- **Behavior change** for users who relied on the default being `Binary()` for vector queries. Vector queries now use `LinearBinary()` via AutoSearch. For **random** vector queries, this is ~2.5–3× slower than `Binary()` — use `search=Binary()` explicitly to restore the old behavior.
- All 1D, ND, series, and integration paths updated. 440+ new test assertions.
