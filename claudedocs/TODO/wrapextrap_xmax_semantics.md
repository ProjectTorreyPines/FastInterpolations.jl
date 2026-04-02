# WrapExtrap: x_max should return y[end], not wrap to x_min

## Problem
`_wrap_to_domain` uses half-open interval `[x_min, x_max)`:
```julia
if xi < x_min || xi >= x_max   # ← x_max treated as outside domain
    return x_min + mod(xi - x_min, period)
end
```

This means `f(x_max)` returns `f(x_min)` instead of `y[end]`. For WrapExtrap users,
querying at `x[end]` (a grid knot) gives the left-boundary value — surprising behavior.

## Proposed Fix
Change `>=` to `>` in `_wrap_to_domain`:
```julia
if xi < x_min || xi > x_max    # ← x_max now in-domain
```

Also update the vector fast-path condition to `<=`:
```julia
if qmin >= x_min && qmax <= x_max   # ← consistent with new semantics
```

## Impact Analysis
- **WrapExtrap**: `f(x_max) → y[end]` instead of `y[1]`. More intuitive.
- **PeriodicBC**: `y[end] == y[1]` is enforced by BC, so value is identical.
  But search/interval selection at the boundary may change — need to verify
  that `_search_binary` handles `xq == x[end]` correctly (returns last interval).
- **Scalar path**: Currently calls `_wrap_to_domain` per-element, so the fix
  applies to both scalar and vector paths uniformly.

## Files Affected
- `src/core/periodic.jl:24` — primary `_wrap_to_domain` (Tg overload)
- `src/core/periodic.jl:38` — generic `_wrap_to_domain` (Real overload)
- `src/linear/linear_oneshot.jl:110` — vector fast-path `<` → `<=`
- `src/cubic/cubic_eval.jl:256` — vector fast-path `<` → `<=`
- `src/hermite/cubic_hermite_eval.jl:176,223` — vector fast-path `<` → `<=`

## Testing
- Verify `f(x_max; extrap=WrapExtrap()) ≈ y[end]` for all 1D methods
- Verify PeriodicBC full-suite regression (test_periodic_bc.jl, test_periodic_exclusive.jl)
- Verify scalar == vector consistency at x_max boundary
