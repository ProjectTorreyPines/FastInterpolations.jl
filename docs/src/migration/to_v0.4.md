# [Migrating to v0.4](@id migration_v0_4)

v0.4.0 has two breaking changes. Both produce compile-time or runtime errors, so your test suite will catch them.

## 1. `Series()` Wrapper

Multi-series input now requires the `Series(...)` wrapper. Bare `Matrix` and `Vector{Vector}` dispatch has been removed to avoid ambiguity with the new custom value type support.

| v0.3 (removed) | v0.4 |
|:----------------|:------|
| `cubic_interp(x, [y1, y2])` | `cubic_interp(x, Series(y1, y2))` |
| `cubic_interp(x, Y_matrix)` | `cubic_interp(x, Series(Y_matrix))` |
| `linear_interp(x, [y1, y2])` | `linear_interp(x, Series(y1, y2))` |

All input forms are supported:

```julia
Series(y1, y2, y3)          # varargs
Series([y1, y2])            # vector of vectors
Series(hcat(y1, y2))        # matrix (columns = series)
```

## 2. `PeriodicBC()` Endpoint Check

**v0.4.0–v0.4.5**: The default `:inclusive` mode required `y[1] == y[end]` (strict bitwise equality). This was overly strict for computed data — e.g., `sin(0) != sin(2π)` due to floating-point round-off.

**v0.4.6+**: Relaxed to `isapprox` with `atol = 8eps(T)` noise floor. Typical computed periodic data now passes without manual fixup:

```julia
t = range(0, 2π, 101)
y = sin.(t)               # sin(0) ≈ sin(2π) — passes (diff ~1 eps)
cubic_interp(t, y; bc=PeriodicBC())  # works (would ERROR in v0.4.0–v0.4.5)
```

For scaled data where noise exceeds `8eps`, use `check=false` or set the endpoint explicitly:

```julia
y_scaled = 1e6 .* sin.(t)
cubic_interp(t, y_scaled; bc=PeriodicBC(check=false))  # skip validation
# or: y_scaled[end] = y_scaled[1]
```

Alternatively, use `:exclusive` mode if your data does not include the repeated endpoint:

```julia
t = range(0, 2π, 101)[1:end-1]   # exclude last point
y = sin.(t)
cubic_interp(t, y; bc=PeriodicBC(:exclusive))
```
