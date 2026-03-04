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

## 2. `PeriodicBC()` Strict Endpoint Check

The default `:inclusive` mode now requires `y[1] == y[end]` (exact equality) instead of `isapprox`. This catches silent data errors from floating-point arithmetic:

```julia
t = range(0, 2π, 101)
y = sin.(t)
# sin(2π) ≈ -2.4e-16, NOT exactly 0.0

cubic_interp(t, y; bc=PeriodicBC())
# ERROR: y[1] != y[end] for inclusive PeriodicBC
```

Fix by explicitly ensuring the endpoint matches:

```julia
y[end] = y[1]   # force exact equality
cubic_interp(t, y; bc=PeriodicBC())  # works
```

Alternatively, use `:exclusive` mode if your data does not include the repeated endpoint:

```julia
t = range(0, 2π, 101)[1:end-1]   # exclude last point
y = sin.(t)
cubic_interp(t, y; bc=PeriodicBC(:exclusive))
```
