# PeriodicBC

True periodic boundary conditions for cubic splines with C² continuity at the wrap point.

---

## PeriodicBC vs `extrap=:wrap`

These two features sound similar but solve fundamentally different problems:

| | `PeriodicBC()` | `extrap=:wrap` |
|---|---|---|
| **What it does** | Solves a cyclic tridiagonal system (Sherman-Morrison) so the spline is **C² continuous** at the period boundary | Maps out-of-domain queries back into `[x₁, xₙ]` via modular arithmetic |
| **Smoothness** | ``S, S', S''`` all match at the wrap point | No smoothness guarantee — may have jumps in value, slope, or curvature |
| **Data requirement** | `y[1] ≈ y[end]` (inclusive) or `endpoint=:exclusive` | None |
| **Works with** | Cubic splines only | Any interpolation method |
| **Use case** | Physically periodic signals (angles, phases, Fourier-sampled data) | Quick "repeat" behavior without physical periodicity |

!!! tip "Rule of Thumb"
    If your data is truly periodic (e.g., one full period of `sin(x)`), use `PeriodicBC`. If you just want out-of-domain queries to wrap around, use `extrap=:wrap`.

---

## Endpoint Conventions

### Inclusive (default)

The classic convention: the last data point **repeats** the first value.

```julia
# Grid covers [0, 2π] with y[1] = y[end]
x = range(0, 2π, 65)    # 65 points, last point = 2π
y = sin.(x)              # sin(0) ≈ sin(2π)

itp = cubic_interp(x, y; bc=PeriodicBC())
```

The period is ``\tau = x_{\text{end}} - x_1``. The redundant last point is used by the solver but carries no new information.

### Exclusive (new)

The last data point is the **last unique** sample — no redundant endpoint. Common in FFT grids, uniform angular discretizations, and simulation output.

```julia
# Grid covers [0, 2π) — no repeated endpoint
N = 64
x = range(0, step=2π/N, length=N)   # 64 points, last point = 2π - Δx
y = sin.(x)

itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))
```

Internally, the data is extended to inclusive form before entering the solver — the algorithm and evaluation are identical. The exclusive convention simply saves the user from doing this manually.

---

## Period Resolution

For exclusive endpoints, the period is determined as follows:

| Grid type | `period` omitted | `period` provided |
|-----------|-----------------|-------------------|
| `AbstractRange` | Auto-inferred: `step(x) * length(x)` | Cross-validated against inference; error if mismatch |
| `Vector` | **Error** — period cannot be inferred | Used directly |

```julia
# Range grid: period auto-inferred from step(x) * length(x)
x = range(0, step=0.1, length=10)  # period = 0.1 * 10 = 1.0
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))

# Range grid: explicit period (must match inference)
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive, period=1.0))  # OK
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive, period=2.0))  # ERROR

# Vector grid: period required
x = [0.0, 0.3, 0.7, 1.5, 3.0, 5.0]
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive, period=2π))
```

---

## API Summary

```julia
# Inclusive (backward compatible)
PeriodicBC()

# Exclusive with auto-inferred period (Range grids only)
PeriodicBC(endpoint=:exclusive)

# Exclusive with explicit period (any grid)
PeriodicBC(endpoint=:exclusive, period=2π)
```

Works with all cubic spline entry points:

```julia
# 2-arg form (reusable interpolant)
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))
itp(1.0)           # evaluate
itp(1.0; deriv=1)  # first derivative

# 3/4-arg oneshot
cubic_interp(x, y, xq; bc=PeriodicBC(endpoint=:exclusive))
cubic_interp!(out, x, y, xq; bc=PeriodicBC(endpoint=:exclusive))

# Series interpolant
mitp = cubic_interp(x, [y1, y2]; bc=PeriodicBC(endpoint=:exclusive))
```

---

## Example: Inclusive vs Exclusive Equivalence

The two conventions produce identical interpolants when given the same underlying function:

```@example periodicbc
using FastInterpolations

N = 64
dx = 2π / N
f(x) = sin(x)

# Inclusive: N+1 points on [0, 2π]
x_incl = range(0, step=dx, length=N+1)
itp_incl = cubic_interp(x_incl, f.(x_incl); bc=PeriodicBC())

# Exclusive: N points on [0, 2π)
x_excl = range(0, step=dx, length=N)
itp_excl = cubic_interp(x_excl, f.(x_excl); bc=PeriodicBC(endpoint=:exclusive))

# They agree everywhere
xq = [0.1, 1.0, π, 5.5]
for x in xq
    println("x=$x:  inclusive=$(round(itp_incl(x), digits=10)),  exclusive=$(round(itp_excl(x), digits=10))")
end
```

---

## See Also

- [Boundary Conditions Overview](overview.md) — Full BC type hierarchy
- [Cubic Interpolation](../interpolation/cubic.md) — PeriodicBC visual comparison with NaturalBC
- [Extrapolation](../extrapolation.md) — `extrap=:wrap` coordinate mapping (no smoothness)
