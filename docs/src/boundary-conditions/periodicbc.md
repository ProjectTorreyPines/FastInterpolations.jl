# PeriodicBC

True periodic boundary conditions for cubic splines, ensuring C² continuity (value, slope, and curvature) at the period boundary.

---

## PeriodicBC vs `WrapExtrap()`

These two features sound similar but solve fundamentally different problems:

| | `PeriodicBC()` | `WrapExtrap()` |
|---|---|---|
| **What it does** | Solves a cyclic tridiagonal system (Sherman-Morrison) so the spline is **C² continuous** at the period boundary | Maps out-of-domain queries back into `[x₁, xₙ]` via modular arithmetic |
| **Smoothness** | ``S, S', S''`` all match at the wrap point | No smoothness guarantee — may have jumps in value, slope, or curvature |
| **Data requirement** | `y[1] == y[end]` (inclusive) or `endpoint=:exclusive` | None |
| **Works with** | Cubic splines only | Any interpolation method |
| **Use case** | Physically periodic signals (angles, phases, Fourier-sampled data) | Quick "repeat" behavior without physical periodicity |

!!! tip "Rule of Thumb"
    If your data is truly periodic (e.g., one full period of `sin(x)`), use `PeriodicBC`. If you just want out-of-domain queries to wrap around, use `extrap=WrapExtrap()`.

---

## Endpoint Policy

`PeriodicBC` supports two conventions for defining the periodic domain.

### Inclusive Mode (Default)

The grid includes the repeated start point at the end: `y[1] == y[end]` (exact equality required). This is the standard definition in most spline libraries.

```julia
# Grid covers [0, 2π], with repeated endpoint
x = range(0, 2π, 65)  # 65 points, last point is 2π
y = cos.(x)           # y[1] == y[end] (cos(0) == cos(2π) == 1.0)

itp = cubic_interp(x, y; bc=PeriodicBC())
```

### Exclusive Mode

The grid contains only **unique** points. The theoretical "next" point would be the start of the next period. This is common in FFT-based applications or simulation grids.

!!! note "Internal Mechanism"
    Internally, the solver handles this by virtually appending the start point to the end of the data. The user does not need to allocate extra memory for this repeated point.

```julia
# Grid covers [0, 2π), no repeated endpoint
N = 64
x = range(0, step=2π/N, length=N)
y = sin.(x)

# Automatically infers period from step for ranges
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))
```

#### Determining the Period

When `endpoint=:exclusive` is used, the full period length is determined based on the grid type:

- **Uniform Grids (`AbstractRange`)**: The period is automatically inferred because the step size is known and constant.
- **Arbitrary Grids (`Vector`)**: The grid might be non-uniform, so the period cannot be safely guessed. You must provide the `period` keyword explicitly.

| Grid Type | Period Handling |
|---|---|
| `AbstractRange` | **Auto-inferred** (from step & length) |
| `Vector` | **User-specified** (via `period` keyword) |

```julia
# 1. Range (Auto-inferred)
# Input length is 10, step is 0.1 → Period = 1.0
itp = cubic_interp(0:0.1:0.9, y; bc=PeriodicBC(endpoint=:exclusive)) 

# 2. Vector (Explicit)
# Non-uniform grid requires explicit period
x = [0.1, 0.4, 0.8] 
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive, period=1.0))
```

---

## API Summary

```julia
# 1. Standard (Inclusive) - y[1] must match y[end]
PeriodicBC()

# 2. Exclusive (Auto-period) - Ranges only
PeriodicBC(endpoint=:exclusive)

# 3. Exclusive (Explicit period) - Required for Vectors
PeriodicBC(endpoint=:exclusive, period=2π)
```