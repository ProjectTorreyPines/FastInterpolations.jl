# Cubic Spline Interpolation

C²-continuous spline interpolation with smooth first and second derivatives.

---

## Two Fundamental BC Categories

Cubic splines require boundary conditions at **both** endpoints. There are two fundamentally different approaches:

| Category | Algorithm | Mathematical Meaning |
|----------|-----------|----------------------|
| **BCPair** | Standard tridiagonal | Independent constraints at each endpoint |
| **PeriodicBC** | Sherman-Morrison cyclic | True periodic: ``S(x) = S(x+\tau)`` with C² continuity |

👉 See [Boundary Conditions Overview](../boundary-conditions/overview.md) for the complete BC type hierarchy and detailed explanations.

### 1. BCPair: Independent Endpoint Constraints

Each endpoint has its own `PointBC` constraint — either first or second derivative:

```julia
BCPair(left::PointBC, right::PointBC)

# PointBC types:
Deriv1(v)   # S'(endpoint) = v   (slope)
Deriv2(v)   # S''(endpoint) = v  (curvature)
```

**Examples**:
```julia
BCPair(Deriv2(0), Deriv2(0))      # ZeroCurv: zero curvature at both ends
BCPair(Deriv1(0), Deriv1(0))      # ZeroSlope: zero slope at both ends
BCPair(Deriv1(1.0), Deriv2(0))    # Mixed: slope=1 at left, curvature=0 at right
BCPair(Deriv2(-2.0), Deriv1(0.5)) # Mixed: curvature=-2 at left, slope=0.5 at right
```

**Convenience shortcuts**:

| Shortcut | Equivalent | Meaning |
|----------|------------|---------|
| `ZeroCurvBC()` | `BCPair(Deriv2(0), Deriv2(0))` | S''=0 at both ends |
| `ZeroSlopeBC()` | `BCPair(Deriv1(0), Deriv1(0))` | S'=0 at both ends (flat) |

### 2. PeriodicBC: True Periodic C² Continuity

The spline satisfies periodicity with period ``\tau = x_n - x_0``:

```math
S(x) = S(x + \tau), \quad S'(x) = S'(x + \tau), \quad S''(x) = S''(x + \tau)
```

```julia
# Inclusive endpoint (default): y[1] ≈ y[end] required
PeriodicBC()

# Exclusive endpoint: no redundant last point (e.g., FFT grids)
PeriodicBC(endpoint=:exclusive)              # Range grid → period auto-inferred
PeriodicBC(endpoint=:exclusive, period=2π)   # any grid → explicit period
```

!!! warning "Periodicity Requirement"
    For inclusive endpoints, your data must satisfy `y[1] ≈ y[end]`.
    For exclusive endpoints, the data is extended internally — no manual duplication needed.

!!! note "Different Algorithm"
    PeriodicBC uses the **Sherman-Morrison formula** to solve a cyclic tridiagonal system.
    This is fundamentally different from BCPair's standard tridiagonal solver.

👉 See [PeriodicBC Details](../boundary-conditions/periodicbc.md) for endpoint conventions, period inference, and comparison with `WrapExtrap()`.

---

## Usage

```julia
using FastInterpolations

x = range(0.0, 2π, 15)
y = cos.(x)

# One-shot evaluation (default: CubicFit)
cubic_interp(x, y, 1.0)
cubic_interp(x, y, 1.0; bc=ZeroCurvBC())             # zero curvature at endpoints
cubic_interp(x, y, 1.0; bc=ZeroSlopeBC())            # flat endpoints
cubic_interp(x, y, 1.0; bc=BCPair(Deriv1(1), Deriv2(0)))  # custom

# Periodic (closed curve) - requires y[1] ≈ y[end]
cubic_interp(x, y, 1.0; bc=PeriodicBC())

# In-place evaluation (zero allocation)
xq = range(x[1], x[end], 200)
out = similar(xq)
cubic_interp!(out, x, y, xq)

# Create reusable interpolant
itp = cubic_interp(x, y)              # CubicFit (default)
itp(1.0)    # evaluate at single point
itp(xq)     # evaluate at multiple points

# Derivatives
cubic_interp(x, y, 1.0; deriv=DerivOp(1))  # continuous first derivative
d1 = deriv1(itp); d1(1.0)         # same via interpolant
d2 = deriv2(itp); d2(1.0)         # continuous second derivative
```

---

## When to Use Each BC

| Situation | Recommended BC |
|-----------|----------------|
| General data, unknown endpoint behavior | `CubicFit()` (default) |
| Endpoints should be flat (zero slope) | `ZeroSlopeBC()` |
| Known endpoint derivatives (physics) | `BCPair(Deriv1(...), Deriv1(...))` |
| Cyclic data (angles, phases, time-of-day) | `PeriodicBC()` |

---

## Visual Comparison: ZeroCurvBC vs PeriodicBC

Comparing `cos(x)` interpolation — note that `cos''(x) = -cos(x) ≠ 0` at endpoints:

```@example cubic
using FastInterpolations
using Plots # hide

# Uniform grid (9 points)
x = range(0, 2π, 9)
y = cos.(x)
xq = range(0, 2π, 500)

itp_zerocurv = cubic_interp(x, y; bc=ZeroCurvBC())
itp_periodic = cubic_interp(x, y; bc=PeriodicBC())

# Compare: S(x), S'(x), S''(x) for both BCs
d1_nat, d2_nat = deriv1(itp_zerocurv), deriv2(itp_zerocurv)
d1_per, d2_per = deriv1(itp_periodic), deriv2(itp_periodic)

p = plot(layout=(3, 2), size=(900, 700), legend=:topright) # hide
plot!(p[1], xq, itp_zerocurv.(xq), label="ZeroCurvBC", linewidth=2) # hide
plot!(p[1], xq, cos.(xq), label="cos(x)", linestyle=:dash, color=:black, alpha=0.7) # hide
scatter!(p[1], x, y, label="data", markersize=5, color=:black) # hide
title!(p[1], "ZeroCurvBC: S(x)") # hide
ylims!(p[1], -1.3, 1.3) # hide
plot!(p[2], xq, itp_periodic.(xq), label="PeriodicBC", linewidth=2, color=:red) # hide
plot!(p[2], xq, cos.(xq), label="cos(x)", linestyle=:dash, color=:black, alpha=0.7) # hide
scatter!(p[2], x, y, label="data", markersize=5, color=:black) # hide
title!(p[2], "PeriodicBC: S(x)") # hide
ylims!(p[2], -1.3, 1.3) # hide
plot!(p[3], xq, d1_nat.(xq), label="S'(x)", linewidth=2) # hide
plot!(p[3], xq, -sin.(xq), label="-sin(x)", linestyle=:dash, linewidth=2, color=:black, alpha=0.7) # hide
scatter!(p[3], x, -sin.(x), label=nothing, markersize=5, color=:black) # hide
title!(p[3], "ZeroCurvBC: S'(x)") # hide
ylims!(p[3], -1.3, 1.3) # hide
plot!(p[4], xq, d1_per.(xq), label="S'(x)", linewidth=2, color=:red) # hide
plot!(p[4], xq, -sin.(xq), label="-sin(x)", linestyle=:dash, linewidth=2, color=:black, alpha=0.7) # hide
scatter!(p[4], x, -sin.(x), label=nothing, markersize=5, color=:black) # hide
title!(p[4], "PeriodicBC: S'(x)") # hide
ylims!(p[4], -1.3, 1.3) # hide
plot!(p[5], xq, d2_nat.(xq), label="S''(x)", linewidth=2) # hide
plot!(p[5], xq, -cos.(xq), label="-cos(x)", linestyle=:dash, linewidth=2, color=:black, alpha=0.7) # hide
scatter!(p[5], x, -cos.(x), label=nothing, markersize=5, color=:black) # hide
hline!(p[5], [0], color=:gray, linestyle=:dot, label=nothing) # hide
title!(p[5], "ZeroCurvBC: S''(x) — forced 0 at ends") # hide
ylims!(p[5], -1.5, 1.5) # hide
plot!(p[6], xq, d2_per.(xq), label="S''(x)", linewidth=2, color=:red) # hide
plot!(p[6], xq, -cos.(xq), label="-cos(x)", linestyle=:dash, linewidth=2, color=:black, alpha=0.7) # hide
scatter!(p[6], x, -cos.(x), label=nothing, markersize=5, color=:black) # hide
hline!(p[6], [0], color=:gray, linestyle=:dot, label=nothing) # hide
title!(p[6], "PeriodicBC: S''(x) — matches at wrap") # hide
ylims!(p[6], -1.5, 1.5) # hide
p # hide
```

!!! tip "Key Observation"
    - **ZeroCurvBC** forces `S''(0) = S''(2π) = 0`, but true `cos''(x) = -cos(x) = -1` at endpoints → mismatch
    - **PeriodicBC** allows `S''(0) = S''(2π)` to match naturally through cyclic continuity
