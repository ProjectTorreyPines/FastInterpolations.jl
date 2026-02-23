# Derivatives

FastInterpolations.jl provides **analytical derivatives** computed directly from spline coefficients—no finite differences.

!!! tip "Visual Comparison"
    See [Visual Comparison](comparison.md) for side-by-side derivative plots.

## Overview

| Interpolation | 1st Derivative | 2nd Derivative | 3rd Derivative |
|---------------|----------------|----------------|----------------|
| Constant | 0 | 0 | 0 |
| Linear | Piecewise constant | 0 | 0 |
| Quadratic | Continuous (C¹) | Piecewise constant | 0 |
| Cubic | Smooth (C¹) | Continuous (C²) | Piecewise constant |

## `DerivOp{N}` — Derivative Order Type

Derivative orders are specified via `DerivOp{N}`, a parametric singleton type that encodes the derivative order as a **type parameter**. This enables compile-time specialization — the Julia compiler generates a dedicated kernel for each derivative order with zero runtime overhead.

```julia
DerivOp(0)          # value evaluation (alias: EvalValue())
DerivOp(1)          # 1st derivative   (alias: EvalDeriv1())
DerivOp(2)          # 2nd derivative   (alias: EvalDeriv2())
DerivOp(3)          # 3rd derivative   (alias: EvalDeriv3())
```

!!! note "Backward-compatible aliases"
    `EvalValue`, `EvalDeriv1`, `EvalDeriv2`, `EvalDeriv3` are const aliases for `DerivOp{0}` through `DerivOp{3}`.

## Usage

### One-Shot API

```@example deriv
using FastInterpolations

x = range(0.0, 2π, 50)
y = sin.(x)

cubic_interp(x, y, 1.0)                    # value at x=1.0
cubic_interp(x, y, 1.0; deriv=DerivOp(1))  # 1st derivative at x=1.0
cubic_interp(x, y, 1.0; deriv=DerivOp(2))  # 2nd derivative at x=1.0
nothing # hide
```

### Interpolant API

```@example deriv
itp = cubic_interp(x, y)

itp(1.0; deriv=DerivOp(1))  # 1st derivative at x=1.0
itp(1.0; deriv=DerivOp(2))  # 2nd derivative at x=1.0
nothing # hide
```

## DerivativeView

`deriv1(itp)`, `deriv2(itp)`, `deriv3(itp)` create lightweight **callable wrappers** with the derivative order fixed at construction. Calling `d1(x)` is equivalent to `itp(x; deriv=DerivOp(1))`. Same performance, cleaner syntax.

!!! note "All interpolants support all derivative orders"
    `deriv1`, `deriv2`, `deriv3` work with **all** interpolant types (constant, linear, quadratic, cubic). Higher-order derivatives simply return 0 for lower-order methods.

```@example deriv
# Create derivative views (callable objects, not values)
d1 = deriv1(itp)
d2 = deriv2(itp)
d3 = deriv3(itp)

d1(1.0)  # 1st derivative at x=1.0 (same as itp(1.0; deriv=DerivOp(1)))
d2(1.0)  # 2nd derivative at x=1.0
d3(1.0)  # 3rd derivative at x=1.0
nothing # hide
```

### Keyword Forwarding

DerivativeView forwards all keyword arguments to the parent interpolant (`deriv` excepted—it's fixed at construction):

```julia
d1 = deriv1(itp)

# Override search policy
d1(0.5; search=LinearBinary())

# Use hint for sequential access
hint = Ref(1)
for xq in sorted_queries
    val = d1(xq; hint=hint)
end
```

### Broadcasting & Fused Operations

```@example deriv
xq = range(0.0, 2π, 100)

slopes = d1.(xq)                    # broadcast over query points
result = @. itp(xq) + 0.1 * d1(xq)  # fused broadcast (zero allocations)
nothing # hide
```

### In-Place Evaluation

```@example deriv
output = zeros(5)
xq_small = [0.5, 1.0, 1.5, 2.0, 2.5]

d1(output, xq_small)  # in-place: writes 1st derivatives into output
output
```

### Higher-Order Functions

DerivativeView works with any function expecting a callable:

```julia
using Roots
d1 = deriv1(itp)
root = find_zero(d1, 0.5)  # Find where derivative = 0
```

## Boundary Conditions & Extrapolation

- **Boundary conditions** affect derivative behavior at endpoints. See [Cubic Splines](cubic.md).
- **Extrapolation** settings are inherited by derivatives:

```julia
itp = cubic_interp(x, y; extrap=ExtendExtrap())
d1 = deriv1(itp)
d1(-0.5)  # Uses ExtendExtrap extrapolation
```

## API Summary

| Method | Description |
|--------|-------------|
| `itp(xq; deriv=DerivOp(N))` | Direct derivative evaluation |
| `deriv1(itp)`, `deriv2(itp)`, `deriv3(itp)` | Convenience wrappers (all interpolants) |
| `d.(xq)` | Broadcast evaluation |
| `d(output, xq)` | In-place vector evaluation |
| `d(xq; search=..., hint=...)` | With keyword arguments |

## API Reference

```@docs
deriv1
deriv2
deriv3
deriv_view
AbstractDerivativeView
DerivativeView
```

## See Also

- **[Visual Comparison](comparison.md)**: Side-by-side derivative plots
- **[Constant](constant.md)** | **[Linear](linear.md)** | **[Quadratic](quadratic.md)** | **[Cubic](cubic.md)**
