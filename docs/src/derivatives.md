# Derivatives

FastInterpolations.jl provides **analytical derivatives** for all interpolation methods. No finite difference approximation needed—derivatives are computed directly from the spline coefficients.

!!! tip "Visual Comparison"
    See [Interpolation Overview](interpolation/overview.md) for side-by-side derivative plots of all 4 methods.

## Overview

| Interpolation | First Derivative | Second Derivative |
|---------------|------------------|-------------------|
| Constant | Always 0 | Always 0 |
| Linear | Piecewise constant (slope) | Always 0 |
| Quadratic | Continuous (C¹) | Piecewise constant |
| Cubic | Smooth (C¹ continuous) | Continuous (C²) |

## One-Shot API

Use the `deriv` keyword argument:

```@example deriv
using FastInterpolations

x = range(0.0, 2π, 50)
y = sin.(x)

# Value (default)
val = cubic_interp(x, y, 1.0)

# First derivative
d1 = cubic_interp(x, y, 1.0; deriv=1)

# Second derivative
d2 = cubic_interp(x, y, 1.0; deriv=2)

println("At x = 1.0:")
println("  Value:      ", round(val, digits=6), " (true: ", round(sin(1.0), digits=6), ")")
println("  1st deriv:  ", round(d1, digits=6), " (true: ", round(cos(1.0), digits=6), ")")
println("  2nd deriv:  ", round(d2, digits=6), " (true: ", round(-sin(1.0), digits=6), ")")
```

### Vector Evaluation

```@example deriv
xq = range(0.0, 2π, 100)

# Evaluate derivatives at multiple points
values = cubic_interp(x, y, xq)
first_derivs = cubic_interp(x, y, xq; deriv=1)
second_derivs = cubic_interp(x, y, xq; deriv=2)

println("First 5 first derivatives: ", round.(first_derivs[1:5], digits=4))
```

### In-Place Evaluation

```@example deriv
out = zeros(5)
xq_small = [0.5, 1.0, 1.5, 2.0, 2.5]

cubic_interp!(out, x, y, xq_small; deriv=1)
println("In-place first derivatives: ", round.(out, digits=4))
```

## Interpolant API

For repeated evaluation, use `deriv1()` and `deriv2()` wrapper functions:

```@example deriv
itp = cubic_interp(x, y)

# Create derivative views
d1 = deriv1(itp)  # First derivative
d2 = deriv2(itp)  # Second derivative

# Scalar evaluation
println("d1(1.0) = ", round(d1(1.0), digits=6))
println("d2(1.0) = ", round(d2(1.0), digits=6))
```

### Broadcasting

Derivative views support broadcasting for efficient vector evaluation:

```@example deriv
xq = range(0.0, 2π, 100)

# Broadcast over query points
first_derivs = d1.(xq)
second_derivs = d2.(xq)

println("Broadcast results (first 5): ", round.(first_derivs[1:5], digits=4))
```

### Fused Broadcast

Derivative views work seamlessly with Julia's fused broadcast:

```@example deriv
# Fused broadcast example
result = @. 2.0 * d1(xq) + d2(xq)
println("Fused broadcast (first 5): ", round.(result[1:5], digits=4))
```

## Derivatives with Boundary Conditions

Different boundary conditions affect derivative behavior at endpoints. See [Cubic Splines](interpolation/cubic.md) for details on `NaturalBC`, `ClampedBC`, and `PeriodicBC`.

## Derivatives with Extrapolation

Derivatives respect the `extrap` setting of the interpolant:

```julia
itp = cubic_interp(x, y; extrap=:extension)
d1 = deriv1(itp)
d1(-0.5)  # Uses :extension extrapolation for derivative too
```

See [Extrapolation](extrapolation.md) for available modes.

## API Summary

### One-Shot API

| Function | `deriv=1` | `deriv=2` |
|----------|-----------|-----------|
| `constant_interp(x, y, xq; deriv=...)` | 0 | 0 |
| `linear_interp(x, y, xq; deriv=...)` | Piecewise constant | 0 |
| `quadratic_interp(x, y, xq; deriv=...)` | Continuous | Piecewise constant |
| `cubic_interp(x, y, xq; deriv=...)` | Smooth | Continuous |

### Interpolant API

| Method | Description |
|--------|-------------|
| `deriv1(itp)` | First derivative view |
| `deriv2(itp)` | Second derivative view |
| `d1.(xq)` | Vector evaluation via broadcast |

## See Also

- **[Interpolation Overview](interpolation/overview.md)**: Visual comparison of all 4 methods
- **[Constant](interpolation/constant.md)** | **[Linear](interpolation/linear.md)** | **[Quadratic](interpolation/quadratic.md)** | **[Cubic](interpolation/cubic.md)**
