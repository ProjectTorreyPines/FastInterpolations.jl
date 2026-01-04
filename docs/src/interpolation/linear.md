# Linear Interpolation

Piecewise linear interpolation connecting data points with straight line segments.

**Key Feature**: O(1) index lookup with `Range` grids for maximum performance.

| Grid Type | Lookup | Recommended For |
|-----------|--------|-----------------|
| `AbstractRange` | **O(1)** | Uniform grids (fastest) |
| `AbstractVector` | O(log n) | Non-uniform grids |

---

## Usage

```julia
using FastInterpolations

x = range(0.0, 2π, 20)   # Range grid → O(1) lookup
y = sin.(x)

# One-shot evaluation
linear_interp(x, y, 1.0)       # single point → 0.8269...
linear_interp(x, y, [1.0, 2.0]) # multiple points

# In-place evaluation (zero allocation)
xq = range(0.0, 2π, 200)
out = similar(xq)
linear_interp!(out, x, y, xq)

# Create reusable interpolant
itp = linear_interp(x, y)
itp(1.0)    # evaluate at single point
itp(xq)     # evaluate at multiple points

# Derivatives
linear_interp(x, y, 1.0; deriv=1)  # piecewise constant slope
d1 = deriv1(itp); d1(1.0)          # same via interpolant
d2 = deriv2(itp); d2(1.0)          # always 0
```

!!! tip "Performance"
    Always prefer `Range` over `Vector` when possible. Direct O(1) indexing vs O(log n) binary search.

!!! note "Visualization"
    See [Visual Comparison](comparison.md) for side-by-side plots of all methods, or [Derivatives](derivatives.md) for detailed derivative documentation.

---

## When to Use

- Speed is critical and smoothness is not required
- Data is already nearly linear between points
- Guaranteed monotonicity preservation
- Memory-constrained (no spline coefficient overhead)

**Need smooth curves?** → [Quadratic](quadratic.md) (C¹) or [Cubic](cubic.md) (C²)
