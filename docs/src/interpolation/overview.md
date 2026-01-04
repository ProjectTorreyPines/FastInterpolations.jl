# Interpolation Overview

FastInterpolations.jl provides four interpolation methods with increasing smoothness. All methods support analytical **1st and 2nd derivatives**.

| Method | Continuity | 1st Derivative | 2nd Derivative |
|--------|------------|----------------|----------------|
| Constant | C⁻¹ (discontinuous) | 0 | 0 |
| Linear | C⁰ (continuous) | Piecewise constant | 0 |
| Quadratic | C¹ (smooth) | Continuous | Piecewise constant |
| Cubic | C² (smooth) | Continuous | Continuous |

---

## Grid Types

All interpolation methods support both uniform and non-uniform grids:

| Grid Type | Index Lookup | Recommended For |
|-----------|--------------|-----------------|
| `AbstractRange` | **O(1)** direct | Uniform grids (fastest) |
| `AbstractVector` | O(log n) binary search | Non-uniform grids |

!!! tip "Performance"
    Always prefer `Range` over `Vector` when your grid is uniform. Direct O(1) indexing vs O(log n) binary search makes a significant difference in tight loops.

```julia
# Uniform grid → O(1) lookup
x_uniform = range(0.0, 10.0, 100)

# Non-uniform grid → O(log n) lookup
x_nonuniform = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
```

---

## API Styles

`FastInterpolations.jl` provides two API styles designed for maximum performance through strict **type stability** and **compile-time dispatch**.

### 1. One-shot API (Recommended)
**Best when `y` values change frequently but the grid `x` remains fixed** — the same x-grid is reused but y-values change over time.

```julia
# x, y: known data points (target)
# xq: query points → yq: interpolated values
yq = constant_interp(x, y, xq)
yq = linear_interp(x, y, xq)
yq = quadratic_interp(x, y, xq)
yq = cubic_interp(x, y, xq)
```

**Example**: Simulation where y evolves each timestep
```julia
x = range(0.0, 10.0, 100)
out = zeros(N_query)

for step in 1:1000
    y = compute_new_values(step)  # y changes every iteration
    cubic_interp!(out, x, y, xq)  # zero-allocation ✅
end
```
!!! tip "Zero-Allocation"
    After a single **warm-up** call, the One-shot API is **guaranteed zero-allocation** for repeated calls on the same grid—perfect for high-performance simulations.

### 2. Interpolant API
**Best when both `x` and `y` are fixed** — pre-computes coefficients once for fast repeated evaluation.

```julia
itp = cubic_interp(x, y)   # construct once
itp(xq)                    # evaluate many times
```

**Example**: Lookup table with fixed data
```julia
x = range(0.0, 10.0, 100)
y = sin.(x)
itp = cubic_interp(x, y)  # pre-compute once

for query in queries
    result = itp(query)   # zero-allocation ✅
end
```

---

## Next Steps

- **[Constant](constant.md)**: Step interpolation with `side` modes
- **[Linear](linear.md)**: Simple and fast
- **[Quadratic](quadratic.md)**: C¹ with single-endpoint BC
- **[Cubic](cubic.md)**: C² with various boundary conditions
- **[Visual Comparison](comparison.md)**: Side-by-side plots of all methods
- **[Derivatives](derivatives.md)**: Detailed derivative documentation
