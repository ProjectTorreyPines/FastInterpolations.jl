# Interpolation Overview

FastInterpolations.jl provides four interpolation methods with increasing smoothness. All methods support analytical **1st and 2nd derivatives**.

| Method | Continuity | 1st Derivative | 2nd Derivative |
|--------|------------|----------------|----------------|
| Constant | C⁻¹ (discontinuous) | 0 | 0 |
| Linear | C⁰ (continuous) | Piecewise constant | 0 |
| Quadratic | C¹ (smooth) | Continuous | Piecewise constant |
| Cubic | C² (smooth) | Continuous | Continuous |

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

-----

## Visual Comparison

All examples use the same sparse, non-uniform grid interpolating `sin(x)`:

```julia
using FastInterpolations

x = [0.0, 0.9, 1.5, 2.2, 3.5, 4.5, 5.5, 2π]  # 8 non-uniform points
y = sin.(x)
xq = range(0, 2π, 500)  # query points
```

```@example overview
using FastInterpolations  # hide
using Plots  # hide
x = [0.0, 0.9, 1.5, 2.2, 3.5, 4.5, 5.5, 2π]  # hide
y = sin.(x)  # hide
xq = range(0, 2π, 500)  # hide
```

### Value: ``S(x)``

```julia
constant_interp(x, y, xq)   # step function
linear_interp(x, y, xq)     # piecewise linear
quadratic_interp(x, y, xq)  # C¹ smooth
cubic_interp(x, y, xq)      # C² smooth
```

```@example overview
p = plot(layout=(2,2), size=(800, 600), legend=:bottomleft, dpi=250)  # hide
methods = [  # hide
    ("Constant", xq -> constant_interp(x, y, xq)),  # hide
    ("Linear", xq -> linear_interp(x, y, xq)),  # hide
    ("Quadratic", xq -> quadratic_interp(x, y, xq)),  # hide
    ("Cubic", xq -> cubic_interp(x, y, xq))  # hide
]  # hide
for (i, (name, interp)) in enumerate(methods)  # hide
    yq = interp(xq)  # hide
    plot!(p[i], xq, yq, label="$name", linewidth=2)  # hide
    plot!(p[i], xq, sin.(xq), label="sin(x)", linestyle=:dash, alpha=0.7, color=:black)  # hide
    scatter!(p[i], x, y, label="data", markersize=6, color=:black)  # hide
    title!(p[i], name)  # hide
    ylims!(p[i], -1.3, 1.3)  # hide
end  # hide
p  # hide
```

### First Derivative: ``\frac{dS}{dx}``

```julia
constant_interp(x, y, xq; deriv=1)   # always 0
linear_interp(x, y, xq; deriv=1)     # piecewise constant
quadratic_interp(x, y, xq; deriv=1)  # continuous
cubic_interp(x, y, xq; deriv=1)      # smooth
```

```@example overview
p = plot(layout=(2,2), size=(800, 600), legend=:bottomleft, dpi=250)  # hide
methods_d1 = [  # hide
    ("Constant", xq -> constant_interp(x, y, xq; deriv=1)),  # hide
    ("Linear", xq -> linear_interp(x, y, xq; deriv=1)),  # hide
    ("Quadratic", xq -> quadratic_interp(x, y, xq; deriv=1)),  # hide
    ("Cubic", xq -> cubic_interp(x, y, xq; deriv=1))  # hide
]  # hide
for (i, (name, interp)) in enumerate(methods_d1)  # hide
    yq = interp(xq)  # hide
    plot!(p[i], xq, yq, label="S'(x)", linewidth=2, color=:red)  # hide
    plot!(p[i], xq, cos.(xq), label="cos(x)", linestyle=:dash, alpha=0.7, color=:black)  # hide
    scatter!(p[i], x, cos.(x), label="cos(xᵢ)", markersize=6, color=:black)  # hide
    hline!(p[i], [0], color=:gray, linestyle=:dot, label=nothing)  # hide
    title!(p[i], "$name")  # hide
    ylims!(p[i], -1.3, 1.3)  # hide
end  # hide
p  # hide
```

### Second Derivative: ``\frac{d^2S}{dx^2}``

```julia
constant_interp(x, y, xq; deriv=2)   # always 0
linear_interp(x, y, xq; deriv=2)     # always 0
quadratic_interp(x, y, xq; deriv=2)  # piecewise constant
cubic_interp(x, y, xq; deriv=2)      # continuous
```

```@example overview
p = plot(layout=(2,2), size=(800, 600), legend=:bottomleft, dpi=250)  # hide
methods_d2 = [  # hide
    ("Constant", xq -> constant_interp(x, y, xq; deriv=2)),  # hide
    ("Linear", xq -> linear_interp(x, y, xq; deriv=2)),  # hide
    ("Quadratic", xq -> quadratic_interp(x, y, xq; deriv=2)),  # hide
    ("Cubic", xq -> cubic_interp(x, y, xq; deriv=2))  # hide
]  # hide
for (i, (name, interp)) in enumerate(methods_d2)  # hide
    yq = interp(xq)  # hide
    plot!(p[i], xq, yq, label="S''(x)", linewidth=2, color=:green)  # hide
    plot!(p[i], xq, -sin.(xq), label="-sin(x)", linestyle=:dash, alpha=0.7, color=:black)  # hide
    scatter!(p[i], x, -sin.(x), label="-sin(xᵢ)", markersize=6, color=:black)  # hide
    hline!(p[i], [0], color=:gray, linestyle=:dot, label=nothing)  # hide
    title!(p[i], "$name")  # hide
    ylims!(p[i], -1.3, 1.3)  # hide
end  # hide
p  # hide
```

---

## Next Steps

- **[Constant](constant.md)**: Step interpolation with `side` modes
- **[Linear](linear.md)**: Simple and fast
- **[Quadratic](quadratic.md)**: C¹ with single-endpoint BC
- **[Cubic](cubic.md)**: C² with various boundary conditions
