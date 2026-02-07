# Advanced Usage

Once you're comfortable with basic interpolation, these features help optimize for specific use cases.

## Grid Selection: Range vs Vector

The most impactful optimization is choosing the right grid type:

| Grid Type | Lookup | When to Use |
|:----------|:-------|:------------|
| `Range` (`0.0:0.1:10.0`) | **O(1)** always | Uniform spacing |
| `Vector` (`[0.0, 0.2, 0.5, ...]`) | O(log n) | Non-uniform spacing |

```julia
# ✅ Uniform grid: use Range (O(1) lookup)
x = 0.0:0.01:10.0

# ✅ Non-uniform grid: must use Vector
x = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 5.0, 10.0]

# ❌ Avoid: collect() on uniform grid (degrades O(1) → O(log n))
x = collect(0.0:0.01:10.0)
```

For **Vector grids**, see [**Search & Hints**](@ref search_hints) section help optimize lookup performance.

---

## Features

| Feature | Use Case | Description |
|:--------|:---------|:------------|
| [**Search & Hints**](@ref search_hints) | Sequential or streaming queries | O(1) amortized lookup with hints |
| [**Series Interpolant**](@ref series_interpolant) | Multiple y-series on shared x-grid | 10-120× faster than separate interpolants |
| [**Memory & Allocation**](@ref memory_allocation) | Tight loops, real-time systems | Zero-allocation patterns |
| [**Optimization (Optim.jl)**](@ref optimization_guide) | Minimization over interpolated surfaces | Analytical `gradient!`/`hessian!` for Optim.jl |

## Quick Decision Guide

**Which feature do I need?**

- **"I query points in sequential order (ODE solvers, streaming)"** → [Search & Hints](@ref search_hints)
- **"I have multiple outputs for the same input grid"** → [Series Interpolant](@ref series_interpolant)
- **"I need zero allocations in my hot loop"** → [Memory & Allocation](@ref memory_allocation)
- **"I want to minimize/optimize over an interpolated surface"** → [Optimization (Optim.jl)](@ref optimization_guide)

These features can be combined. For example, a Series Interpolant can use `LinearBinary` search policy with external hints for maximum performance in ODE integrator callbacks.
