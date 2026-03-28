# API Selection Guide

FastInterpolations.jl offers several API styles optimized for different **data dynamics** and **usage patterns**—all fast by default, but each with distinct trade-offs to maximize performance for your specific use case.

## When to Choose Which API

- **One-shot API** (e.g., `linear_interp`):
    - **Best for:** Dynamic Data or Simplicity.
    - Use when `y` data changes frequently (e.g., inside a simulation loop).
    - No need to manage interpolator objects; just call the function.

- **Interpolant Object** (e.g., `itp = cubic_interp(x, y)`):
    - **Best for:** Static Data.
    - Use when `x` and `y` are constant, but you query at many different points over time.
    - Pre-computes coefficients once for faster reuse.

- **One-shot Series** (e.g., `linear_interp(x, Series(y1, y2), xq)`):
    - **Best for:** Dynamic data with multiple series on the same grid.
    - No interpolant construction — search once, eval per y-vector.
    - **Zero allocation** for in-place paths.

- **SeriesInterpolant** (e.g., `sitp = cubic_interp(x, Series(y1, y2))`):
    - **Best for:** Static data with repeated queries on the same series.
    - Uses unified matrix storage with **SIMD-optimized** point-contiguous layout.
    - **10-120× faster** for repeated scalar queries due to cache locality and amortized coefficient solve.

### Quick Decision Matrix

| Scenario | Y Changes? | Series Count | Recommended API |
|:---------|:----------:|:------------:|:----------------|
| Simulation loop | Yes | 1 | **One-shot** |
| Simulation loop | Yes | 2+ | **One-shot Series** |
| Static lookup | No | 1 | **Interpolant** |
| Static lookup, repeated queries | No | 2+ | **SeriesInterpolant** |
| Scalar-heavy loop | No | 2+ | **SeriesInterpolant** (10-120× faster) |


## Basic Usage (Scalar Query)

The simplest way to use the library is with the **One-shot API** for a single point.

```julia
x = 0.0:1.0:10.0   # Grid
y = sin.(x)        # Data
xq = 2.5           # Query point

# Simple scalar evaluation
val = linear_interp(x, y, xq)
```

Usage is consistent across different interpolation methods:
- `constant_interp(x, y, xq)`
- `linear_interp(x, y, xq)`
- `quadratic_interp(x, y, xq)`
- `cubic_interp(x, y, xq)`

!!! tip "Vector Queries & Performance"
    If you need to query multiple points (a vector `xq`), **do not use a loop** with scalar queries.
    FastInterpolations.jl has optimized vector APIs. See [Memory & Allocation](memory_allocation.md).

---

## See Also

- [Memory & Allocation](memory_allocation.md) - Zero-allocation patterns and optimization
- [Interpolation Overview](../interpolation/overview.md) - Methods and grid types
