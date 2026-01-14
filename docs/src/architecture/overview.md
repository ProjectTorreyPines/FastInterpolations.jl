# Architecture Overview

FastInterpolations.jl is designed for **zero-allocation** interpolation on hot paths. This section explains the architecture choices that enable this performance.

## Design Goals

1. **Zero-Allocation Hot Paths**: No heap allocation after warmup
2. **O(1) Cache Lookup**: Lock-free cache hits via RCU pattern
3. **Thread Safety**: Safe for concurrent use from multiple threads
4. **Simple API**: One function call for both construction and evaluation

## Two Usage Patterns

FastInterpolations.jl supports two patterns depending on your use case:

### Pattern 1: One-Shot (Recommended for Dynamic Y)

Use when the x-grid is fixed but y-values change over time (e.g., simulation loops).

```@example arch
using FastInterpolations

x = range(0.0, 10.0, 100)
xq = [1.0, 2.0, 3.0]
out = zeros(3)

# Simulated loop - y changes each iteration
for step in 1:3
    y = sin.(x .+ step * 0.1)  # y evolves
    cubic_interp!(out, x, y, xq)  # zero-allocation (after first call)
    println("Step $step: ", round.(out, digits=4))
end
```

**Why zero-allocation?** The auto-cache stores the LU factorization keyed by the x-grid. On subsequent calls with the same grid, the cached factorization is reused.

### Pattern 2: Interpolant (Recommended for Static Data)

Use when both x and y are fixed and you need repeated evaluation.

```@example arch
using FastInterpolations

x = range(0.0, 2π, 100)
y = sin.(x)

# Create interpolant once (pre-computes coefficients)
itp = cubic_interp(x, y)

# Evaluate many times (zero-allocation)
println("itp(0.5) = ", round(itp(0.5), digits=4))
println("itp(1.0) = ", round(itp(1.0), digits=4))
println("itp(π) = ", round(itp(π), digits=4))
```

**Why zero-allocation?** The interpolant stores pre-computed spline coefficients. Evaluation only requires local arithmetic.

## Choosing the Right Pattern

!!! tip "Comprehensive API Selection Guide"
    For detailed decision-making support including the **SeriesInterpolant** pattern,
    performance trade-offs, and optimization tips, see the [API Selection Guide](../guides/api_selection.md).

## Allocation Behavior Summary

| Method | Scalar Query | Vector Query | In-Place |
|--------|--------------|--------------|----------|
| `linear_interp` | zero | output only | zero |
| `cubic_interp` | zero* | output only | zero* |
| Interpolant `itp(x)` | zero | output only | zero |

*zero after first call (auto-cached)

## Performance Characteristics

| Operation | Cost |
|-----------|------|
| Cache lookup | ~10 ns (lock-free) |
| Full interpolation (cache hit) | ~800 ns (100 points) |
| Cache miss | +LU factorization time |

## Julia Version Notes

- **Minimum**: Julia 1.7+ (for `@atomic` support)
- **Recommended**: Julia 1.12+ (improved atomic semantics, guaranteed zero allocation)
- Older versions may show minor (~16-64 bytes) allocation overhead in edge cases

## See Also

- **[Auto-Cache](caching.md)**: How the automatic caching system works
- **[Thread Safety](thread_safety.md)**: RCU pattern and concurrent access guarantees
