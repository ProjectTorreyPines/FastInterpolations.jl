# Auto-Cache System

Cubic spline interpolation requires solving a tridiagonal linear system, which normally allocates working memory. FastInterpolations.jl eliminates this overhead through **automatic caching**.

## Why Caching Matters

For a cubic spline with n points:

1. **Matrix Construction**: Build n×n tridiagonal matrix from x-grid spacing
2. **LU Factorization**: O(n) decomposition for efficient solving
3. **Forward/Back Substitution**: Apply factorization to y-values

Steps 1-2 depend **only on the x-grid**, not on y-values. By caching the LU factorization, we skip expensive matrix operations on repeated calls.

## How It Works

```
                     ┌─────────────────────────────────────┐
                     │       cubic_interp!(out, x, y, xq)  │
                     └─────────────────────────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │        Cache Lookup (lock-free)     │
                     │   Key: x-grid + BC type signature   │
                     └─────────────────────────────────────┘
                            │                    │
                       Cache Hit              Cache Miss
                       (~10 ns)               (acquire lock)
                            │                    │
                            ▼                    ▼
                     ┌──────────┐      ┌──────────────────┐
                     │ Reuse LU │      │ Build LU → Store │
                     │ Factor   │      │   in Cache       │
                     └──────────┘      └──────────────────┘
                            │                    │
                            └────────┬───────────┘
                                     ▼
                     ┌─────────────────────────────────────┐
                     │     Solve: L⁻¹ U⁻¹ (y + BC RHS)     │
                     │   (task-local workspace, no alloc)  │
                     └─────────────────────────────────────┘
```

## Multi-Grid Caching

The cache stores multiple x-grid entries (default: 16), enabling zero-allocation across different grids:

```@example cache
using FastInterpolations

# Two different grids (Range for both x and query points)
x1 = range(0.0, 1.0, 50)
x2 = range(0.0, 10.0, 100)
y1 = sin.(x1)
y2 = cos.(x2)
out1 = zeros(10)
out2 = zeros(10)
query1 = range(0.1, 0.9, 10)
query2 = range(1.0, 9.0, 10)

# First calls cache both grids
cubic_interp!(out1, x1, y1, query1)  # caches x1
cubic_interp!(out2, x2, y2, query2)  # caches x2

# Subsequent calls are cache hits
cubic_interp!(out1, x1, y1, query1)  # cache hit → zero-allocation
cubic_interp!(out2, x2, y2, query2)  # cache hit → zero-allocation

println("Results from grid 1: ", round.(out1[1:3], digits=4))
println("Results from grid 2: ", round.(out2[1:3], digits=4))
```

## Cache Key: Type Signature

The cache is keyed by:
1. **x-grid identity** (fast path: `objectid`)
2. **x-grid equality** (slow path: `isequal`)
3. **Boundary condition type** (not values!)

```julia
# These share the same cache entry (same BC type)
cubic_interp(x, y; bc=NaturalBC())
cubic_interp(x, y; bc=NaturalBC())  # cache hit

# Different BC type → different cache entry
cubic_interp(x, y; bc=ClampedBC())  # new cache entry
```

!!! note "BC Values vs Types"
    `BCPair(Deriv1(0.0), Deriv2(0.0))` and `BCPair(Deriv1(1.0), Deriv2(2.0))` share the same cache because they have the same type signature. The LU factorization depends only on matrix structure (x-grid + BC type), not on RHS values.

## Cache Configuration

```@example cache
using FastInterpolations

# Check current cache size
println("Current cache size: ", get_cubic_cache_size())
```

### API Functions

| Function | Description |
|----------|-------------|
| `get_cubic_cache_size()` | Returns current maximum cache size |
| `set_cubic_cache_size!(n)` | Sets cache size (requires Julia restart) |
| `clear_cubic_cache!()` | Clears all cached entries |

### Adjusting Cache Size

```julia
# Increase for many distinct grids
set_cubic_cache_size!(32)

# Reduce for memory-constrained environments
set_cubic_cache_size!(4)

# Restart Julia to apply changes
```

!!! warning "Restart Required"
    Cache size is a **load-time constant** for thread-safety and compiler optimization. Changes require a Julia restart to take effect.

## Eviction Policy

When the cache is full, entries are evicted using a **ring buffer (FIFO)** policy:

```
Cache slots: [1] [2] [3] [4] ... [16]
                              ↑
                         next eviction
```

This provides:
- O(1) insertion time
- Predictable eviction behavior
- No LRU overhead in multi-threaded context

## Pre-Built Cache

For maximum control, you can pre-build a cache explicitly:

```@example cache
using FastInterpolations

x = range(0.0, 2π, 51)

# Pre-build cache with specific BC
cache = CubicSplineCache(x; bc=PeriodicBC())

# Use the same cache for multiple y-values
y1 = sin.(x)
y2 = cos.(x)
y3 = sin.(2 .* x)

xq = range(0.0, 2π, 100)
result1 = cubic_interp(cache, y1, xq)
result2 = cubic_interp(cache, y2, xq)
result3 = cubic_interp(cache, y3, xq)

println("sin interpolated: ", round.(result1[1:5], digits=4))
```

This is useful when:
- You want to bypass the auto-cache entirely
- You need to share a cache explicitly across calls
- You want to control cache lifetime precisely

## See Also

- **[Architecture Overview](overview.md)**: When to use one-shot vs interpolant
- **[Thread Safety](thread_safety.md)**: How caching works in multi-threaded code
