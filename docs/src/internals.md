# Internals & Design Documents

This section provides links to internal design documents for developers.

## Design Documents

The following documents are available directly in the GitHub repository:

- [Derivative Design](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master/docs/design/derivative_design.md): Mathematical foundations and kernel implementation for derivative computation
- [Interpolant Derivative API](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master/docs/design/interpolant_derivative_api.md): Hybrid B+C API design decisions

## Architecture Overview

FastInterpolations.jl internal architecture:

- **Operation Types** (`src/core/eval_ops.jl`): `DerivOp{N}` parametric singleton for compile-time derivative dispatch (aliases: `EvalValue`, `EvalDeriv1`, `EvalDeriv2`, `EvalDeriv3`)
- **Kernel Functions** (`src/*_kernels.jl`): Pure math functions for interpolation and derivatives
- **Boundary Conditions** (`src/bc_types.jl`): `ZeroCurvBC`, `ZeroSlopeBC`, `PeriodicBC` types

## Anchored Queries (Internal API)

For maximum performance in hot loops with **fixed query points**, anchored queries pre-compute grid positions to skip O(log n) binary search entirely.

!!! warning "Internal API"
    Functions prefixed with `_` are internal and may change without notice.

### How It Works

Anchored queries pre-compute:
1. Which interval each query point falls into
2. The local coordinate within that interval

This eliminates O(log n) binary search on every evaluation (~2x speedup).

### Usage Pattern

```julia
x = range(0.0, 10.0, 100)
xq = range(0.0, 10.0, 500)
out = similar(xq)

# Pre-compute anchors ONCE
aq_vec = FastInterpolations._anchor_query(x, xq)

# Use in hot loop
for i in 1:10000
    cubic_interp!(out, x, y, aq_vec)
end
```

### With SeriesInterpolant

SeriesInterpolant uses a **unified matrix storage** with point-contiguous layout, enabling SIMD-optimized scalar queries (10-120× faster than looping over individual interpolants).

```julia
sitp = cubic_interp(x, Series(y1, y2, y3))
outputs = [similar(xq) for _ in 1:3]

# Pre-compute anchors
aq_vec = FastInterpolations._anchor_query(x, xq, Val(:cubic))

# Fastest possible path - scalar queries especially benefit from SIMD
for i in 1:10000
    sitp(outputs, aq_vec)
end
```

!!! note "Small Series Caveat"
    For very small series counts (n ≤ 2-4) with **vector queries only**, the anchor allocation overhead may make a manual loop marginally faster (~10-25%). For scalar queries or n ≥ 4, SeriesInterpolant always wins.

## Advanced Optimization

### Cache Optimization

The one-shot API uses auto-caching to avoid redundant LU factorizations.

Cache keys are based on: `(x_grid, method, boundary_condition, extrapolation)`

```julia
# Same cache key → cache HIT
x = 0:0.1:10
cubic_interp(x, y1, xq)  # cache miss (first call)
cubic_interp(x, y2, xq)  # cache HIT (reuses factorization)

# Different x object → cache MISS
cubic_interp(collect(0:0.1:10), y, xq)  # new object = cache miss!
```

**Define grid once outside loops:**
```julia
x = range(0.0, 10.0, 100)

for step in 1:10000
    y = compute(step)
    cubic_interp!(out, x, y, xq)  # cache HIT every time
end
```

### Thread Safety

FastInterpolations.jl is thread-safe, but requires careful buffer management. Each thread **must** have its own output buffer:

```julia
# Create thread-local buffers
thread_outputs = [similar(xq) for _ in 1:Threads.nthreads()]

Threads.@threads for i in 1:1000
    tid = Threads.threadid()
    cubic_interp!(thread_outputs[tid], x, y[i], xq)
end
```

### Type Stability

Ensure type stability for maximum performance:

```julia
# Type-stable: compile-time method selection
result = cubic_interp(x, y, xq)

# Type-unstable: runtime dispatch (slower)
method = user_input == "cubic" ? cubic_interp : linear_interp
result = method(x, y, xq)  # dynamic dispatch

# Fix: use if-else for compile-time dispatch
if user_input == "cubic"
    result = cubic_interp(x, y, xq)
else
    result = linear_interp(x, y, xq)
end
```
