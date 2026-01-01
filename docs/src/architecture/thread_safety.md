# Thread Safety

FastInterpolations.jl is designed for **safe concurrent use** from multiple threads. This page explains the thread-safety architecture and guarantees.

## Thread-Safety Guarantees

| Operation | Thread-Safe? | Mechanism |
|-----------|--------------|-----------|
| Cache lookup (hit) | Yes, lock-free | RCU pattern |
| Cache insert (miss) | Yes, locked | Global lock + copy-on-write |
| Interpolation | Yes | Task-local workspaces |
| Clear cache | Yes | Atomic snapshot replacement |

## RCU Pattern (Read-Copy-Update)

The auto-cache uses **RCU (Read-Copy-Update)**, a concurrent programming pattern that enables:
- **Lock-free reads** (~10 ns cache lookup)
- **Safe concurrent access** without data races
- **Zero allocation** on cache hits

### How RCU Works

```
     ┌──────────────────────────────────────────────────┐
     │                   Read Path                       │
     │                  (lock-free)                      │
     └──────────────────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │    @atomic :acquire load snapshot    │
          │         (immutable reference)        │
          └─────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │      Scan snapshot for x-grid       │
          │    (safe - snapshot never changes)   │
          └─────────────────────────────────────┘


     ┌──────────────────────────────────────────────────┐
     │                   Write Path                      │
     │                (under lock)                       │
     └──────────────────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │          Acquire global lock         │
          └─────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │   Double-check (another thread may   │
          │    have inserted while we waited)    │
          └─────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │     Copy snapshot → Modify copy     │
          │     → Atomic publish new snapshot    │
          └─────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────┐
          │          Release global lock         │
          └─────────────────────────────────────┘
```

### Key Insight: Immutable Snapshots

The core idea is that **readers never see partially-modified data**:

1. A snapshot contains the cache state at a point in time
2. Once created, a snapshot is **immutable** (never modified in place)
3. Writers create a **new snapshot** and atomically publish it
4. Readers always see a complete, consistent view

## Task-Local Workspaces

Solving the spline system requires temporary arrays. These are allocated from **task-local pools** via `AdaptiveArrayPools.jl`:

```
        Thread 1                    Thread 2
            │                           │
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │  Task-local   │           │  Task-local   │
    │    Pool 1     │           │    Pool 2     │
    └───────────────┘           └───────────────┘
            │                           │
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │  d_workspace  │           │  d_workspace  │
    │  z_workspace  │           │  z_workspace  │
    └───────────────┘           └───────────────┘
```

Benefits:
- No contention between threads (each task has isolated workspace)
- Zero allocation after warmup (pool reuses arrays)
- Automatic cleanup when task completes

## Memory Ordering

The cache uses explicit memory ordering for correctness:

| Operation | Ordering | Purpose |
|-----------|----------|---------|
| Reader load | `:acquire` | See all writes before snapshot publish |
| Writer store | `:release` | Ensure snapshot fully initialized before visible |
| Under lock | `:monotonic` | Lock provides synchronization |

## Immutable Cache Design

The `CubicSplineCache` struct contains **only immutable data**:

```julia
struct CubicSplineCache{T, X, F, BC}
    x::X              # Grid points (immutable reference)
    h::Vector{T}      # Spacing (computed once, never modified)
    lu_factor::F      # LU factorization (computed once)
    bc_config::BC     # Boundary condition config (immutable)
end
```

Multiple threads can safely read the same cache simultaneously because the data never changes after construction.

## Performance Comparison

| Package | Multi-Thread Safety | Cache Hit Cost |
|---------|---------------------|----------------|
| FastInterpolations.jl | Yes (RCU) | ~10 ns (lock-free) |
| Interpolations.jl | Limited | N/A (no auto-cache) |
| DataInterpolations.jl | Yes (per-object) | N/A (no global cache) |

## Multi-Threaded Example

```julia
using FastInterpolations
using Base.Threads

x = range(0.0, 10.0, 100)
queries = [rand(10) .* 10 for _ in 1:nthreads()]
results = [zeros(10) for _ in 1:nthreads()]

# Safe concurrent interpolation
@threads for i in 1:nthreads()
    y = sin.(x .+ i * 0.1)  # Each thread has different y
    cubic_interp!(results[i], x, y, queries[i])
end

# All threads share the cached LU factorization for x
```

## Data Structures Summary

| Component | Thread-Safety Mechanism |
|-----------|------------------------|
| `CubicSplineCache` | Immutable (no mutable fields) |
| Workspaces | Task-local via `AdaptiveArrayPools` |
| Registry | RCU with atomic snapshot |
| Cache Bank | RCU with atomic snapshot |
| Eviction | Protected by global lock |

## Julia Version Notes

- **Minimum**: Julia 1.7+ (for `@atomic` field support)
- **Recommended**: Julia 1.12+ (improved atomic semantics)

The RCU pattern works correctly on all supported Julia versions, though Julia 1.12+ provides improved performance characteristics for atomic operations.

## See Also

- **[Auto-Cache](caching.md)**: Cache key semantics and configuration
- **[Architecture Overview](overview.md)**: When to use one-shot vs interpolant
