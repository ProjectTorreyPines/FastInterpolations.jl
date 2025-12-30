# Thread-Safety Architecture for FastInterpolations.jl

## Overview

This document describes the thread-safety architecture for the cubic spline autocache system. The design achieves **lock-free cache hits** while maintaining correctness under concurrent access.

### Design Goals

1. **Lock-Free Hit Path**: Cache lookups must not acquire locks (hot path optimization)
2. **Zero Allocation**: No heap allocation on cache hits after warmup
3. **Thread Safety**: Eliminate all race conditions without sacrificing single-thread performance
4. **Backward Compatibility**: Preserve existing API semantics

## Architecture

### 1. Immutable Cache Design

The `CubicSplineCache` struct contains **only immutable data**:

```julia
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F,BC}
    x::X              # Grid points (immutable)
    h::Vector{T}      # Spacing (computed once)
    lu_factor::F      # LU factorization (computed once)
    bc_config::BC     # Boundary condition config (immutable)
end
```

**Key insight**: By removing mutable workspace fields, the cache becomes inherently thread-safe. Multiple threads can read the same cache simultaneously.

### 2. Task-Local Workspaces

Temporary arrays required for solving the spline system are allocated from **task-local pools** via `@with_pool` backed by the `AdaptiveArrayPools.jl` dependency:

```
Interpolation Call
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  @with_pool pool begin                  │
  │      d_workspace = similar!(pool, y)    │ ← Task-local
  │      z_workspace = similar!(pool, y)    │ ← Task-local
  │      _solve_system!(z, cache, y, bc)    │
  │  end                                    │
  └─────────────────────────────────────────┘
```

Benefits:
- Zero allocation after warmup (pool reuses arrays)
- No contention between threads (each task has its own pool)
- Automatic cleanup when task completes
- **Thread-safe by isolation**: No shared mutable state during system solve.

### 3. RCU Pattern for Registry and Banks

The autocache uses **Read-Copy-Update (RCU)** for thread-safe cache management:

```
                    ┌──────────────────┐
     Readers ──────▶│ Immutable        │
     (lock-free)    │ Snapshot         │
                    └──────────────────┘
                            ▲
                            │ atomic load
                            │
    ┌─────────────┬─────────┴─────────┬─────────────┐
    │  CacheBank  │  @atomic snapshot │             │
    └─────────────┴───────────────────┴─────────────┘
                            ▲
                            │ atomic store
                            │ (under lock)
    ┌──────────────────────────────────────────────┐
    │  Writer: copy → modify → publish atomically  │
    └──────────────────────────────────────────────┘
```

#### Data Structures

**Bank Snapshot** (immutable after creation):
```julia
struct BankSnapshot{E<:AbstractCacheEntry}
    store::Vector{E}  # Cache entries
    count::Int        # Valid entry count
    ring::Int         # Next eviction index (1-based)
end
```

**Cache Bank** (atomic reference holder):
```julia
mutable struct CacheBank{E<:AbstractCacheEntry}
    @atomic snapshot::BankSnapshot{E}
end
```

**Registry** (type → bank mapping):
```julia
mutable struct GlobalRegistry
    @atomic snapshot::Vector{Pair{DataType, Any}}
end
```

#### Read Path (Lock-Free)

```julia
# 1. Atomic load (acquire ordering)
snap = @atomic :acquire bank.snapshot

# 2. Two-pass lookup (safe - snapshot is immutable)
# Pass 1: Identity check (fast)
for entry in snap.store[1:snap.count]
    entry.id === objectid(x) && return entry.cache
end
# Pass 2: Equality check (slow, but guaranteed correct)
for entry in snap.store[1:snap.count]
    isequal(entry.x, x) && return entry.cache
end
```

**Note**: No self-healing is performed on the read path. Snapshots are fully immutable to ensure thread safety.

#### Write Path (Under Lock)

The write path is protected by a global `_CACHE_LOCK` to prevent race conditions during bank creation or entry eviction.

```julia
lock(_CACHE_LOCK)
try
    # Double-check after acquiring lock
    snap = @atomic :monotonic bank.snapshot
    found = _rcu_lookup(snap, id, x)
    found !== nothing && return found

    # Build new cache and create new snapshot
    new_cache = _build_cache(E, x, bc_config)
    new_store = copy(snap.store)
    # ... modify new_store (ring buffer eviction) ...
    new_snap = BankSnapshot{E}(new_store, new_count, new_ring)
    @atomic :release bank.snapshot = new_snap
finally
    unlock(_CACHE_LOCK)
end
```

**Eviction Policy**: A simple **ring buffer** (FIFO) is used for cache eviction. This ensures $O(1)$ replacement time and avoids the complexity/overhead of LRU in a multi-threaded context.

## Configuration and Management

Users and developers can manage the autocache via the following API:

- `set_cubic_cache_size!(n)`: Sets the maximum number of entries per bank (persisted via `Preferences.jl`). Requires a Julia restart.
- `get_cubic_cache_size()`: Returns the current load-time constant for cache size.
- `clear_cubic_cache!()`: Atomically clears all cached grids by replacing registry snapshots.

## Memory Ordering

| Operation | Ordering | Purpose |
|-----------|----------|---------|
| Reader load | `:acquire` | See all writes before snapshot publish |
| Writer store | `:release` | Ensure new snapshot fully initialized before visible |
| Under lock | `:monotonic` | Lock provides synchronization |

## Performance Characteristics

| Operation | Cost |
|-----------|------|
| Registry lookup | 1 atomic load + linear scan (~20 entries) |
| Bank lookup | 1 atomic load + 2-pass linear scan (~16 entries) |
| Workspace | Pool reuse (0 allocation after warmup) |

**Zero Allocation**: On Julia 1.12+, cache hits are guaranteed to be zero-allocation. On older versions, minor allocations may occur due to closure captures in some edge cases, though the core RCU path remains highly efficient.

## Julia Version Requirements

- **Minimum**: Julia 1.7+ (for `@atomic` field support)
- **LTS Compatible**: Julia 1.10+
- **Recommended**: Julia 1.12+ (improved atomic semantics, zero allocation)

## Component Summary

| Component | Thread-Safety Mechanism |
|-----------|------------------------|
| `CubicSplineCache` | Immutable (no mutable fields) |
| `PeriodicData` | Immutable (`q` vector + `period` scalar only) |
| Workspaces | Task-local via `AdaptiveArrayPools` |
| Registry | RCU with atomic snapshot |
| Bank | RCU with atomic snapshot |
| Ring buffer eviction | Protected by global lock |

## Related Files

- `src/cubic_types.jl`: Cache and BC type definitions
- `src/cubic_solver.jl`: System solvers with `@with_pool`
- `src/cubic_autocache.jl`: RCU registry and bank implementation
- `src/FastInterpolations.jl`: `AdaptiveArrayPools` integration and pool setup
