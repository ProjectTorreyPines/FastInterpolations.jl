# Thread-Safety Design for FastInterpolations.jl Autocache

**Status**: In Progress
**Date**: 2024-12-29
**Version**: 3.0

## Executive Summary

Two race conditions exist in `cubic_autocache.jl`:

| Problem | Symptom | Root Cause | Solution |
|---------|---------|------------|----------|
| **1. Ring Buffer Race** | `ConcurrencyViolationError` | Concurrent `push!` during warm-up | Double-check locking |
| **2. Workspace Race** | Data corruption (~50%) | Shared mutable workspaces | **Remove from Cache** |

### Design Principle: Zero Overhead for Single-Thread Users

Most users run single-threaded. The design ensures **zero lock overhead** for them:

| Mode | Registry | Cache Hit | Self-healing | Cache Miss |
|------|----------|-----------|--------------|------------|
| Single-thread | Lock-free | Lock-free | **Enabled** | **Lock-free** |
| Multi-thread | Lock-free | Lock-free | **Disabled** | Lock |

**Key insight**: `Threads.nthreads() == 1` check costs ~1ns vs lock acquire ~50-100ns.

---

## Implementation Order (Revised)

**Phase 2 first** - Workspace removal is:
1. Easier to test with `test/minimal_thread_test.jl`
2. More impactful (eliminates ~50% data corruption)
3. Cleaner architecture (no mutable state in cache)

### Progress

- [x] **Phase 0**: Configuration Simplification (Preferences.jl) ✅
- [ ] **Phase 2**: Workspace Removal (AdaptiveArrayPools) ← **Current**
- [ ] **Phase 1**: Ring Buffer & Registry Safety (Double-check locking)
- [ ] **Phase 3**: Testing & Validation

---

## Phase 2: Workspace Removal (CURRENT PRIORITY)

### Goal: Eliminate All Mutable State from Cache

**Before** (Thread-unsafe):
```
CubicSplineCache
├── x::X                    (immutable ✓)
├── h::Vector{T}            (immutable ✓)
├── lu_factor::F            (immutable ✓)
├── d_workspace::Vector{T}  (MUTABLE - race!)
├── z_workspace::Vector{T}  (MUTABLE - race!)
└── bc_data::BC
    └── (PeriodicData)
        ├── q::Vector{T}    (immutable ✓)
        ├── y_temp::Vector  (MUTABLE - race!)
        └── period::T       (immutable ✓)
```

**After** (Thread-safe by design):
```
CubicSplineCache
├── x::X                    (immutable ✓)
├── h::Vector{T}            (immutable ✓)
├── lu_factor::F            (immutable ✓)
└── bc_data::BC
    └── (PeriodicData)
        ├── q::Vector{T}    (immutable ✓)
        └── period::T       (immutable ✓)

Workspaces: AdaptiveArrayPools (task-local, zero-allocation after warmup)
```

### Type Changes

#### 1. CubicSplineCache (Remove d_workspace, z_workspace)

```julia
# BEFORE
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F,BC}
    x::X
    h::Vector{T}
    lu_factor::F
    d_workspace::Vector{T}  # REMOVE
    z_workspace::Vector{T}  # REMOVE
    bc_data::BC
end

# AFTER
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F,BC}
    x::X
    h::Vector{T}
    lu_factor::F
    bc_data::BC
end
```

#### 2. PeriodicData (Remove y_temp)

```julia
# BEFORE
struct PeriodicData{T<:AbstractFloat}
    q::Vector{T}
    y_temp::Vector{T}  # REMOVE
    period::T
end

# AFTER
struct PeriodicData{T<:AbstractFloat}
    q::Vector{T}
    period::T
end
```

### Workspace Size Analysis (CRITICAL)

Workspace sizes differ between BC types. This is critical for correct pool allocation:

| BC Type | Workspace | Size | Relation to `length(y)` |
|---------|-----------|------|-------------------------|
| **Derivative (BCPair)** | d_workspace | n+1 | `length(y)` |
| **Derivative (BCPair)** | z_workspace | n+1 | `length(y)` |
| **Periodic** | d_workspace | n | `length(y) - 1` |
| **Periodic** | z_workspace | n+1 | `length(y)` |
| **Periodic** | y_temp | n | `length(y) - 1` |

Where `n = length(y) - 1` (number of intervals).

### Workspace Acquisition Pattern

Two allocation methods depending on size requirements:

#### Method 1: `similar!(pool, arr)` - Same size as reference array
```julia
# Use when workspace size matches an existing array
d_workspace = similar!(pool, y)  # Creates array with same size as y
```

#### Method 2: `acquire!(pool, T, N)` - Explicit size
```julia
# Use when workspace size differs from available arrays (e.g., Periodic BC)
N = length(y)
n = N - 1  # Periodic d_workspace needs n elements, not N
d_workspace = acquire!(pool, T, n)
y_temp = acquire!(pool, T, n)
```

#### Example: Derivative BC Solver (uses `similar!`)
```julia
@inline @with_pool pool function _solve_system!(
    out_z::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BCPair{T,L,R}},
    y::AbstractVector{T},
    bc_pair::BCPair{T,L,R}
) where {T,X,F,L,R}
    # d_workspace needs length(y) = n+1, same as y
    d_workspace = similar!(pool, y)

    compute_rhs!(d_workspace, y, cache.h, bc_pair)
    ldiv!(out_z, cache.lu_factor, d_workspace)
    return out_z
end
```

#### Example: Periodic BC Solver (uses `acquire!` for different sizes)
```julia
@inline @with_pool pool function _solve_cubic_system_periodic!(
    z_workspace::Vector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T}
) where {T,X,F}
    N = length(y)
    n = N - 1

    # d_workspace needs n elements (NOT N!)
    d_workspace = acquire!(pool, T, n)
    # y_temp also needs n elements
    y_temp = acquire!(pool, T, n)

    compute_rhs_periodic!(d_workspace, y, cache.h)
    ldiv!(y_temp, cache.lu_factor, d_workspace)
    # ... Sherman-Morrison formula ...
    return z_workspace
end
```

### API Pattern Change: Deprecate `_get_cache_and_solve!` Helpers

**Problem**: Old helpers (`_get_cache_and_solve!`, `_get_cache_and_solve_periodic!`) conflated
cache retrieval with workspace management, making thread-safety difficult.

**Solution**: Separate concerns using new dispatch pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                     OLD PATTERN (DEPRECATED)                    │
├─────────────────────────────────────────────────────────────────┤
│  cache = _get_cache_and_solve!(x, y, bc, autocache)             │
│  z = cache.z_workspace  ← Shared mutable workspace (RACE!)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     NEW PATTERN (THREAD-SAFE)                   │
├─────────────────────────────────────────────────────────────────┤
│  cache = _get_cubic_cache(x, bc, autocache)  ← Immutable cache  │
│  @with_pool pool begin                                          │
│      tmp_z = similar!(pool, y)               ← Task-local       │
│      _solve_system!(tmp_z, cache, y, bc)     ← No shared state  │
│      # Use tmp_z or copy as needed                              │
│  end                                                            │
└─────────────────────────────────────────────────────────────────┘
```

#### New Unified Cache Getter (Already exists in `cubic_autocache.jl`)
```julia
@inline function _get_cubic_cache(
    x::AbstractVector{T},
    bc::AbstractBC,
    autocache::Bool
) where {T<:AbstractFloat}
    if autocache
        cache = _get_cubic_cache(x, bc)  # From ring buffer
    else
        cache = CubicSplineCache(x; bc=bc)  # Fresh allocation
    end
    return cache  # Immutable, no workspace fields
end
```

#### Migration Examples

**BCPair Scalar Interpolation**:
```julia
# OLD (deprecated)
cache = _get_cache_and_solve!(x, y, bc_pair, autocache)
z = cache.z_workspace

# NEW (thread-safe)
@with_pool pool function _cubic_interp_bcpair_scalar(...)
    cache = _get_cubic_cache(x, bc, autocache)
    tmp_z = similar!(pool, y)
    _solve_system!(tmp_z, cache, y, bc)
    # ... evaluate using tmp_z ...
end
```

**CubicInterpolant Construction**:
```julia
# OLD (deprecated)
cache = _get_cache_and_solve!(x, y, bc_pair, autocache)
return CubicInterpolant(cache, y, cache.z_workspace, ev)

# NEW (thread-safe)
@with_pool pool function _build_interpolant_bcpair(...)
    cache = _get_cubic_cache(x, bc, autocache)
    tmp_z = similar!(pool, y)
    _solve_system!(tmp_z, cache, y, bc)
    return CubicInterpolant(cache, y, tmp_z, ev)  # tmp_z copied in constructor
end
```

### CubicInterpolant: Pool Only at Construction

**Critical Insight**: `CubicInterpolant` stores pre-computed `z` coefficients.
Pool access is needed **only at construction**, not at call time.

```
┌─ Construction (1x) ──────────────────────────────────────────────┐
│  @with_pool pool begin                                           │
│      cache = _get_cubic_cache(x, bc, autocache)                  │
│      tmp_z = similar!(pool, y)                                   │
│      _solve_system!(tmp_z, cache, y, bc)                         │
│      return CubicInterpolant(cache, y, tmp_z, ev)                │
│  end                           └─ Copied into itp.z              │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─ Call (Nx) ──────────────────────────────────────────────────────┐
│  itp(xi)                                                         │
│    _eval_with_bc(cache, itp.y, cache.h, itp.z, xi, ...)          │
│                                        ^^^^^^                    │
│                                        Pre-computed, stored      │
│  ✅ NO POOL ACCESS                                               │
│  ✅ ZERO ALLOCATION                                              │
│  ✅ HOT PATH PRESERVED                                           │
└──────────────────────────────────────────────────────────────────┘
```

### Files to Modify

| File | Changes |
|------|---------|
| `cubic_types.jl` | Remove workspace fields from structs |
| `cubic_solver.jl` | Update builders, remove workspace allocation |
| `cubic_interp.jl` | Remove `_get_cache_and_solve!` helpers, use `_get_cubic_cache` + pool |
| `cubic_interpolant.jl` | Update builders with `@with_pool`, callable unchanged |
| `cubic_autocache.jl` | Update CacheEntry type annotations |

### Implementation Checklist

#### Step 1: Update Type Definitions
- [ ] Remove `d_workspace`, `z_workspace` from `CubicSplineCache`
- [ ] Remove `y_temp` from `PeriodicData`
- [ ] Update docstrings

#### Step 2: Update Cache Builders
- [ ] `_build_derivative_bc_cache`: Remove workspace allocation
- [ ] `_build_periodic_cache`: Remove workspace allocation
- [ ] `CubicSplineCache` constructor: Remove workspace params

#### Step 3: Update Solvers (`cubic_solver.jl`)
- [x] `_solve_system!(out_z, cache, y, bc_pair)` - 4-arg BCPair (already done)
- [ ] `_solve_system!(cache, y, bc_pair)` - 3-arg BCPair: **REMOVE** (no longer valid without workspace)
- [ ] `_solve_cubic_system_periodic!(out_z, d_ws, y_temp, cache, y)` - New signature with explicit workspaces
- [ ] `_solve_system!(out_z, cache, y, ::PeriodicData)` - Wrapper with pool

#### Step 4: Update Interp Functions (`cubic_interp.jl`)
- [ ] **DELETE** `_get_cache_and_solve!` helper
- [ ] **DELETE** `_get_cache_and_solve_periodic!` helper
- [x] `_cubic_interp_bcpair_scalar` - Already uses `@with_pool`, update to use `_get_cubic_cache`
- [ ] `_cubic_interp_bcpair!` - Add `@with_pool`, use `_get_cubic_cache`
- [ ] `_cubic_interp_periodic_scalar` - Add `@with_pool`, use `_get_cubic_cache`
- [ ] `_cubic_interp_periodic!` - Add `@with_pool`, use `_get_cubic_cache`

#### Step 4b: Update Cache-Based API (MISSING - added per review)
These functions accept pre-built cache and also need workspace removal:
- [ ] `cubic_interp!(output, cache, y, x_query)` in `cubic_interp.jl:154` - uses `_solve_system!(cache, y, cache.bc_data)`
- [ ] `cubic_interp_scalar(cache, y, x_query)` in `cubic_eval.jl:212` - uses `_solve_system!(cache, y, cache.bc_data)`
- [ ] `cubic_interp(cache, y; extrap)` in `cubic_interpolant.jl:189` - uses `_solve_system!(cache, y, cache.bc_data)`

#### Step 5: Update CubicInterpolant (`cubic_interpolant.jl`)
- [ ] `_build_interpolant_bcpair`: Add `@with_pool`, use `_get_cubic_cache`
- [ ] `_build_interpolant_periodic`: Add `@with_pool`, use `_get_cubic_cache`
- [ ] Callable `(itp::CubicInterpolant)(x)`: **NO CHANGE** - uses stored `itp.z`

#### Step 6: Update Autocache (`cubic_autocache.jl`)
- [ ] `CacheEntry` type annotation (if explicit)
- [ ] Any workspace references in lookup functions

### Test Strategy

Use `test/minimal_thread_test.jl`:
```bash
julia -t 4 --project test/minimal_thread_test.jl
```

Expected results:
- **Before fix**: `❌ 데이터 손상 발생!` (~20-50% corruption)
- **After fix**: `✅ 문제 없음` (0% corruption)

---

## Phase 0: Configuration Simplification ✅ COMPLETED

- [x] Replace `_CACHE_SIZE_REF[]` with `const _CACHE_SIZE` via `@load_preference`
- [x] Add `set_cubic_cache_size!(n)` API using `@set_preferences!`
- [x] Update `_add_to_ring!` to use `_CACHE_SIZE` directly
- [x] Update tests for load-time constant behavior

---

## Phase 1: Ring Buffer & Registry Safety (DEFERRED)

Lower priority than workspace race - ring buffer race only occurs during cache warmup, workspace race occurs on every call.

### Problem Analysis: Ring Buffer Race

**Symptom**: `ConcurrencyViolationError` during cache warm-up phase.

**Root Cause**: `push!` to `Vector` is not thread-safe when:
1. Multiple threads try to add entries simultaneously
2. Vector needs to reallocate (grow) during `push!`

```
Thread 1: push!(store, entry1)  ─┐
                                 ├─ Vector reallocates → RACE!
Thread 2: push!(store, entry2)  ─┘
```

**When it happens**: Only during warm-up (first `_CACHE_SIZE` insertions per bank).
After warm-up, ring buffer overwrites existing slots (no reallocation).

### Solution: `sizehint!` + `__init__()`

#### Why `__init__()`?

Julia's module loading has two phases:
1. **Precompilation**: Code compiled to `.ji` file (happens once)
2. **Load-time**: `__init__()` runs every time module is loaded

```julia
# This runs at PRECOMPILE time (cached in .ji)
const _DERIVATIVE_BANK_REGISTRY = IdDict{DataType, Any}()

# This runs at LOAD time (every session)
function __init__()
    # Safe to modify mutable state here
end
```

#### Why `sizehint!`?

`sizehint!(vec, n)` pre-allocates capacity without changing length:

```julia
v = Vector{Int}()
sizehint!(v, 100)  # Allocates space for 100 elements
length(v)          # Still 0
push!(v, 1)        # No reallocation needed!
```

**Key insight**: If `store` vector has capacity ≥ `_CACHE_SIZE` before any `push!`,
no reallocation occurs during warm-up → no race condition.

#### Implementation Plan

**Two levels of `sizehint!`**:

1. **Registry level** (`IdDict`): Pre-allocate slots for bank types in `__init__()`
2. **Store level** (`Vector`): Pre-allocate slots for cache entries in bank creation

```julia
# At module initialization (top-level, precompiled)
const _DERIVATIVE_BANK_REGISTRY = IdDict{DataType, Any}()
const _PERIODIC_BANK_REGISTRY = IdDict{DataType, Any}()

function __init__()
    # Pre-allocate to prevent resize race
    # IdDict resizes at ~33% load factor
    # sizehint!(100) → 256 slots → safe for ~85 entries
    # Realistic usage: <20 different type combinations
    sizehint!(_DERIVATIVE_BANK_REGISTRY, 100)
    sizehint!(_PERIODIC_BANK_REGISTRY, 100)
end
```

**Why `__init__()` and not top-level?**
- `__init__()` runs at `using` time (every Julia session)
- Precompilation serializes empty IdDict to `.ji` file
- `__init__()` restores runtime state → `sizehint!` always applied

**Capacity analysis**:

| `sizehint!` | Hash table size | Safe entries (~33% load) |
|-------------|-----------------|--------------------------|
| 50 | 128 | ~42 |
| 100 | 256 | ~85 |

**Realistic usage**: Most applications use 1-20 type combinations (Float64/Float32 × BC types × Vector types).

**Result**: With `sizehint!(100)`, resize will **never** occur in realistic usage → **completely safe without locks**.

```julia
# Modify bank creation to pre-size store vectors:
function _get_derivative_bank(::Type{T}, ::Type{L}, ::Type{R}, ::Type{X}) where {T,L,R,X}
    BankType = CacheBank{T,L,R,X}
    if !haskey(_DERIVATIVE_BANK_REGISTRY, BankType)
        lock(_CACHE_LOCK) do
            if !haskey(_DERIVATIVE_BANK_REGISTRY, BankType)
                store = Vector{CacheEntry{T,L,R,X}}()
                sizehint!(store, _CACHE_SIZE)  # ← Pre-allocate for ring buffer!
                _DERIVATIVE_BANK_REGISTRY[BankType] = CacheBank{T,L,R,X}(store, Ref(1))
            end
        end
    end
    return _DERIVATIVE_BANK_REGISTRY[BankType]::BankType
end
```

#### Why This Works

```
                    WITHOUT sizehint!                     WITH sizehint!
                    ─────────────────                     ──────────────
Thread 1: push!() ─┐                          Thread 1: push!() ─┐
                   │ capacity=4, need grow!                      │ capacity=16, no grow
Thread 2: push!() ─┤ ← RACE during realloc    Thread 2: push!() ─┤ ← Safe: just writes
                   │                                             │
                   ▼                                             ▼
              ConcurrencyViolationError                     ✅ Success
```

### Checklist (for later)
- [ ] Add `sizehint!(store, _CACHE_SIZE)` in `_get_derivative_bank`
- [ ] Add `sizehint!(store, _CACHE_SIZE)` in `_get_periodic_bank`
- [ ] Make self-healing conditional: `if Threads.nthreads() == 1`
- [ ] Add double-check locking in `_lookup_or_insert!` (multi-thread only)
- [ ] Add double-check locking in `_lookup_or_insert_periodic!` (multi-thread only)
- [ ] Make registry creation lock conditional: `if Threads.nthreads() > 1`

### Remaining Race Conditions (Trade-off Analysis)

**⚠️ IMPORTANT**: Phase 1 (`sizehint!` + DCL) addresses the most critical race (vector reallocation during warm-up), but does NOT provide full thread-safety for all operations.

#### Remaining Races (Accepted Trade-offs)

| Operation | Race Condition | Mitigation | Risk Level |
|-----------|----------------|------------|------------|
| **Lock-free read during write** | Iteration sees partially-written entry | `sizehint!` prevents realloc; entry assignment is atomic reference | Low |
| **Registry read during `clear_cubic_cache!`** | `empty!()` races with `haskey()` | `clear_cubic_cache!` is rare (testing only); use lock | Low |
| **Self-healing in multi-thread** | Disabled; no race | N/A | None |

#### Design Decision: "Good Enough" Thread-Safety

**Why not lock every read?**
- Hit path is the hot path (>99% of calls after warm-up)
- Lock acquisition: ~50-100ns per call
- For 1M interpolations: 50-100ms overhead vs ~0ns

**Accepted limitations**:
1. `clear_cubic_cache!()` should only be called when no other threads are interpolating
2. During warm-up, rare duplicate entries may be created (harmless, just wastes memory temporarily)

**For strict thread-safety**, users should:
- Pre-warm cache single-threaded before parallel work
- Avoid `clear_cubic_cache!()` during parallel interpolation

### Double-Check Locking Pattern

**⚠️ BUG FIX**: Previous version had control flow error (`return` inside `lock() do` returns from lambda, not outer function).

```julia
@inline function _lookup_or_insert!(bank::CacheBank{T,L,R,X}, x::X, bc_pair::BCPair{T,L,R}) where {T,L,R,X}
    store = bank.store
    id = objectid(x)
    is_single_thread = Threads.nthreads() == 1  # ~1ns check

    # === LOCK-FREE READ PATH (Cache Hit) ===
    @inbounds for entry in store
        entry.id === id && return entry.cache
    end
    @inbounds for entry in store
        if isequal(entry.x, x)
            is_single_thread && (entry.id = id)  # Self-healing
            return entry.cache
        end
    end

    # === WRITE PATH (Cache Miss) ===
    new_cache = _build_derivative_bc_cache(x, bc_pair.left, bc_pair.right)

    if is_single_thread
        new_entry = CacheEntry{T,L,R,X}(id, x, new_cache)
        _add_to_ring!(store, bank.ring, new_entry)
        return new_cache
    else
        # Double-check locking with proper control flow
        # (lock() do ... end return only exits lambda, not outer function)
        found_cache = lock(_CACHE_LOCK) do
            for entry in store
                if entry.id === id || isequal(entry.x, x)
                    return entry.cache  # Returns from lambda → assigned to found_cache
                end
            end
            new_entry = CacheEntry{T,L,R,X}(id, x, new_cache)
            _add_to_ring!(store, bank.ring, new_entry)
            return nothing  # No existing entry found
        end
        # Return found cache if double-check hit, else return new_cache
        return found_cache !== nothing ? found_cache : new_cache
    end
end
```

---

## Phase 3: Testing & Validation

### Test Files
- `test/minimal_thread_test.jl` - Quick smoke test
- `test/test_thread_safety.jl` - Comprehensive stress test

### Test Matrix

| Test | BC Type | Query Type | Expected |
|------|---------|------------|----------|
| `test_workspace_corruption` | Natural | Vector | 0 errors |
| `test_periodic_corruption` | Periodic | Vector | 0 errors |
| `test_scalar_corruption` | Natural | Scalar | 0 errors |
| `test_cache_thrashing` | Natural | Scalar | 0 errors |

### Run Commands
```bash
# Quick test
julia -t 4 --project test/minimal_thread_test.jl

# Full test suite
julia -t 4 --project test/test_thread_safety.jl
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    cubic_interp(x, y, q)                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              _get_cubic_cache(x, bc, autocache)             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  autocache=true → _lookup_or_insert!(bank, x, bc)   │    │
│  │  autocache=false → CubicSplineCache(x; bc=bc)       │    │
│  └─────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  CubicSplineCache (IMMUTABLE)               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  x::X           - Grid points                          │ │
│  │  h::Vector{T}   - Spacing                              │ │
│  │  lu_factor::F   - LU factorization (reusable!)         │ │
│  │  bc_data::BC    - Boundary condition data              │ │
│  │                                                        │ │
│  │  NO WORKSPACES - Thread-safe by design                 │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              @with_pool _solve_system!(...)                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  d_workspace = similar!(pool, y)   ← Task-local        │ │
│  │  z_workspace = similar!(pool, y)   ← Task-local        │ │
│  │  y_temp = similar!(pool, y)        ← Periodic only     │ │
│  │                                                        │ │
│  │  compute_rhs!(d_workspace, y, h, bc)                   │ │
│  │  ldiv!(z_workspace, lu_factor, d_workspace)            │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  _eval_cubic(..., z, ...)                   │
│                  Return interpolated value                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Impact

### Single-Thread Mode (Majority of Users)

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| Cache struct | 5 fields | 4 fields | Smaller |
| Registry lookup | Lock-free | Lock-free | None |
| Cache hit | Lock-free | Lock-free | None |
| Workspace | cache.d_workspace | pool.similar!() | ~same |

**Single-thread overhead: ~1ns** (`Threads.nthreads()` check only)

### Multi-Thread Mode

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| Cache miss | Race! | Lock (Phase 1) | +50-100ns |
| Workspace | Shared race! | Task-local | **Race eliminated** |

**Multi-thread**: Race conditions eliminated, minimal overhead.

---

## Appendix: AdaptiveArrayPools

`AdaptiveArrayPools` uses `task_local_storage()`:
- Each Julia Task gets its own pool instance
- Pools follow tasks when they migrate between OS threads
- Memory reclaimed when tasks complete
- Nested `@with_pool` calls share the same pool within a task
- Zero allocation after warmup (pool reuses arrays)

```julia
using AdaptiveArrayPools

@with_pool pool function example(n)
    # First call: allocates
    # Subsequent calls: reuses from pool
    arr = similar!(pool, Vector{Float64}, n)
    # ... use arr ...
    # arr automatically returned to pool when function exits
end
```
