"""
    cubic_autocache.jl

Automatic caching layer for cubic spline interpolation.
Transparently reuses LU factorization for repeated x-grids.

# Implementation Details

- Generic `CacheBank{E}` - single bank type parameterized by entry type
- **RCU (Read-Copy-Update)** pattern for lock-free cache hits
- Zero-allocation cache hit via 2-pass lookup (objectid → isequal)
- Ring buffer eviction for O(1) cache replacement
- `GlobalRegistry` with atomic `Vector{Pair}` for lock-free bank lookup

# Thread Safety (RCU Pattern)

**Read path (cache hit)**: Lock-free
- `@atomic :acquire` snapshot load (registry and bank)
- Linear scan for type match (registry: N<20 banks)
- 2-pass lookup: objectid fast path → isequal slow path

**Write path (cache miss)**: Lock protected
- Lock → double-check → copy snapshot → modify → `@atomic :release` publish

**Performance**:
- Full interp cache hit: ~810 ns/op
- **Zero allocation** on cache hit (Julia 1.12+, avoid closure capture)

**Compatibility**: Julia 1.7+ (uses `@atomic` field, not AtomicMemory)
"""

# ===============================================================
# Cache Entry Types
# ===============================================================

"""
    AbstractCacheEntry{T, X}

Abstract type for cache entries. Subtypes must have:
- `id::UInt` - object identity for fast lookup
- `x::X` - grid data for equality check
- `cache` - CubicSplineCache instance
"""
abstract type AbstractCacheEntry{T<:AbstractFloat, X<:AbstractVector{T}} end

"""
    CacheEntry{T, L, R, X, S}

Cache entry for derivative BC (uses BCPair).

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `L`: Left boundary condition type (Deriv1{T} or Deriv2{T})
- `R`: Right boundary condition type (Deriv1{T} or Deriv2{T})
- `X`: Grid type (Vector{T} or StepRangeLen)
- `S`: Grid spacing type (ScalarSpacing{T} or VectorSpacing{T})

# Fields
- `id::UInt`: objectid of the ORIGINAL input x (hint for fast lookup)
- `x::X`: SNAPSHOT of the grid (copy for Vector, same object for Range)
- `cache`: CubicSplineCache built from the snapshot

# Mutation Safety
The `x` field stores a snapshot (copy) for Vector inputs, preventing external
mutation from corrupting the cache. Lookup verifies `isequal(entry.x, input_x)`
even on objectid match to detect in-place mutation.
"""
mutable struct CacheEntry{T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}, S<:AbstractGridSpacing{T}} <: AbstractCacheEntry{T, X}
    id::UInt
    x::X
    cache::CubicSplineCache{T, X, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, BCPair{T,L,R}, S, Vector{T}}
end

"""
    PeriodicCacheEntry{T, X, S}

Cache entry for periodic BC (uses PeriodicData).

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `X`: Grid type (Vector{T} or StepRangeLen)
- `S`: Grid spacing type (ScalarSpacing{T} or VectorSpacing{T})

# Fields
- `id::UInt`: objectid of the ORIGINAL input x (hint for fast lookup)
- `x::X`: SNAPSHOT of the grid (copy for Vector, same object for Range)
- `cache`: CubicSplineCache built from the snapshot

# Mutation Safety
See `CacheEntry` documentation for details on mutation safety pattern.
"""
mutable struct PeriodicCacheEntry{T<:AbstractFloat, X<:AbstractVector{T}, S<:AbstractGridSpacing{T}} <: AbstractCacheEntry{T, X}
    id::UInt
    x::X
    cache::CubicSplineCache{T, X, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, PeriodicData{T}, S, Vector{T}}
end

# ===============================================================
# RCU Snapshot (Immutable Bank State)
# ===============================================================

"""
    BankSnapshot{E}

Immutable snapshot of bank state at a point in time.
Used for RCU (Read-Copy-Update) pattern to enable lock-free reads.

# Fields
- `store::Vector{E}` - Cache entries (effectively immutable after creation)
- `count::Int` - Number of valid entries
- `ring::Int` - Next eviction index (1-based)

# Thread Safety
Readers atomically load a snapshot reference and can safely read it
without locks since the snapshot content never changes after creation.
"""
struct BankSnapshot{E<:AbstractCacheEntry}
    store::Vector{E}
    count::Int
    ring::Int  # 1-based next eviction index
end

# Empty snapshot constructor
function BankSnapshot{E}() where {E<:AbstractCacheEntry}
    store = E[]
    sizehint!(store, _CACHE_SIZE)
    BankSnapshot{E}(store, 0, 1)
end

# ===============================================================
# Generic Cache Bank (RCU-style)
# ===============================================================

"""
    CacheBank{E}

A generic cache bank holding entries of type `E`.
Uses RCU pattern with atomic snapshot for lock-free reads.

# Type Parameters
- `E`: Entry type (CacheEntry{T,L,R,X} or PeriodicCacheEntry{T,X})

# Thread Safety
- Read path: Lock-free via `@atomic :acquire` snapshot load
- Write path: Lock + copy-on-write + `@atomic :release` publish
"""
mutable struct CacheBank{E<:AbstractCacheEntry}
    @atomic snapshot::BankSnapshot{E}
end

# Single constructor for all entry types
function CacheBank{E}() where {E<:AbstractCacheEntry}
    CacheBank{E}(BankSnapshot{E}())
end

# ===============================================================
# RCU Registry (Lock-Free Bank Lookup)
# ===============================================================

"""
    RegistrySnapshot

Type alias for the registry's immutable snapshot.
Simple association list: [(BankType, BankInstance), ...]

# Design Choice
Using Vector{Pair{DataType, Any}} instead of IdDict because:
- N < 20 banks → linear scan faster than hash lookup (cache-friendly)
- Immutable snapshot enables lock-free reads via RCU pattern
- Copy-on-write for writes is O(N) with small N
"""
const RegistrySnapshot = Vector{Pair{DataType, Any}}

"""
    GlobalRegistry

Atomic registry wrapper for RCU-style lock-free bank lookup.

# Thread Safety
- Read path: Lock-free via `@atomic :acquire` snapshot load
- Write path: Lock → copy → push → `@atomic :release` publish

# Example
```julia
registry = GlobalRegistry()
snap = @atomic :acquire registry.snapshot  # Lock-free read
```
"""
mutable struct GlobalRegistry
    @atomic snapshot::RegistrySnapshot
end

# Default constructor with empty snapshot
GlobalRegistry() = GlobalRegistry(RegistrySnapshot())

# Global registries (RCU style)
const _DERIVATIVE_REGISTRY = GlobalRegistry()
const _PERIODIC_REGISTRY = GlobalRegistry()
const _CACHE_LOCK = ReentrantLock()

# Load-time constant: immutable after package load, enables compiler optimizations
# To change: call set_cubic_cache_size!(n), then restart Julia for it to take effect
# Default reduced from 16→8: with isequal verification, smaller cache = faster worst-case scan
const _CACHE_SIZE = @load_preference("cache_size", 8)::Int

# ===============================================================
# Module Initialization
# ===============================================================

# NOTE: With RCU pattern, __init__ is no longer needed for registry pre-allocation.
# The atomic registry handles concurrent access safely without pre-sizing.

# ===============================================================
# Public API
# ===============================================================

"""
    set_cubic_cache_size!(n::Int)

Set maximum cache size for future Julia sessions via Preferences.jl.

**Requires Julia restart** to take effect (cache size is a load-time constant
for thread-safety and compiler optimization).

# Arguments
- `n::Int`: Maximum cache entries (default: 16)

# Example
```julia
set_cubic_cache_size!(32)  # Sets preference
# Restart Julia to apply new size
get_cubic_cache_size()     # Now returns 32
```
"""
function set_cubic_cache_size!(n::Int)
    n > 0 || throw(ArgumentError("Cache size must be positive"))
    @set_preferences!("cache_size" => n)
    @info "Cache size preference set to $(n). Restart Julia to apply the change; the current session continues to use $(get_cubic_cache_size())."
    return n
end

"""
    get_cubic_cache_size()

Get current maximum cache size (load-time constant).
"""
get_cubic_cache_size() = _CACHE_SIZE

"""
    clear_cubic_cache!()

Clear all cached x-grids.

# Thread Safety (RCU)
Atomically replaces registry snapshots with empty vectors.
Existing readers continue to see their old snapshots until they finish.
"""
function clear_cubic_cache!()
    lock(_CACHE_LOCK) do
        # Atomic replace with empty snapshots (RCU pattern)
        @atomic :release _DERIVATIVE_REGISTRY.snapshot = RegistrySnapshot()
        @atomic :release _PERIODIC_REGISTRY.snapshot = RegistrySnapshot()
    end
    return nothing
end

# ===============================================================
# Internal: RCU Registry Lookup
# ===============================================================

"""
    _registry_lookup(registry, BankType) -> bank or nothing

Lock-free registry lookup using RCU pattern.
Atomically loads the snapshot and scans for matching bank type.

# Thread Safety
- Lock-free read via `@atomic :acquire`
- Linear scan on immutable snapshot (safe for concurrent reads)
"""
@inline function _registry_lookup(registry::GlobalRegistry, ::Type{B}) where {B}
    snap = @atomic :acquire registry.snapshot

    # Linear scan (N < 20 → faster than hash lookup)
    for (TypeKey, Bank) in snap
        if TypeKey === B
            return Bank::B
        end
    end

    return nothing
end

"""
    _get_bank(registry, BankType) -> bank

RCU-style bank retrieval with copy-on-write for new bank creation.

# Thread Safety (RCU)
- Read path (hit): Lock-free via `_registry_lookup`
- Write path (miss): Lock → copy → push → atomic publish

# Performance
- Registry hit: ~10 ns (atomic load + linear scan)
- Registry miss: Lock overhead + O(N) copy (N < 20)
"""
@inline function _get_bank(registry::GlobalRegistry, ::Type{CacheBank{E}}) where {E<:AbstractCacheEntry}
    BankType = CacheBank{E}

    # === RCU Read Path (Lock-Free) ===
    bank = _registry_lookup(registry, BankType)
    bank !== nothing && return bank

    # === RCU Write Path (Lock + Copy-on-Write) ===
    lock(_CACHE_LOCK)
    try
        # Double-check after acquiring lock
        bank = _registry_lookup(registry, BankType)
        bank !== nothing && return bank

        # Create new bank
        new_bank = CacheBank{E}()

        # Copy-on-write: copy snapshot → push → publish
        old_snap = @atomic :monotonic registry.snapshot
        new_snap = copy(old_snap)
        push!(new_snap, BankType => new_bank)
        @atomic :release registry.snapshot = new_snap

        return new_bank
    finally
        unlock(_CACHE_LOCK)
    end
end

# Helper to determine spacing type from grid type
@inline _spacing_type(::Type{X}) where {T, X<:AbstractRange{T}} = ScalarSpacing{T}
@inline _spacing_type(::Type{X}) where {T, X<:AbstractVector{T}} = VectorSpacing{T}

"""
Get or create a derivative BC cache bank for the given (T, L, R, X, S) combination.
"""
@inline function _get_derivative_bank(::X, ::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}}
    S = _spacing_type(X)
    EntryType = CacheEntry{T,L,R,X,S}
    return _get_bank(_DERIVATIVE_REGISTRY, CacheBank{EntryType})
end

"""
Get or create a periodic BC cache bank for the given (T, X, S) combination.
"""
@inline function _get_periodic_bank(::X) where {T<:AbstractFloat, X<:AbstractVector{T}}
    S = _spacing_type(X)
    EntryType = PeriodicCacheEntry{T,X,S}
    return _get_bank(_PERIODIC_REGISTRY, CacheBank{EntryType})
end

# ===============================================================
# Internal: RCU Lookup (Lock-Free Read Path)
# ===============================================================

"""
    _rcu_lookup(snap, id, x) -> cache or nothing

Lock-free cache lookup on an immutable snapshot.
Uses 2-pass algorithm with verification:
- Pass 1: objectid hint match → verify with isequal (catches in-place mutation)
- Pass 2: isequal fallback (different object, same content)

# Thread Safety
- This function is called on an immutable snapshot
- No self-healing (snapshot is immutable)
- Safe for concurrent calls from multiple threads

# Mutation Safety
Entry stores a snapshot of x (copy for Vector, same for Range).
Pass 1 verifies content even on objectid match to detect in-place mutation.
"""
@inline function _rcu_lookup(snap::BankSnapshot{E}, id::UInt, x::X) where {E, X}
    store = snap.store
    count = snap.count

    # [Pass 1] Identity hint check with verification
    # objectid match is a HINT that this might be the right entry.
    # We MUST verify content because user may have mutated x in-place.
    @inbounds for i in 1:count
        entry = store[i]
        if entry.id === id
            # Verify content matches snapshot (catches in-place mutation)
            if isequal(entry.x, x)
                return entry.cache
            end
            # objectid matched but content differs → mutation detected!
            # Continue to Pass 2 (might find another entry with matching content)
        end
    end

    # [Pass 2] Equality check fallback (no self-healing in RCU - snapshot is immutable)
    # This handles: different object with same content (cache hit)
    @inbounds for i in 1:count
        entry = store[i]
        if isequal(entry.x, x)
            return entry.cache
        end
    end

    return nothing
end

# ---------------------------------------------------------------
# Cache Builder (type-dispatched)
# ---------------------------------------------------------------

# Build cache for derivative BC entry
@inline function _build_cache(::Type{<:CacheEntry{T,L,R,X}}, x::X, bc::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}}
    return _build_derivative_bc_cache(x, bc.left, bc.right)
end

# Build cache for periodic BC entry
@inline function _build_cache(::Type{<:PeriodicCacheEntry{T,X}}, x::X, ::Nothing) where {T<:AbstractFloat, X<:AbstractVector{T}}
    return _build_periodic_cache(x)
end

# ---------------------------------------------------------------
# Unified Lookup/Insert (RCU Pattern)
# ---------------------------------------------------------------

"""
Core lookup/insert logic for CacheBank{E} using RCU pattern.

# Thread-Safety (RCU - Read-Copy-Update)
- Read path (hit): Lock-free via atomic snapshot load
- Write path (miss): Lock → copy snapshot → modify → atomic publish

# Performance
- Cache hit: ~11 ns (atomic load + linear scan, no lock)
- Cache miss: Lock overhead + O(N) copy (N ≤ 16, negligible)
"""
@inline function _lookup_or_insert!(bank::CacheBank{E}, x::X, bc_config) where {E<:AbstractCacheEntry, X}
    id = objectid(x)

    # === RCU Read Path (Lock-Free) ===
    snap = @atomic :acquire bank.snapshot
    found = _rcu_lookup(snap, id, x)
    found !== nothing && return found

    # === RCU Write Path (Lock + Copy-on-Write) ===
    lock(_CACHE_LOCK)
    try
        # Double-check after acquiring lock
        snap = @atomic :monotonic bank.snapshot
        found = _rcu_lookup(snap, id, x)
        found !== nothing && return found

        # Create snapshot of x to prevent external mutation from corrupting cache.
        # - Vector: copy to break aliasing (user can mutate original, snapshot is safe)
        # - AbstractRange: immutable in Julia, no copy needed (O(1) storage)
        snapshot_x = x isa Vector ? copy(x) : x

        # Build new cache using snapshot (ensures cache.x === snapshot_x)
        new_cache = _build_cache(E, snapshot_x, bc_config)
        new_entry = E(id, snapshot_x, new_cache)

        # Copy-on-write: create new snapshot with added entry
        new_store = copy(snap.store)
        new_count = snap.count
        new_ring = snap.ring

        if new_count < _CACHE_SIZE
            push!(new_store, new_entry)
            new_count += 1
        else
            # Ring buffer eviction
            new_store[new_ring] = new_entry
            new_ring = (new_ring % _CACHE_SIZE) + 1
        end

        # Atomic publish
        new_snap = BankSnapshot{E}(new_store, new_count, new_ring)
        @atomic :release bank.snapshot = new_snap

        return new_cache
    finally
        unlock(_CACHE_LOCK)
    end
end

# ===============================================================
# Internal API: _get_cubic_cache
# ===============================================================

"""
    _get_cubic_cache(x; bc=NaturalBC()) -> CubicSplineCache  [Internal]

Get or create a cached CubicSplineCache for the given x-grid.

# Warning: Internal API
This is an internal function. For interpolation, use the full API:
- `cubic_interp(x, y; bc=...)` - applies BC values correctly at solve time

Direct use of cached results with `cubic_interp(cache, y)` may give incorrect
results if BC values differ between cache creation and solve time (cache is
keyed by BC *type* only, not values). See `_solve_system!` 3-arg overload.

# Cache Sharing Behavior
Cache is keyed by BC *type*, not BC *values*.
`BCPair(Deriv1(0.0), Deriv2(0.0))` and `BCPair(Deriv1(1.0), Deriv2(2.0))` share the same cache
because they have the same type signature.

LU factorization depends only on matrix structure (x-grid + BC type),
not RHS values (y-data + BC values).
"""
@inline function _get_cubic_cache(x; bc::AbstractBC=NaturalBC())
    # Handle periodic BC
    if _is_periodic_bc(bc)
        return _get_periodic_cache_impl(x)
    end

    # Determine float type (convert Int to Float64)
    T = eltype(x)
    FT = T <: AbstractFloat ? T : Float64

    # Normalize BC to BCPair
    bc_pair = _normalize_bc(bc, FT)

    # Get bank and lookup
    return _get_derivative_cache_impl(x, bc_pair)
end

# ===============================================================
# Type-Stable Direct API (Internal - bypasses Union for zero-allocation)
# ===============================================================

# Typed BC API - direct path, no Union
@inline function _get_cubic_cache(x, ::NaturalBC)
    T = eltype(x)
    FT = T <: AbstractFloat ? T : Float64
    bc_pair = BCPair(Deriv2(zero(FT)), Deriv2(zero(FT)))
    return _get_derivative_cache_impl(x, bc_pair)
end

@inline function _get_cubic_cache(x, ::ClampedBC)
    T = eltype(x)
    FT = T <: AbstractFloat ? T : Float64
    bc_pair = BCPair(Deriv1(zero(FT)), Deriv1(zero(FT)))
    return _get_derivative_cache_impl(x, bc_pair)
end

@inline function _get_cubic_cache(x, ::PeriodicBC)
    return _get_periodic_cache_impl(x)
end

# BCPair direct API - type stable (used by internal paths)
@inline function _get_cubic_cache(x, bc::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    return _get_derivative_cache_impl(x, bc)
end

# PointBC convenience - convert to symmetric BCPair
@inline function _get_cubic_cache(x, bc::PointBC)
    T = eltype(x)
    FT = T <: AbstractFloat ? T : Float64
    bc_t = _promote_pointbc(bc, FT)
    bc_pair = BCPair(bc_t, bc_t)
    return _get_derivative_cache_impl(x, bc_pair)
end

@inline function _get_cubic_cache(
    x::AbstractVector{T},
    bc::AbstractBC,
    autocache::Bool
) where {T<:AbstractFloat}
    if autocache
        cache = _get_cubic_cache(x, bc)
    else
        cache = CubicSplineCache(x; bc=bc)
    end
    return cache
end


# ===============================================================
# Internal: Cache Implementation
# ===============================================================

# StepRangeLen concrete types (from `range(a, b, n)`)
const _StepRangeLen_F64 = StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
const _StepRangeLen_F32 = StepRangeLen{Float32, Float64, Float64, Int64}

"""
Internal implementation for derivative BC cache lookup.
"""
@inline function _get_derivative_cache_impl(x::AbstractVector{T}, bc_pair::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    x_normalized = x isa Vector ? x : collect(x)
    bank = _get_derivative_bank(x_normalized, bc_pair)
    return _lookup_or_insert!(bank, x_normalized, bc_pair)
end

@inline function _get_derivative_cache_impl(x::AbstractRange{T}, bc_pair::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    # Normalize to StepRangeLen for consistent cache key type.
    # LinRange and other Range types are converted (minor overhead on first call).
    x_normalized = (x isa _StepRangeLen_F64 || x isa _StepRangeLen_F32) ? x : range(first(x), last(x), length(x))
    bank = _get_derivative_bank(x_normalized, bc_pair)
    return _lookup_or_insert!(bank, x_normalized, bc_pair)
end

# Fallback: other Real types → convert to Float64
@inline function _get_derivative_cache_impl(x::AbstractVector{<:Real}, bc_pair::BCPair)
    x_float = Vector{Float64}(x)
    bc_float = _normalize_bc(bc_pair, Float64)
    _get_derivative_cache_impl(x_float, bc_float)
end

@inline function _get_derivative_cache_impl(x::AbstractRange{<:Real}, bc_pair::BCPair)
    x_float = range(Float64(first(x)), Float64(last(x)), length(x))
    bc_float = _normalize_bc(bc_pair, Float64)
    _get_derivative_cache_impl(x_float, bc_float)
end

"""
Internal implementation for periodic BC cache lookup.
"""
@inline function _get_periodic_cache_impl(x::AbstractVector{T}) where {T<:Union{Float64,Float32}}
    x_normalized = x isa Vector ? x : collect(x)
    bank = _get_periodic_bank(x_normalized)
    return _lookup_or_insert!(bank, x_normalized, nothing)
end

@inline function _get_periodic_cache_impl(x::AbstractRange{T}) where {T<:Union{Float64,Float32}}
    # Normalize to StepRangeLen for consistent cache key type.
    # LinRange and other Range types are converted (minor overhead on first call).
    x_normalized = (x isa _StepRangeLen_F64 || x isa _StepRangeLen_F32) ? x : range(first(x), last(x), length(x))
    bank = _get_periodic_bank(x_normalized)
    return _lookup_or_insert!(bank, x_normalized, nothing)
end

# Fallback: other Real types → convert to Float64
@inline function _get_periodic_cache_impl(x::AbstractVector{<:Real})
    _get_periodic_cache_impl(Vector{Float64}(x))
end

@inline function _get_periodic_cache_impl(x::AbstractRange{<:Real})
    _get_periodic_cache_impl(range(Float64(first(x)), Float64(last(x)), length(x)))
end
