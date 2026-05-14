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
abstract type AbstractCacheEntry{T <: AbstractFloat, X <: AbstractVector{T}} end

# Map user input grid type to the cache's wrapped axis type. Mirrors
# `_cache_axis(_convert_copy(x, T), bc)` (copy-then-wrap):
# - Non-periodic / `:inclusive`: Range → `_CachedRange{T, Tinv}`,
#   Vector → `_CachedVector{T, Tinv}`.
# - `:exclusive` periodic: extra `_ExclusivePeriodicAxis` wrapper around the
#   above inner.
# Cubic always promotes to `T <: AbstractFloat`, so `Tinv == T`; pin it
# explicitly to keep `EntryType` concrete in the bank.
@inline _cached_axis_type(::Type{<:AbstractRange}, ::Type{T}) where {T} = _CachedRange{T, T}
@inline _cached_axis_type(::Type{<:AbstractVector}, ::Type{T}) where {T} =
    _CachedVector{T, typeof(inv(oneunit(T)))}

@inline _cached_axis_type(::Type{X}, ::Type{T}, ::Val{:inclusive}) where {X, T} =
    _cached_axis_type(X, T)
@inline _cached_axis_type(::Type{X}, ::Type{T}, ::Val{:extended}) where {X, T} =
    _cached_axis_type(X, T)  # same axis shape as :inclusive
@inline function _cached_axis_type(::Type{X}, ::Type{T}, ::Val{:exclusive}) where {X, T}
    Inner = _cached_axis_type(X, T)
    return _ExclusivePeriodicAxis{T, Inner, T}
end

"""
    CacheEntry{T, L, R, X, S}

Cache entry for derivative BC (uses BCPair).

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `L`: Left boundary condition type (Deriv1{T} or Deriv2{T})
- `R`: Right boundary condition type (Deriv1{T} or Deriv2{T})
- `X`: Grid type (Vector{T} or StepRangeLen)
- `C`: Concrete `CubicSplineCache` parametrization

# Fields
- `id::UInt`: objectid of the ORIGINAL input x (hint for fast lookup)
- `x::X`: SNAPSHOT of the grid (copy for Vector, same object for Range)
- `cache`: CubicSplineCache built from the snapshot

# Mutation Safety
The `x` field stores a snapshot (copy) for Vector inputs, preventing external
mutation from corrupting the cache. Lookup verifies `isequal(entry.x, input_x)`
even on objectid match to detect in-place mutation.
"""
mutable struct CacheEntry{T <: AbstractFloat, L <: PointBC, R <: PointBC, X <: AbstractVector{T}, C <: CubicSplineCache{T, <:AbstractVector{T}, ThomasFactorization{T, Vector{T}}, BCPair{L, R}}} <: AbstractCacheEntry{T, X}
    id::UInt
    x::X                                  # user-input snapshot (Vector / Range)
    cache::C                              # concrete cache type — preserves wrapped-axis X for inference
end

"""
    PeriodicCacheEntry{T, X, E}

Cache entry for periodic BC (uses PeriodicData).

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `X`: Grid type (Vector{T} or StepRangeLen)
- `E`: Endpoint variant (`:inclusive`, `:exclusive`, or `:extended`) — encoded
  so the bank registry holds *separate* banks per variant. The cache content
  (Sherman-Morrison `q`, period, seam-cell width) differs between variants on
  the same grid object, so cache lookup must be partitioned to avoid mixing
  them. `:extended` is produced internally by `_bc_after_extend` and never
  appears in user-supplied BCs.

# Fields
- `id::UInt`: objectid of the ORIGINAL input x (hint for fast lookup)
- `x::X`: SNAPSHOT of the grid (copy for Vector, same object for Range)
- `cache`: CubicSplineCache built from the snapshot

# Mutation Safety
See `CacheEntry` documentation for details on mutation safety pattern.
"""
mutable struct PeriodicCacheEntry{T <: AbstractFloat, X <: AbstractVector{T}, E, C <: CubicSplineCache{T, <:AbstractVector{T}, ThomasFactorization{T, Vector{T}}, <:PeriodicBC}} <: AbstractCacheEntry{T, X}
    id::UInt
    x::X
    cache::C
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
struct BankSnapshot{E <: AbstractCacheEntry}
    store::Vector{E}
    count::Int
    ring::Int  # 1-based next eviction index
end

# Empty snapshot constructor
function BankSnapshot{E}() where {E <: AbstractCacheEntry}
    store = E[]
    sizehint!(store, _CACHE_SIZE)
    return BankSnapshot{E}(store, 0, 1)
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
mutable struct CacheBank{E <: AbstractCacheEntry}
    @atomic snapshot::BankSnapshot{E}
end

# Single constructor for all entry types
function CacheBank{E}() where {E <: AbstractCacheEntry}
    return CacheBank{E}(BankSnapshot{E}())
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
@inline function _get_bank(registry::GlobalRegistry, ::Type{CacheBank{E}}) where {E <: AbstractCacheEntry}
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

"""
Get or create a derivative BC cache bank for the given (T, L, R, X, S) combination.
Type-Free design: L, R are PointBC subtypes without type parameter constraint.
Accepts Type{X} to avoid needing an instance (eliminates collect() for views).
"""
@inline function _get_derivative_bank(::Type{X}, ::BCPair{L, R}) where {T <: AbstractFloat, L <: PointBC, R <: PointBC, X <: AbstractVector{T}}
    Xc = _cached_axis_type(X, T)
    Cc = CubicSplineCache{T, Xc, ThomasFactorization{T, Vector{T}}, BCPair{L, R}}
    EntryType = CacheEntry{T, L, R, X, Cc}
    return _get_bank(_DERIVATIVE_REGISTRY, CacheBank{EntryType})
end
# Instance convenience: forward to Type dispatch (used by tests / external callers)
@inline _get_derivative_bank(x::AbstractVector, bc::BCPair) = _get_derivative_bank(typeof(x), bc)

"""
Get or create a periodic BC cache bank for the given (T, X, E, C) combination.
Accepts Type{X} to avoid needing an instance (eliminates collect() for views).

`E` (`:inclusive`/`:exclusive`/`:extended`) is encoded in the entry type so
each endpoint variant lives in a *separate* bank for the same grid object —
their cache contents (cycle length, seam-cell width, Sherman-Morrison `q`)
differ. `:extended` is produced internally by `_bc_after_extend` for inputs
that were promoted from `:exclusive` at build time.

`C` (`check::Bool` of `PeriodicBC`) is also threaded into the bank key because
`_with_resolved_period` preserves it and `_bc_after_extend` flips it to `false`.
Without partitioning on `C`, a cache built from `check=false` BC would fail to
fit into a bank typed for `check=true`.
"""
@inline function _get_periodic_bank(::Type{X}, ::Val{E}, ::Val{C}) where {T <: AbstractFloat, X <: AbstractVector{T}, E, C}
    Xc = _cached_axis_type(X, T, Val(E))
    # `E` ∈ {:inclusive, :exclusive, :extended} — each gets its own bank.
    bc = PeriodicBC{E, T, C}
    Cc = CubicSplineCache{T, Xc, ThomasFactorization{T, Vector{T}}, bc}
    EntryType = PeriodicCacheEntry{T, X, E, Cc}
    return _get_bank(_PERIODIC_REGISTRY, CacheBank{EntryType})
end
# Instance convenience: forward to Type dispatch (reads C from the bc type-param)
@inline _get_periodic_bank(::Type{X}, ::PeriodicBC{E, P, C}) where {X, E, P, C} = _get_periodic_bank(X, Val(E), Val(C))
@inline _get_periodic_bank(x::AbstractVector, bc::PeriodicBC) = _get_periodic_bank(typeof(x), bc)

# ===============================================================
# Internal: RCU Lookup (Lock-Free Read Path)
# ===============================================================

"""
    _rcu_lookup(snap, id, x, bc_config) -> cache or nothing

Lock-free cache lookup on an immutable snapshot.
Uses 2-pass algorithm with verification:
- Pass 1: objectid hint match → verify with isequal + `_verify_cache_match`
  (catches in-place mutation AND BC mismatch)
- Pass 2: isequal + `_verify_cache_match` fallback (different object, same content)

The `bc_config` argument is the user-supplied BC for the lookup. It is passed to
`_verify_cache_match` so exclusive periodic caches can additionally check that
the cached `bc_config.period` matches the requested period — same `x` with two
distinct explicit periods (or auto-inferred vs explicit) must yield distinct
caches. For non-periodic / inclusive caches `_verify_cache_match` falls through
to `true`, so this argument is a no-op in those banks.

# Thread Safety
- This function is called on an immutable snapshot
- No self-healing (snapshot is immutable)
- Safe for concurrent calls from multiple threads

# Mutation Safety
Entry stores a snapshot of x (copy for Vector, same for Range).
Pass 1 verifies content even on objectid match to detect in-place mutation.
"""
@inline function _rcu_lookup(snap::BankSnapshot{E}, id::UInt, x::X, bc_config) where {E, X}
    store = snap.store
    count = snap.count

    # [Pass 1] Identity hint check with verification
    # objectid match is a HINT that this might be the right entry.
    # We MUST verify content because user may have mutated x in-place.
    # Periodic exclusive caches additionally verify the user-supplied period
    # against the cached `bc_config.period` (codex P1) — same `x` with two
    # different explicit periods must yield distinct caches.
    @inbounds for i in 1:count
        entry = store[i]
        if entry.id === id
            if isequal(entry.x, x) && _verify_cache_match(entry.cache, bc_config)
                return entry.cache
            end
        end
    end

    # [Pass 2] Equality check fallback (no self-healing in RCU - snapshot is immutable)
    # This handles: different object with same content (cache hit)
    @inbounds for i in 1:count
        entry = store[i]
        if isequal(entry.x, x) && _verify_cache_match(entry.cache, bc_config)
            return entry.cache
        end
    end

    return nothing
end

# BC-aware cache match verification. Default is `true` (objectid + isequal
# already adequate for non-periodic and inclusive caches). Exclusive periodic
# caches must additionally check that the user-supplied period matches the
# cached period — same `x` with two distinct explicit periods is a valid
# configuration and must produce two distinct caches.
@inline _verify_cache_match(::Any, ::Any) = true
@inline function _verify_cache_match(cache::CubicSplineCache, bc::PeriodicBC{:exclusive, P}) where {P}
    # `bc.period === Nothing` means "auto-infer from grid"; the requested
    # period is `step(x) * n_cells` for Range grids (the only shape allowed
    # to omit it). Comparing to `cache.bc.period` here prevents reusing
    # a stale cache built with an explicit period that passed the Range
    # tolerance but does not match the inferred value.
    #
    # IMPORTANT: `cache.x` is the WRAPPED axis with virtual length n+1
    # (`_ExclusivePeriodicAxis`), so `n_cells = length(cache.x) - 1`. Using
    # `length(cache.x)` directly (= n+1) inflates the requested period and
    # forces cache miss on every lookup → KB allocs per query.
    requested = if P === Nothing
        step(cache.x) * (length(cache.x) - 1)
    else
        bc.period
    end
    return cache.bc.period == requested
end

# ---------------------------------------------------------------
# Cache Builder (type-dispatched)
# ---------------------------------------------------------------

# Build cache for derivative BC entry
# Accepts AbstractVector (not x::X) so views can be passed directly;
# CubicSplineCache inner constructor handles copy() → Vector.
@inline function _build_cache(::Type{<:CacheEntry{T, L, R}}, x::AbstractVector{T}, bc::BCPair{L, R}) where {T <: AbstractFloat, L <: PointBC, R <: PointBC}
    return _build_derivative_bc_cache(x, bc.left, bc.right)
end

# Non-AbstractFloat Real input (Int, Rational): convert to float, build cache.
# Only called on cache miss — subsequent hits are zero-alloc.
@inline function _build_cache(::Type{<:CacheEntry{T, L, R}}, x::AbstractVector, bc::BCPair{L, R}) where {T <: AbstractFloat, L <: PointBC, R <: PointBC}
    return _build_derivative_bc_cache(_to_float(x, T), bc.left, bc.right)
end

# Build cache for periodic BC entry. The `bc::PeriodicBC{E}` argument is required
# (not optional `Nothing`) because cache content is BC-form-dependent: cycle
# length, period, and seam-cell width all differ between `:inclusive` and
# `:exclusive`. The entry's E type-param matches `bc`'s E by dispatch.
@inline function _build_cache(::Type{<:PeriodicCacheEntry{T, X, E}}, x::AbstractVector{T}, bc::PeriodicBC{E}) where {T <: AbstractFloat, X, E}
    return _build_periodic_cache(x, bc)
end

# Non-AbstractFloat Real input: convert to float for periodic cache.
@inline function _build_cache(::Type{<:PeriodicCacheEntry{T, X, E}}, x::AbstractVector, bc::PeriodicBC{E}) where {T <: AbstractFloat, X, E}
    return _build_periodic_cache(_to_float(x, T), bc)
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
@inline function _lookup_or_insert!(bank::CacheBank{E}, x::X, bc_config) where {E <: AbstractCacheEntry, X}
    id = objectid(x)

    # === RCU Read Path (Lock-Free) ===
    snap = @atomic :acquire bank.snapshot
    found = _rcu_lookup(snap, id, x, bc_config)
    found !== nothing && return found

    # === RCU Write Path (Lock + Copy-on-Write) ===
    lock(_CACHE_LOCK)
    try
        # Double-check after acquiring lock
        snap = @atomic :monotonic bank.snapshot
        found = _rcu_lookup(snap, id, x, bc_config)
        found !== nothing && return found

        # Build cache — the builder wraps the user's x into the cached/wrapped
        # axis form via `_cache_axis(_convert_copy(x, T), bc)`. Snapshot the *raw*
        # user input on the entry so `isequal(entry.x, input_x)` lookups remain comparable.
        new_cache = _build_cache(E, x, bc_config)
        new_entry = E(id, copy(x), new_cache)

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
    _get_cubic_cache(x; bc=CubicFit()) -> CubicSplineCache  [Internal]

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
@inline function _get_cubic_cache(x; bc::AbstractBC = CubicFit())
    xp = _resolve_axis(x)
    # Handle periodic BC. `bc` carries the endpoint variant (E type-param) which
    # the periodic pool uses to partition inclusive/exclusive caches.
    if bc isa PeriodicBC
        return _get_periodic_cache_impl(xp, bc)
    end

    # Normalize BC to BCPair and route to cache
    bc_pair = _normalize_bc(bc)
    return _get_derivative_cache_impl(xp, bc_pair)
end

# ===============================================================
# Type-Stable Direct API (Internal - bypasses Union for zero-allocation)
# ===============================================================

# Helper: resolve cache float type.
# Float64→Float64, Float32→Float32, Int→Float64, Dual→Dual (duck passthrough).
@inline _cache_float_type(::Type{T}) where {T} =
    T <: _PromotableValue ? float(T) : T

# Typed BC API - direct path, no Union
# _resolve_axis: Vector as-is, Range → _CachedRange{float(T)} (normalizes Int Range).
@inline function _get_cubic_cache(x, ::ZeroCurvBC)
    FT = _cache_float_type(eltype(x))
    return _get_derivative_cache_impl(_resolve_axis(x), BCPair(Deriv2(zero(FT)), Deriv2(zero(FT))))
end

@inline function _get_cubic_cache(x, ::ZeroSlopeBC)
    FT = _cache_float_type(eltype(x))
    return _get_derivative_cache_impl(_resolve_axis(x), BCPair(Deriv1(zero(FT)), Deriv1(zero(FT))))
end

@inline function _get_cubic_cache(x, bc::PeriodicBC)
    return _get_periodic_cache_impl(_resolve_axis(x), bc)
end

# BCPair: convert to cache-compatible form, route to cache impl.
@inline function _get_cubic_cache(x::AbstractVector, bc::BCPair{L, R}) where {L <: PointBC, R <: PointBC}
    FT = _cache_float_type(eltype(x))
    bc_cache = _cache_bc_pair(bc, FT)
    return _get_derivative_cache_impl(_resolve_axis(x), bc_cache)
end

# PointBC convenience - convert to symmetric BCPair
@inline function _get_cubic_cache(x, bc::PointBC)
    FT = _cache_float_type(eltype(x))
    bc_c = _cache_pointbc(bc, FT)
    return _get_derivative_cache_impl(_resolve_axis(x), BCPair(bc_c, bc_c))
end

# BCPair + autocache API.
# _resolve_axis normalizes Range → _CachedRange (stack); Vector passes as-is.
# autocache=true → pool lookup, false → build fresh.
@inline function _get_cubic_cache(
        x::AbstractVector,
        bc::BCPair{L, R},
        autocache::Bool
    ) where {L <: PointBC, R <: PointBC}
    FT = _cache_float_type(eltype(x))
    bc_cache = _cache_bc_pair(bc, FT)
    x_norm = _resolve_axis(x)
    if autocache
        return _get_derivative_cache_impl(x_norm, bc_cache)
    else
        return _build_derivative_bc_cache(_to_float(x_norm, FT), bc_cache.left, bc_cache.right)
    end
end

@inline function _get_cubic_cache(
        x::AbstractVector,
        bc::AbstractBC,
        autocache::Bool
    )
    x_norm = _resolve_axis(x)
    if autocache
        return _get_cubic_cache(x_norm, bc)
    else
        # Bypass the public `CubicSplineCache(x; bc=bc)` outer constructor —
        # it rejects `:exclusive` PeriodicBC for direct user use, but the
        # internal `autocache=false` path is safe (oneshot / persistent
        # callers thread `bc` into the searcher via `_resolve_search`).
        FT = _cache_float_type(eltype(x))
        x_typed = _to_float(x_norm, FT)
        if _is_periodic_bc(bc)
            return _build_periodic_cache(x_typed, bc)
        else
            bc_normalized = _normalize_bc(bc)
            return _build_derivative_bc_cache(x_typed, bc_normalized.left, bc_normalized.right)
        end
    end
end


# ===============================================================
# Internal: Cache Implementation
# ===============================================================
#
# Bank type X matches: _CachedRange{T} bank or Vector{T} bank — no Union, no dispatch explosion.

"""
Internal implementation for derivative BC cache lookup.
Type-Free design: bc_pair should already be cache-compatible (via _cache_bc_pair).
"""
@inline function _get_derivative_cache_impl(x::AbstractVector{T}, bc_pair::BCPair{L, R}) where {T <: AbstractFloat, L <: PointBC, R <: PointBC}
    # Bank always keyed on Vector{T} (views/SubArrays share the same bank).
    # No collect() needed: isequal(::Vector, ::SubArray) compares element-wise,
    # and CubicSplineCache inner constructor copies → Vector on miss.
    bank = _get_derivative_bank(Vector{T}, bc_pair)
    return _lookup_or_insert!(bank, x, bc_pair)
end

# _CachedRange: bank keyed on _CachedRange{T, T} (Tinv == T for Float grids).
# objectid is deterministic for isbits → fast hit.
@inline function _get_derivative_cache_impl(x::_CachedRange{T, T}, bc_pair::BCPair{L, R}) where {T <: AbstractFloat, L <: PointBC, R <: PointBC}
    bank = _get_derivative_bank(_CachedRange{T, T}, bc_pair)
    return _lookup_or_insert!(bank, x, bc_pair)
end

# AbstractRange fallback: normalize via _to_float → _CachedRange dispatch above.
@inline function _get_derivative_cache_impl(x::AbstractRange{T}, bc_pair::BCPair{L, R}) where {T <: AbstractFloat, L <: PointBC, R <: PointBC}
    return _get_derivative_cache_impl(_to_float(x, T), bc_pair)
end

# Integer grids: look up in the Float64 bank directly.
# isequal(Float64_entry, Int_input) = true, so cache hit works cross-type.
# On miss, _build_cache converts to float via CubicSplineCache constructor.
@inline function _get_derivative_cache_impl(x::AbstractVector{T}, bc_pair::BCPair{L, R}) where {T <: Integer, L <: PointBC, R <: PointBC}
    bank = _get_derivative_bank(Vector{float(T)}, bc_pair)
    return _lookup_or_insert!(bank, x, bc_pair)
end

# Rational grids: same cross-type lookup pattern.
@inline function _get_derivative_cache_impl(x::AbstractVector{T}, bc_pair::BCPair{L, R}) where {T <: Rational, L <: PointBC, R <: PointBC}
    bank = _get_derivative_bank(Vector{float(T)}, bc_pair)
    return _lookup_or_insert!(bank, x, bc_pair)
end

# Duck-typed grids (Dual, etc.): build fresh, no caching.
# These are ephemeral — cache hit rate ≈ 0%.
@inline function _get_derivative_cache_impl(x::AbstractVector, bc_pair::BCPair{L, R}) where {L <: PointBC, R <: PointBC}
    FT = _cache_float_type(eltype(x))
    return _build_derivative_bc_cache(_to_float(x, FT), bc_pair.left, bc_pair.right)
end

"""
Internal implementation for periodic BC cache lookup. `bc` is threaded through
so `_get_periodic_bank` selects the right E-variant bank — `:inclusive`,
`:exclusive`, and `:extended` caches partition into separate banks (see
`PeriodicCacheEntry`).
"""
@inline function _get_periodic_cache_impl(x::AbstractVector{T}, bc::PeriodicBC) where {T <: AbstractFloat}
    bank = _get_periodic_bank(Vector{T}, bc)
    return _lookup_or_insert!(bank, x, bc)
end

# _CachedRange: bank keyed on _CachedRange{T, T} (Tinv == T for Float grids).
@inline function _get_periodic_cache_impl(x::_CachedRange{T, T}, bc::PeriodicBC) where {T <: AbstractFloat}
    bank = _get_periodic_bank(_CachedRange{T, T}, bc)
    return _lookup_or_insert!(bank, x, bc)
end

# AbstractRange fallback: normalize via _to_float → _CachedRange dispatch above.
@inline function _get_periodic_cache_impl(x::AbstractRange{T}, bc::PeriodicBC) where {T <: AbstractFloat}
    return _get_periodic_cache_impl(_to_float(x, T), bc)
end

# Integer grids: look up in the Float64 bank directly.
@inline function _get_periodic_cache_impl(x::AbstractVector{T}, bc::PeriodicBC) where {T <: Integer}
    bank = _get_periodic_bank(Vector{float(T)}, bc)
    return _lookup_or_insert!(bank, x, bc)
end

# Rational grids: same pattern.
@inline function _get_periodic_cache_impl(x::AbstractVector{T}, bc::PeriodicBC) where {T <: Rational}
    bank = _get_periodic_bank(Vector{float(T)}, bc)
    return _lookup_or_insert!(bank, x, bc)
end

# Duck-typed grids (Dual, etc.): build fresh, no caching.
@inline function _get_periodic_cache_impl(x::AbstractVector, bc::PeriodicBC)
    return _build_periodic_cache(_to_float(x, _cache_float_type(eltype(x))), bc)
end
