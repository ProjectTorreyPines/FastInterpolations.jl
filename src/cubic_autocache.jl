"""
    cubic_autocache.jl

Automatic caching layer for cubic spline interpolation.
Transparently reuses LU factorization for repeated x-grids.

# Implementation Details

- Generic `CacheBank{E}` - single bank type parameterized by entry type
- Zero-allocation cache hit via 2-pass lookup (objectid → isequal)
- Ring buffer eviction for O(1) cache replacement
- Self-healing: updates objectid on content match for future fast-path hits
- Dynamic bank creation via IdDict keyed by type
- Thread-safe with ReentrantLock on cache access
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
    CacheEntry{T, L, R, X}

Cache entry for derivative BC (uses BCPair).

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `L`: Left boundary condition type (Deriv1{T} or Deriv2{T})
- `R`: Right boundary condition type (Deriv1{T} or Deriv2{T})
- `X`: Grid type (Vector{T} or StepRangeLen)
"""
mutable struct CacheEntry{T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}} <: AbstractCacheEntry{T, X}
    id::UInt
    x::X
    cache::CubicSplineCache{T, X, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, BCPair{T,L,R}}
end

"""
    PeriodicCacheEntry{T, X}

Cache entry for periodic BC (uses PeriodicData).
"""
mutable struct PeriodicCacheEntry{T<:AbstractFloat, X<:AbstractVector{T}} <: AbstractCacheEntry{T, X}
    id::UInt
    x::X
    cache::CubicSplineCache{T, X, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, PeriodicData{T}}
end

# ===============================================================
# Generic Cache Bank
# ===============================================================

"""
    CacheBank{E}

A generic cache bank holding entries of type `E`.
Bank is parameterized by entry type, not by BC types directly.

# Type Parameters
- `E`: Entry type (CacheEntry{T,L,R,X} or PeriodicCacheEntry{T,X})
"""
mutable struct CacheBank{E<:AbstractCacheEntry}
    store::Vector{E}
    ring::Base.RefValue{Int}
end

# Single constructor for all entry types
function CacheBank{E}() where {E<:AbstractCacheEntry}
    store = E[]
    sizehint!(store, _CACHE_SIZE)  # Pre-allocate to prevent push! reallocation race
    CacheBank{E}(store, Ref(1))
end

# ===============================================================
# Bank Registry (Dynamic/Lazy)
# ===============================================================

# NOTE: Value type is `Any` because banks are created dynamically for each
# entry type combination at runtime. Type-stable access is ensured by
# the typed getter functions (_get_derivative_bank, _get_periodic_bank).
const _DERIVATIVE_BANK_REGISTRY = IdDict{DataType, Any}()
const _PERIODIC_BANK_REGISTRY = IdDict{DataType, Any}()
const _CACHE_LOCK = ReentrantLock()

# Load-time constant: immutable after package load, enables compiler optimizations
# Change via set_cubic_cache_size!(n) then restart Julia
const _CACHE_SIZE = @load_preference("cache_size", 16)::Int

# ===============================================================
# Module Initialization (Thread-Safety)
# ===============================================================

"""
    __init__()

Module initialization function called at `using` time (not precompile time).
Pre-allocates registry capacity to prevent IdDict resize race during concurrent bank creation.

# Thread-Safety
- IdDict resizes at ~33% load factor
- `sizehint!(100)` → 256 slots → safe for ~85 entries
- Realistic usage: <20 different type combinations
"""
function __init__()
    # Pre-allocate registry capacity to prevent resize race during bank creation.
    # IdDict resizes at ~33% load factor; sizehint!(100) → 256 slots → ~85 safe entries.
    # Realistic usage: <20 different type combinations.
    sizehint!(_DERIVATIVE_BANK_REGISTRY, 100)
    sizehint!(_PERIODIC_BANK_REGISTRY, 100)
end

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
    @info "Cache size set to $n. Restart Julia to apply."
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
"""
function clear_cubic_cache!()
    lock(_CACHE_LOCK) do
        empty!(_DERIVATIVE_BANK_REGISTRY)
        empty!(_PERIODIC_BANK_REGISTRY)
    end
    return nothing
end

# ===============================================================
# Internal: Bank Retrieval
# ===============================================================

"""
Generic bank retrieval with DCL pattern.

# Thread-Safety
- Single-thread: Lock-free (no overhead)
- Multi-thread: Double-checked locking for safe bank creation

# Note
`CacheBank{E}()` calls constructor which includes sizehint! initialization.
"""
@inline function _get_bank(registry::IdDict, ::Type{CacheBank{E}}) where {E<:AbstractCacheEntry}
    BankType = CacheBank{E}
    if Threads.nthreads() == 1
        # Single-thread: Lock-free
        bank = get(registry, BankType, nothing)
        if bank === nothing
            bank = CacheBank{E}()
            registry[BankType] = bank
        end
        return bank::BankType
    else
        # Multi-thread: DCL (optimistic read → lock → double-check)
        bank = get(registry, BankType, nothing)
        bank !== nothing && return bank::BankType

        lock(_CACHE_LOCK)
        try
            bank = get(registry, BankType, nothing)
            if bank === nothing
                bank = CacheBank{E}()
                registry[BankType] = bank
            end
            return bank::BankType
        finally
            unlock(_CACHE_LOCK)
        end
    end
end

"""
Get or create a derivative BC cache bank for the given (T, L, R, X) combination.
"""
@inline function _get_derivative_bank(::X, ::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}}
    EntryType = CacheEntry{T,L,R,X}
    return _get_bank(_DERIVATIVE_BANK_REGISTRY, CacheBank{EntryType})
end

"""
Get or create a periodic BC cache bank for the given (T, X) combination.
"""
@inline function _get_periodic_bank(::X) where {T<:AbstractFloat, X<:AbstractVector{T}}
    EntryType = PeriodicCacheEntry{T,X}
    return _get_bank(_PERIODIC_BANK_REGISTRY, CacheBank{EntryType})
end

# ===============================================================
# Internal: Ring Buffer Add
# ===============================================================

@inline function _add_to_ring!(store::Vector{E}, ring::Base.RefValue{Int}, entry::E) where E
    if length(store) < _CACHE_SIZE
        push!(store, entry)
    else
        idx = ring[]
        store[idx] = entry
        ring[] = (idx % _CACHE_SIZE) + 1
    end
    return nothing
end

# ===============================================================
# Internal: Lookup/Insert
# ===============================================================

"""
Cache lookup (2-pass: identity check → equality check).
Self-healing enabled only in single-thread mode to avoid write race.
"""
@inline function _lookup(store::Vector{E}, id::UInt, x::X) where {E, X}
    # [Pass 1] Identity check (fast path)
    @inbounds for entry in store
        if entry.id === id
            return entry.cache
        end
    end
    # [Pass 2] Equality check (+ self-healing in single-thread)
    @inbounds for entry in store
        if isequal(entry.x, x)
            if Threads.nthreads() == 1
                entry.id = id  # Self-healing for future fast-path
            end
            return entry.cache
        end
    end
    return nothing
end

# ---------------------------------------------------------------
# Cache Builder (type-dispatched)
# ---------------------------------------------------------------

# Build cache for derivative BC entry
@inline function _build_cache(::Type{CacheEntry{T,L,R,X}}, x::X, bc::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}}
    return _build_derivative_bc_cache(x, bc.left, bc.right)
end

# Build cache for periodic BC entry
@inline function _build_cache(::Type{PeriodicCacheEntry{T,X}}, x::X, ::Nothing) where {T<:AbstractFloat, X<:AbstractVector{T}}
    return _build_periodic_cache(x)
end

# ---------------------------------------------------------------
# Unified Lookup/Insert
# ---------------------------------------------------------------

"""
Core lookup/insert logic for CacheBank{E}.

# Thread-Safety
- Single-thread: Lock-free lookup with self-healing, direct insert
- Multi-thread: Coarse-grained locking (lookup + build + insert under single lock)
"""
@inline function _lookup_or_insert!(bank::CacheBank{E}, x::X, bc_data) where {E<:AbstractCacheEntry, X}
    store = bank.store  # Direct field access (type-stable)
    ring = bank.ring
    id = objectid(x)

    if Threads.nthreads() == 1
        # Single-thread: Lock-free
        found = _lookup(store, id, x)
        found !== nothing && return found

        # Cache miss - build & insert
        new_cache = _build_cache(E, x, bc_data)
        new_entry = E(id, x, new_cache)  # Direct constructor call
        _add_to_ring!(store, ring, new_entry)
        return new_cache
    else
        # Multi-thread: Coarse-grained locking
        lock(_CACHE_LOCK)
        try
            found = _lookup(store, id, x)
            found !== nothing && return found

            new_cache = _build_cache(E, x, bc_data)
            new_entry = E(id, x, new_cache)
            _add_to_ring!(store, ring, new_entry)
            return new_cache
        finally
            unlock(_CACHE_LOCK)
        end
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
