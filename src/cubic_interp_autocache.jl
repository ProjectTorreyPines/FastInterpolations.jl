"""
    cubic_interp_autocache.jl

Automatic caching layer for cubic spline interpolation.
Transparently reuses LU factorization for repeated x-grids.

# Implementation Details

- Zero-allocation cache hit via 2-pass lookup (objectid → isequal)
- Ring buffer eviction for O(1) cache replacement
- Self-healing: updates objectid on content match for future fast-path hits
- Type-parametric entries for Float32/Float64 compatibility
- Defensive x-grid copying in CubicSplineCache constructor
- Thread-safe with ReentrantLock on cache access
"""

# ===============================================================
# Cache Infrastructure
# ===============================================================

# Concrete LU factorization types (for zero-allocation cache lookup)
# The LU type is always: LU{T, Tridiagonal{T, Vector{T}}, Vector{Int64}}
const _LU_Type_F64 = LinearAlgebra.LU{Float64, LinearAlgebra.Tridiagonal{Float64, Vector{Float64}}, Vector{Int64}}
const _LU_Type_F32 = LinearAlgebra.LU{Float32, LinearAlgebra.Tridiagonal{Float32, Vector{Float32}}, Vector{Int64}}

# Concrete CubicSplineCache types (fully specified for type stability)
const _CubicSplineCache_F64 = CubicSplineCache{Float64, _LU_Type_F64}
const _CubicSplineCache_F32 = CubicSplineCache{Float32, _LU_Type_F32}

# Mutable cache entry for self-healing objectid updates
# Uses concrete spline types to avoid boxing (32 bytes → 0 bytes)
mutable struct CacheEntryF64
    id::UInt                       # objectid(x) - Fast Path lookup
    x::Vector{Float64}             # Original x - Slow Path lookup
    spline::_CubicSplineCache_F64  # Concrete type for zero-allocation
end

mutable struct CacheEntryF32
    id::UInt
    x::Vector{Float32}
    spline::_CubicSplineCache_F32
end

# Global cache stores: Type-stable vectors for each supported float type
# Separate vectors avoid type instability during iteration (critical for zero-allocation)
const _CUBIC_CACHE_STORE_F64 = Vector{CacheEntryF64}()
const _CUBIC_CACHE_STORE_F32 = Vector{CacheEntryF32}()
const _CACHE_LOCK = ReentrantLock()
const _CACHE_SIZE_REF = Ref{Int}(16)  # Max cache entries (shared across all types)
const _RING_IDX_F64 = Ref{Int}(1)     # Ring buffer pointer for Float64
const _RING_IDX_F32 = Ref{Int}(1)     # Ring buffer pointer for Float32

# Statistics (individual Refs to avoid NamedTuple allocation on update)
const _CACHE_HITS = Ref{Int}(0)
const _CACHE_MISSES = Ref{Int}(0)
const _CACHE_EVICTIONS = Ref{Int}(0)

# ===============================================================
# Public API
# ===============================================================

"""
    set_cubic_cache_size!(n::Int)

Set maximum number of cached x-grids.

# Arguments
- `n::Int`: Maximum cache entries (default: 16)

# Example
```julia
set_cubic_cache_size!(32)  # Increase cache size
```
"""
function set_cubic_cache_size!(n::Int)
    n > 0 || throw(ArgumentError("Cache size must be positive"))
    _CACHE_SIZE_REF[] = n
    return n
end

"""
    get_cubic_cache_size()

Get current maximum cache size.
"""
get_cubic_cache_size() = _CACHE_SIZE_REF[]

"""
    clear_cubic_cache!()

Clear all cached x-grids and reset statistics.

Useful for benchmarking or memory management.

# Example
```julia
clear_cubic_cache!()
```
"""
function clear_cubic_cache!()
    lock(_CACHE_LOCK)
    try
        empty!(_CUBIC_CACHE_STORE_F64)
        empty!(_CUBIC_CACHE_STORE_F32)
        _RING_IDX_F64[] = 1
        _RING_IDX_F32[] = 1
        _CACHE_HITS[] = 0
        _CACHE_MISSES[] = 0
        _CACHE_EVICTIONS[] = 0
    finally
        unlock(_CACHE_LOCK)
    end
    return nothing
end

"""
    cubic_cache_stats()

Return cache hit/miss statistics for debugging.

# Returns
- `NamedTuple`: (hits, misses, evictions, collisions, size, efficiency)

# Example
```julia
stats = cubic_cache_stats()
println("Cache efficiency: \$(stats.efficiency)%")
```
"""
function cubic_cache_stats()
    hits = _CACHE_HITS[]
    misses = _CACHE_MISSES[]
    evictions = _CACHE_EVICTIONS[]

    local total_size
    lock(_CACHE_LOCK)
    try
        total_size = length(_CUBIC_CACHE_STORE_F64) + length(_CUBIC_CACHE_STORE_F32)
    finally
        unlock(_CACHE_LOCK)
    end

    total = hits + misses
    efficiency = total > 0 ? round(100 * hits / total, digits=1) : 0.0

    return (
        hits = hits,
        misses = misses,
        evictions = evictions,
        collisions = 0,  # No longer tracked (no hash buckets)
        size = total_size,
        efficiency = efficiency
    )
end

# ===============================================================
# Internal Cache Management
# ===============================================================

# Type-specific cache accessors (for type stability)
@inline _get_cache_store(::Type{Float64}) = _CUBIC_CACHE_STORE_F64
@inline _get_cache_store(::Type{Float32}) = _CUBIC_CACHE_STORE_F32
@inline _get_ring_idx(::Type{Float64}) = _RING_IDX_F64
@inline _get_ring_idx(::Type{Float32}) = _RING_IDX_F32

"""
    add_to_cache_f64!(entry::CacheEntryF64)

Add Float64 entry to cache with ring buffer eviction.
O(1) eviction by overwriting oldest slot.
Must be called within lock.
"""
function add_to_cache_f64!(entry::CacheEntryF64)
    store = _CUBIC_CACHE_STORE_F64
    ring_idx = _RING_IDX_F64
    max_size = _CACHE_SIZE_REF[]

    if length(store) < max_size
        push!(store, entry)
    else
        idx = ring_idx[]
        store[idx] = entry
        ring_idx[] = (idx % max_size) + 1
        _CACHE_EVICTIONS[] += 1
    end
end

"""
    add_to_cache_f32!(entry::CacheEntryF32)

Add Float32 entry to cache with ring buffer eviction.
"""
function add_to_cache_f32!(entry::CacheEntryF32)
    store = _CUBIC_CACHE_STORE_F32
    ring_idx = _RING_IDX_F32
    max_size = _CACHE_SIZE_REF[]

    if length(store) < max_size
        push!(store, entry)
    else
        idx = ring_idx[]
        store[idx] = entry
        ring_idx[] = (idx % max_size) + 1
        _CACHE_EVICTIONS[] += 1
    end
end

"""
    get_cubic_cache(x::AbstractVector{T}) where {T<:AbstractFloat}

Internal: Lookup or create cache for x-grid with zero-allocation hit.

Thread-safe with ReentrantLock. Returns existing cache if x matches
(using 2-pass lookup: objectid → isequal), otherwise creates new cache.

# Implementation Details

- **Pass 1 (Fast Path)**: objectid(x) comparison - O(1), zero allocation
- **Pass 2 (Slow Path)**: isequal(x, entry.x) - O(N), zero allocation
- **Self-Healing**: Updates entry.id on Pass 2 hit for future fast-path
- **Ring Buffer**: O(1) eviction by circular index overwrite
- **Type-stable**: Separate stores for Float64/Float32 avoid boxing
- **Defensive copy**: x is copied via collect() in CubicSplineCache constructor
"""
function get_cubic_cache(x::AbstractVector{Float64})
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_F64

    lock(_CACHE_LOCK)
    try
        # [Pass 1] Identity Check (Ultra Fast Path)
        # O(K) where K ≤ 16, just integer comparison
        @inbounds for entry in store
            if entry.id === id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Pass 2] Equality Check (Slow Path)
        # O(K×N) where K ≤ 16, N = length(x)
        @inbounds for entry in store
            if isequal(entry.x, x)
                # Self-Healing: Update ID for next call to hit Pass 1
                entry.id = id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Cache Miss] Create and register new cache
        _CACHE_MISSES[] += 1
        new_spline = CubicSplineCache(x)  # Defensive copy via collect(x)
        new_entry = CacheEntryF64(id, collect(x), new_spline)
        add_to_cache_f64!(new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

function get_cubic_cache(x::AbstractVector{Float32})
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_F32

    lock(_CACHE_LOCK)
    try
        # [Pass 1] Identity Check (Ultra Fast Path)
        @inbounds for entry in store
            if entry.id === id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Pass 2] Equality Check (Slow Path)
        @inbounds for entry in store
            if isequal(entry.x, x)
                entry.id = id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Cache Miss]
        _CACHE_MISSES[] += 1
        new_spline = CubicSplineCache(x)
        new_entry = CacheEntryF32(id, collect(x), new_spline)
        add_to_cache_f32!(new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

# Fallback for other AbstractFloat types (non-optimized path)
function get_cubic_cache(x::AbstractVector{T}) where {T<:AbstractFloat}
    # Convert to Float64 for unsupported types
    x_f64 = convert(Vector{Float64}, x)
    return get_cubic_cache(x_f64)
end
