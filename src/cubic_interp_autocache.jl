"""
    cubic_interp_autocache.jl

Automatic caching layer for cubic spline interpolation.
Transparently reuses LU factorization for repeated x-grids.

# Implementation Details

- Zero-allocation cache hit via 2-pass lookup (objectid → isequal)
- Ring buffer eviction for O(1) cache replacement
- Self-healing: updates objectid on content match for future fast-path hits
- Type-parametric entries for Float32/Float64 compatibility
- Thread-safe with ReentrantLock on cache access
"""

# ===============================================================
# Cache Infrastructure
# ===============================================================

# Concrete LU factorization types (for zero-allocation cache lookup)
# The LU type is always: LU{T, Tridiagonal{T, Vector{T}}, Vector{Int64}}
const _LU_Type_F64 = LinearAlgebra.LU{Float64, LinearAlgebra.Tridiagonal{Float64, Vector{Float64}}, Vector{Int64}}
const _LU_Type_F32 = LinearAlgebra.LU{Float32, LinearAlgebra.Tridiagonal{Float32, Vector{Float32}}, Vector{Int64}}

# StepRangeLen concrete types (from `range(a, b, n)`)
# These are the canonical Range types we optimize for
const _StepRangeLen_F64 = StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
const _StepRangeLen_F32 = StepRangeLen{Float32, Float64, Float64, Int64}

# Concrete CubicSplineCache types for Vector-based caches (Natural BC)
# Using concrete types ensures zero-allocation on cache hit
# 4th type parameter: Nothing for natural BC
const _CubicSplineCache_Vec_F64 = CubicSplineCache{Float64, Vector{Float64}, _LU_Type_F64, Nothing}
const _CubicSplineCache_Vec_F32 = CubicSplineCache{Float32, Vector{Float32}, _LU_Type_F32, Nothing}

# Concrete CubicSplineCache types for Range-based caches (O(1) index lookup!, Natural BC)
const _CubicSplineCache_Range_F64 = CubicSplineCache{Float64, _StepRangeLen_F64, _LU_Type_F64, Nothing}
const _CubicSplineCache_Range_F32 = CubicSplineCache{Float32, _StepRangeLen_F32, _LU_Type_F32, Nothing}

# Concrete CubicSplineCache types for Periodic BC
const _CubicSplineCache_Vec_F64_Periodic = CubicSplineCache{Float64, Vector{Float64}, _LU_Type_F64, PeriodicData{Float64}}
const _CubicSplineCache_Vec_F32_Periodic = CubicSplineCache{Float32, Vector{Float32}, _LU_Type_F32, PeriodicData{Float32}}
const _CubicSplineCache_Range_F64_Periodic = CubicSplineCache{Float64, _StepRangeLen_F64, _LU_Type_F64, PeriodicData{Float64}}
const _CubicSplineCache_Range_F32_Periodic = CubicSplineCache{Float32, _StepRangeLen_F32, _LU_Type_F32, PeriodicData{Float32}}

# ═══════════════════════════════════════════════════════════════════════
# Cache Entries - Separate for Vector and Range to maintain type stability
# ═══════════════════════════════════════════════════════════════════════

# Vector-based cache entries (O(log n) binary search during interpolation)
mutable struct CacheEntryVecF64
    id::UInt                          # objectid(x) - Fast Path lookup
    x::Vector{Float64}                # Concrete Vector type
    spline::_CubicSplineCache_Vec_F64 # Concrete spline type for zero-allocation
end

mutable struct CacheEntryVecF32
    id::UInt
    x::Vector{Float32}
    spline::_CubicSplineCache_Vec_F32
end

# Range-based cache entries (O(1) direct index calculation during interpolation!)
mutable struct CacheEntryRangeF64
    id::UInt                            # objectid(x) - Fast Path lookup
    x::_StepRangeLen_F64                # Concrete Range type
    spline::_CubicSplineCache_Range_F64 # Range-based spline for O(1) lookup
end

mutable struct CacheEntryRangeF32
    id::UInt
    x::_StepRangeLen_F32
    spline::_CubicSplineCache_Range_F32
end

# ═══════════════════════════════════════════════════════════════════════
# Global Cache Stores
# ═══════════════════════════════════════════════════════════════════════
# Separate stores for Vector and Range to maintain type stability
# Type-stable vectors avoid boxing during iteration (critical for zero-allocation)

# Vector-based stores (for non-uniform grids)
const _CUBIC_CACHE_STORE_VEC_F64 = Vector{CacheEntryVecF64}()
const _CUBIC_CACHE_STORE_VEC_F32 = Vector{CacheEntryVecF32}()

# Range-based stores (for uniform grids - O(1) index lookup!)
const _CUBIC_CACHE_STORE_RANGE_F64 = Vector{CacheEntryRangeF64}()
const _CUBIC_CACHE_STORE_RANGE_F32 = Vector{CacheEntryRangeF32}()

const _CACHE_LOCK = ReentrantLock()
const _CACHE_SIZE_REF = Ref{Int}(16)  # Max cache entries per store

# Ring buffer pointers for Vector stores
const _RING_IDX_VEC_F64 = Ref{Int}(1)
const _RING_IDX_VEC_F32 = Ref{Int}(1)

# Ring buffer pointers for Range stores
const _RING_IDX_RANGE_F64 = Ref{Int}(1)
const _RING_IDX_RANGE_F32 = Ref{Int}(1)

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
        # Clear Vector stores
        empty!(_CUBIC_CACHE_STORE_VEC_F64)
        empty!(_CUBIC_CACHE_STORE_VEC_F32)
        _RING_IDX_VEC_F64[] = 1
        _RING_IDX_VEC_F32[] = 1

        # Clear Range stores
        empty!(_CUBIC_CACHE_STORE_RANGE_F64)
        empty!(_CUBIC_CACHE_STORE_RANGE_F32)
        _RING_IDX_RANGE_F64[] = 1
        _RING_IDX_RANGE_F32[] = 1

        # Reset statistics
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
`NamedTuple` with fields: `hits`, `misses`, `evictions`, `vec_size`, `range_size`, `size`, `efficiency`
"""
function cubic_cache_stats()
    hits = _CACHE_HITS[]
    misses = _CACHE_MISSES[]
    evictions = _CACHE_EVICTIONS[]

    local vec_size, range_size
    lock(_CACHE_LOCK)
    try
        vec_size = length(_CUBIC_CACHE_STORE_VEC_F64) + length(_CUBIC_CACHE_STORE_VEC_F32)
        range_size = length(_CUBIC_CACHE_STORE_RANGE_F64) + length(_CUBIC_CACHE_STORE_RANGE_F32)
    finally
        unlock(_CACHE_LOCK)
    end

    total = hits + misses
    efficiency = total > 0 ? round(100 * hits / total, digits=1) : 0.0

    return (
        hits = hits,
        misses = misses,
        evictions = evictions,
        vec_size = vec_size,
        range_size = range_size,
        size = vec_size + range_size,
        efficiency = efficiency
    )
end

# ===============================================================
# Internal Cache Management
# ===============================================================

# ─────────────────────────────────────────────────────────────────
# Vector Cache Helpers
# ─────────────────────────────────────────────────────────────────

"""
Add Float64 Vector entry to cache with ring buffer eviction.
Must be called within lock.
"""
function add_to_cache_vec_f64!(entry::CacheEntryVecF64)
    store = _CUBIC_CACHE_STORE_VEC_F64
    ring_idx = _RING_IDX_VEC_F64
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
Add Float32 Vector entry to cache with ring buffer eviction.
"""
function add_to_cache_vec_f32!(entry::CacheEntryVecF32)
    store = _CUBIC_CACHE_STORE_VEC_F32
    ring_idx = _RING_IDX_VEC_F32
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

# ─────────────────────────────────────────────────────────────────
# Range Cache Helpers
# ─────────────────────────────────────────────────────────────────

"""
Add Float64 Range entry to cache with ring buffer eviction.
Must be called within lock.
"""
function add_to_cache_range_f64!(entry::CacheEntryRangeF64)
    store = _CUBIC_CACHE_STORE_RANGE_F64
    ring_idx = _RING_IDX_RANGE_F64
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
Add Float32 Range entry to cache with ring buffer eviction.
"""
function add_to_cache_range_f32!(entry::CacheEntryRangeF32)
    store = _CUBIC_CACHE_STORE_RANGE_F32
    ring_idx = _RING_IDX_RANGE_F32
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

# ═══════════════════════════════════════════════════════════════════════
# get_cubic_cache - Dispatches to Range or Vector stores
# ═══════════════════════════════════════════════════════════════════════
#
# Type dispatch hierarchy:
#   - _StepRangeLen_F64 → Range store (O(1) index lookup preserved!)
#   - Vector{Float64} → Vector store (O(log n) binary search)
#   - AbstractRange → normalized to _StepRangeLen → Range store
#   - AbstractVector → collected to Vector → Vector store

# ─────────────────────────────────────────────────────────────────
# Range-specific methods (O(1) index lookup preserved!)
# ─────────────────────────────────────────────────────────────────

"""
    get_cubic_cache(x::_StepRangeLen_F64)

Lookup/create cache for StepRangeLen grid. Preserves Range type for O(1) index lookup.
"""
function get_cubic_cache(x::_StepRangeLen_F64)
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_RANGE_F64

    lock(_CACHE_LOCK)
    try
        # [Pass 1] Identity Check - O(1)
        @inbounds for entry in store
            if entry.id === id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Pass 2] Equality Check - O(1) for Range (compares start/step/length)
        @inbounds for entry in store
            if entry.x == x
                entry.id = id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Cache Miss] Create Range-based cache (preserves O(1) lookup!)
        _CACHE_MISSES[] += 1
        new_spline = CubicSplineCache(x)  # x stays as Range!
        new_entry = CacheEntryRangeF64(id, x, new_spline)
        add_to_cache_range_f64!(new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

function get_cubic_cache(x::_StepRangeLen_F32)
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_RANGE_F32

    lock(_CACHE_LOCK)
    try
        @inbounds for entry in store
            if entry.id === id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        @inbounds for entry in store
            if entry.x == x
                entry.id = id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        _CACHE_MISSES[] += 1
        new_spline = CubicSplineCache(x)
        new_entry = CacheEntryRangeF32(id, x, new_spline)
        add_to_cache_range_f32!(new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

# AbstractRange fallback: normalize to canonical StepRangeLen
function get_cubic_cache(x::AbstractRange{Float64})
    # Convert to canonical StepRangeLen{Float64, TwicePrecision, TwicePrecision, Int64}
    x_range = range(first(x), last(x), length(x))
    return get_cubic_cache(x_range)
end

function get_cubic_cache(x::AbstractRange{Float32})
    x_range = range(first(x), last(x), length(x))
    return get_cubic_cache(x_range)
end

# ─────────────────────────────────────────────────────────────────
# Vector-specific methods (O(log n) binary search)
# ─────────────────────────────────────────────────────────────────

"""
    get_cubic_cache(x::Vector{Float64})

Lookup/create cache for Vector grid. Uses binary search for interval lookup.
"""
function get_cubic_cache(x::Vector{Float64})
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_VEC_F64

    lock(_CACHE_LOCK)
    try
        # [Pass 1] Identity Check - O(K)
        @inbounds for entry in store
            if entry.id === id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Pass 2] Equality Check - O(K×N)
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
        new_entry = CacheEntryVecF64(id, x, new_spline)
        add_to_cache_vec_f64!(new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

function get_cubic_cache(x::Vector{Float32})
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_VEC_F32

    lock(_CACHE_LOCK)
    try
        @inbounds for entry in store
            if entry.id === id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        @inbounds for entry in store
            if isequal(entry.x, x)
                entry.id = id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        _CACHE_MISSES[] += 1
        new_spline = CubicSplineCache(x)
        new_entry = CacheEntryVecF32(id, x, new_spline)
        add_to_cache_vec_f32!(new_entry)

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
