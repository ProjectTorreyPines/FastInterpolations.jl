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
# Cache Entries - Parametric types for both Natural and Periodic BC
# ═══════════════════════════════════════════════════════════════════════
# Using parametric structs reduces duplication while maintaining type stability.
# Vector{CacheEntryVec{Float64, Nothing}} is a concrete element type → zero-allocation iteration.

# Vector-based cache entries (O(log n) binary search during interpolation)
mutable struct CacheEntryVec{T<:AbstractFloat, BC}
    id::UInt                          # objectid(x) - Fast Path lookup
    x::Vector{T}                      # Concrete Vector type
    spline::CubicSplineCache{T, Vector{T}, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, BC}
end

# Range-based cache entries (O(1) direct index calculation during interpolation!)
mutable struct CacheEntryRange{T<:AbstractFloat, BC, R<:AbstractRange{T}}
    id::UInt                          # objectid(x) - Fast Path lookup
    x::R                              # Concrete Range type
    spline::CubicSplineCache{T, R, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, BC}
end

# Type aliases for Natural BC (backward compatibility)
const CacheEntryVecF64 = CacheEntryVec{Float64, Nothing}
const CacheEntryVecF32 = CacheEntryVec{Float32, Nothing}
const CacheEntryRangeF64 = CacheEntryRange{Float64, Nothing, _StepRangeLen_F64}
const CacheEntryRangeF32 = CacheEntryRange{Float32, Nothing, _StepRangeLen_F32}

# Type aliases for Periodic BC
const CacheEntryVecF64Periodic = CacheEntryVec{Float64, PeriodicData{Float64}}
const CacheEntryVecF32Periodic = CacheEntryVec{Float32, PeriodicData{Float32}}
const CacheEntryRangeF64Periodic = CacheEntryRange{Float64, PeriodicData{Float64}, _StepRangeLen_F64}
const CacheEntryRangeF32Periodic = CacheEntryRange{Float32, PeriodicData{Float32}, _StepRangeLen_F32}

# ═══════════════════════════════════════════════════════════════════════
# Global Cache Stores
# ═══════════════════════════════════════════════════════════════════════
# Separate stores for Vector/Range and Natural/Periodic to maintain type stability
# Type-stable vectors avoid boxing during iteration (critical for zero-allocation)

# Vector-based stores - Natural BC (for non-uniform grids)
const _CUBIC_CACHE_STORE_VEC_F64 = Vector{CacheEntryVecF64}()
const _CUBIC_CACHE_STORE_VEC_F32 = Vector{CacheEntryVecF32}()

# Range-based stores - Natural BC (for uniform grids - O(1) index lookup!)
const _CUBIC_CACHE_STORE_RANGE_F64 = Vector{CacheEntryRangeF64}()
const _CUBIC_CACHE_STORE_RANGE_F32 = Vector{CacheEntryRangeF32}()

# Vector-based stores - Periodic BC
const _CUBIC_CACHE_STORE_VEC_F64_PERIODIC = Vector{CacheEntryVecF64Periodic}()
const _CUBIC_CACHE_STORE_VEC_F32_PERIODIC = Vector{CacheEntryVecF32Periodic}()

# Range-based stores - Periodic BC
const _CUBIC_CACHE_STORE_RANGE_F64_PERIODIC = Vector{CacheEntryRangeF64Periodic}()
const _CUBIC_CACHE_STORE_RANGE_F32_PERIODIC = Vector{CacheEntryRangeF32Periodic}()

const _CACHE_LOCK = ReentrantLock()
const _CACHE_SIZE_REF = Ref{Int}(16)  # Max cache entries per store

# Ring buffer pointers - Natural BC
const _RING_IDX_VEC_F64 = Ref{Int}(1)
const _RING_IDX_VEC_F32 = Ref{Int}(1)
const _RING_IDX_RANGE_F64 = Ref{Int}(1)
const _RING_IDX_RANGE_F32 = Ref{Int}(1)

# Ring buffer pointers - Periodic BC
const _RING_IDX_VEC_F64_PERIODIC = Ref{Int}(1)
const _RING_IDX_VEC_F32_PERIODIC = Ref{Int}(1)
const _RING_IDX_RANGE_F64_PERIODIC = Ref{Int}(1)
const _RING_IDX_RANGE_F32_PERIODIC = Ref{Int}(1)

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
        # Clear Vector stores - Natural BC
        empty!(_CUBIC_CACHE_STORE_VEC_F64)
        empty!(_CUBIC_CACHE_STORE_VEC_F32)
        _RING_IDX_VEC_F64[] = 1
        _RING_IDX_VEC_F32[] = 1

        # Clear Range stores - Natural BC
        empty!(_CUBIC_CACHE_STORE_RANGE_F64)
        empty!(_CUBIC_CACHE_STORE_RANGE_F32)
        _RING_IDX_RANGE_F64[] = 1
        _RING_IDX_RANGE_F32[] = 1

        # Clear Vector stores - Periodic BC
        empty!(_CUBIC_CACHE_STORE_VEC_F64_PERIODIC)
        empty!(_CUBIC_CACHE_STORE_VEC_F32_PERIODIC)
        _RING_IDX_VEC_F64_PERIODIC[] = 1
        _RING_IDX_VEC_F32_PERIODIC[] = 1

        # Clear Range stores - Periodic BC
        empty!(_CUBIC_CACHE_STORE_RANGE_F64_PERIODIC)
        empty!(_CUBIC_CACHE_STORE_RANGE_F32_PERIODIC)
        _RING_IDX_RANGE_F64_PERIODIC[] = 1
        _RING_IDX_RANGE_F32_PERIODIC[] = 1

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

Note: `vec_size` and `range_size` include both natural and periodic BC caches.
"""
function cubic_cache_stats()
    hits = _CACHE_HITS[]
    misses = _CACHE_MISSES[]
    evictions = _CACHE_EVICTIONS[]

    local vec_size, range_size
    lock(_CACHE_LOCK)
    try
        # Include both natural and periodic stores
        vec_natural = length(_CUBIC_CACHE_STORE_VEC_F64) + length(_CUBIC_CACHE_STORE_VEC_F32)
        vec_periodic = length(_CUBIC_CACHE_STORE_VEC_F64_PERIODIC) + length(_CUBIC_CACHE_STORE_VEC_F32_PERIODIC)
        range_natural = length(_CUBIC_CACHE_STORE_RANGE_F64) + length(_CUBIC_CACHE_STORE_RANGE_F32)
        range_periodic = length(_CUBIC_CACHE_STORE_RANGE_F64_PERIODIC) + length(_CUBIC_CACHE_STORE_RANGE_F32_PERIODIC)

        vec_size = vec_natural + vec_periodic
        range_size = range_natural + range_periodic
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

"""
    _add_to_cache!(store::Vector{E}, ring_idx::Ref{Int}, entry::E) where E

Generic add-to-cache with ring buffer eviction. Type-stable via parametric dispatch.
Must be called within lock.

Replaces the previous 4 type-specific functions (add_to_cache_vec_f64!, etc.)
with a single generic implementation that works for all entry types.
"""
@inline function _add_to_cache!(store::Vector{E}, ring_idx::Ref{Int}, entry::E) where E
    max_size = _CACHE_SIZE_REF[]

    if length(store) < max_size
        push!(store, entry)
    else
        idx = ring_idx[]
        store[idx] = entry
        ring_idx[] = (idx % max_size) + 1
        _CACHE_EVICTIONS[] += 1
    end
    return nothing
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
        _add_to_cache!(_CUBIC_CACHE_STORE_RANGE_F64, _RING_IDX_RANGE_F64, new_entry)

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
        _add_to_cache!(_CUBIC_CACHE_STORE_RANGE_F32, _RING_IDX_RANGE_F32, new_entry)

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
        _add_to_cache!(_CUBIC_CACHE_STORE_VEC_F64, _RING_IDX_VEC_F64, new_entry)

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
        _add_to_cache!(_CUBIC_CACHE_STORE_VEC_F32, _RING_IDX_VEC_F32, new_entry)

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

# ═══════════════════════════════════════════════════════════════════════
# Periodic BC Cache Lookup Functions
# ═══════════════════════════════════════════════════════════════════════
#
# Separate functions for periodic BC to maintain type stability.
# Uses separate stores for periodic caches to avoid type mixing.

# ─────────────────────────────────────────────────────────────────
# Periodic Range methods (O(1) index lookup preserved!)
# ─────────────────────────────────────────────────────────────────

"""
    _get_cubic_cache_periodic(x::_StepRangeLen_F64)

Lookup/create periodic cache for StepRangeLen grid.
"""
function _get_cubic_cache_periodic(x::_StepRangeLen_F64)
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_RANGE_F64_PERIODIC

    lock(_CACHE_LOCK)
    try
        # [Pass 1] Identity Check
        @inbounds for entry in store
            if entry.id === id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Pass 2] Equality Check
        @inbounds for entry in store
            if entry.x == x
                entry.id = id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Cache Miss]
        _CACHE_MISSES[] += 1
        new_spline = CubicSplineCache(x; bc=:periodic)
        new_entry = CacheEntryRangeF64Periodic(id, x, new_spline)
        _add_to_cache!(_CUBIC_CACHE_STORE_RANGE_F64_PERIODIC, _RING_IDX_RANGE_F64_PERIODIC, new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

function _get_cubic_cache_periodic(x::_StepRangeLen_F32)
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_RANGE_F32_PERIODIC

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
        new_spline = CubicSplineCache(x; bc=:periodic)
        new_entry = CacheEntryRangeF32Periodic(id, x, new_spline)
        _add_to_cache!(_CUBIC_CACHE_STORE_RANGE_F32_PERIODIC, _RING_IDX_RANGE_F32_PERIODIC, new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

# AbstractRange fallback: normalize to canonical StepRangeLen
function _get_cubic_cache_periodic(x::AbstractRange{Float64})
    x_range = range(first(x), last(x), length(x))
    return _get_cubic_cache_periodic(x_range)
end

function _get_cubic_cache_periodic(x::AbstractRange{Float32})
    x_range = range(first(x), last(x), length(x))
    return _get_cubic_cache_periodic(x_range)
end

# ─────────────────────────────────────────────────────────────────
# Periodic Vector methods (O(log n) binary search)
# ─────────────────────────────────────────────────────────────────

"""
    _get_cubic_cache_periodic(x::Vector{Float64})

Lookup/create periodic cache for Vector grid.
"""
function _get_cubic_cache_periodic(x::Vector{Float64})
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_VEC_F64_PERIODIC

    lock(_CACHE_LOCK)
    try
        # [Pass 1] Identity Check
        @inbounds for entry in store
            if entry.id === id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Pass 2] Equality Check
        @inbounds for entry in store
            if isequal(entry.x, x)
                entry.id = id
                _CACHE_HITS[] += 1
                return entry.spline
            end
        end

        # [Cache Miss]
        _CACHE_MISSES[] += 1
        new_spline = CubicSplineCache(x; bc=:periodic)
        new_entry = CacheEntryVecF64Periodic(id, x, new_spline)
        _add_to_cache!(_CUBIC_CACHE_STORE_VEC_F64_PERIODIC, _RING_IDX_VEC_F64_PERIODIC, new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

function _get_cubic_cache_periodic(x::Vector{Float32})
    id = objectid(x)
    store = _CUBIC_CACHE_STORE_VEC_F32_PERIODIC

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
        new_spline = CubicSplineCache(x; bc=:periodic)
        new_entry = CacheEntryVecF32Periodic(id, x, new_spline)
        _add_to_cache!(_CUBIC_CACHE_STORE_VEC_F32_PERIODIC, _RING_IDX_VEC_F32_PERIODIC, new_entry)

        return new_spline
    finally
        unlock(_CACHE_LOCK)
    end
end

# Fallback for other AbstractFloat types
function _get_cubic_cache_periodic(x::AbstractVector{T}) where {T<:AbstractFloat}
    x_f64 = convert(Vector{Float64}, x)
    return _get_cubic_cache_periodic(x_f64)
end

# ═══════════════════════════════════════════════════════════════════════
# Public API: get_cubic_cache_periodic
# ═══════════════════════════════════════════════════════════════════════

"""
    get_cubic_cache_periodic(x)

Get or create a cached CubicSplineCache with periodic BC for the given x-grid.

# Arguments
- `x`: X-grid (Vector or Range)

# Returns
Cached `CubicSplineCache` with periodic BC for zero-allocation repeated interpolation.

# Example
```julia
x = collect(range(0.0, 2π, 101))
cache = get_cubic_cache_periodic(x)
```
"""
get_cubic_cache_periodic(x::Vector{Float64}) = _get_cubic_cache_periodic(x)
get_cubic_cache_periodic(x::Vector{Float32}) = _get_cubic_cache_periodic(x)
get_cubic_cache_periodic(x::_StepRangeLen_F64) = _get_cubic_cache_periodic(x)
get_cubic_cache_periodic(x::_StepRangeLen_F32) = _get_cubic_cache_periodic(x)
get_cubic_cache_periodic(x::AbstractRange{Float64}) = _get_cubic_cache_periodic(range(first(x), last(x), length(x)))
get_cubic_cache_periodic(x::AbstractRange{Float32}) = _get_cubic_cache_periodic(range(first(x), last(x), length(x)))
get_cubic_cache_periodic(x::AbstractVector{T}) where {T<:AbstractFloat} = _get_cubic_cache_periodic(convert(Vector{Float64}, x))
