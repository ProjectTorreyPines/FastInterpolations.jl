"""
    cubic_autocache.jl

Automatic caching layer for cubic spline interpolation.
Transparently reuses LU factorization for repeated x-grids.

# Implementation Details

- Zero-allocation cache hit via 2-pass lookup (objectid → isequal)
- Ring buffer eviction for O(1) cache replacement
- Self-healing: updates objectid on content match for future fast-path hits
- CacheBank structure for logical grouping with type-stable concrete stores
- Thread-safe with ReentrantLock on cache access
"""

# ===============================================================
# Cache Infrastructure - Type Aliases
# ===============================================================

# StepRangeLen concrete types (from `range(a, b, n)`)
const _StepRangeLen_F64 = StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
const _StepRangeLen_F32 = StepRangeLen{Float32, Float64, Float64, Int64}

# ═══════════════════════════════════════════════════════════════════════
# Cache Entries - Parametric types for both Natural and Periodic BC
# ═══════════════════════════════════════════════════════════════════════

# Vector-based cache entries (O(log n) binary search during interpolation)
mutable struct CacheEntryVec{T<:AbstractFloat, BC}
    id::UInt
    x::Vector{T}
    spline::CubicSplineCache{T, Vector{T}, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, BC}
end

# Range-based cache entries (O(1) direct index calculation during interpolation!)
mutable struct CacheEntryRange{T<:AbstractFloat, BC, R<:AbstractRange{T}}
    id::UInt
    x::R
    spline::CubicSplineCache{T, R, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, BC}
end

# Type aliases for Natural BC
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
# CacheBank - Logical grouping of F64/F32 stores with concrete types
# ═══════════════════════════════════════════════════════════════════════
#
# Each bank holds F64 and F32 stores for a specific (GridKind × BC) combination.
# Internal stores remain concrete vectors for type stability.
# BCVal is a compile-time token (Val{:natural} or Val{:periodic}).

mutable struct CacheBank{E64, E32, BCVal}
    store_f64::Vector{E64}
    store_f32::Vector{E32}
    ring_f64::Base.RefValue{Int}
    ring_f32::Base.RefValue{Int}
end

# 4 Banks: (Vec × Natural), (Vec × Periodic), (Range × Natural), (Range × Periodic)
const _BANK_VEC_NATURAL = CacheBank{CacheEntryVecF64, CacheEntryVecF32, Val{:natural}}(
    Vector{CacheEntryVecF64}(), Vector{CacheEntryVecF32}(), Ref(1), Ref(1)
)
const _BANK_VEC_PERIODIC = CacheBank{CacheEntryVecF64Periodic, CacheEntryVecF32Periodic, Val{:periodic}}(
    Vector{CacheEntryVecF64Periodic}(), Vector{CacheEntryVecF32Periodic}(), Ref(1), Ref(1)
)
const _BANK_RANGE_NATURAL = CacheBank{CacheEntryRangeF64, CacheEntryRangeF32, Val{:natural}}(
    Vector{CacheEntryRangeF64}(), Vector{CacheEntryRangeF32}(), Ref(1), Ref(1)
)
const _BANK_RANGE_PERIODIC = CacheBank{CacheEntryRangeF64Periodic, CacheEntryRangeF32Periodic, Val{:periodic}}(
    Vector{CacheEntryRangeF64Periodic}(), Vector{CacheEntryRangeF32Periodic}(), Ref(1), Ref(1)
)

const _CACHE_LOCK = ReentrantLock()
const _CACHE_SIZE_REF = Ref{Int}(16)

# Statistics
const _CACHE_HITS = Ref{Int}(0)
const _CACHE_MISSES = Ref{Int}(0)
const _CACHE_EVICTIONS = Ref{Int}(0)

# ===============================================================
# Public API
# ===============================================================

"""
    set_cubic_cache_size!(n::Int)

Set maximum number of cached x-grids per store.

# Arguments
- `n::Int`: Maximum cache entries (default: 16)
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

Clear all cached x-grids.
"""
function clear_cubic_cache!()
    lock(_CACHE_LOCK)
    try
        # Clear all 4 banks
        for bank in (_BANK_VEC_NATURAL, _BANK_VEC_PERIODIC, _BANK_RANGE_NATURAL, _BANK_RANGE_PERIODIC)
            empty!(bank.store_f64)
            empty!(bank.store_f32)
            bank.ring_f64[] = 1
            bank.ring_f32[] = 1
        end
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

Return cache hit/miss statistics.

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
        vec_size = length(_BANK_VEC_NATURAL.store_f64) + length(_BANK_VEC_NATURAL.store_f32) +
                   length(_BANK_VEC_PERIODIC.store_f64) + length(_BANK_VEC_PERIODIC.store_f32)
        range_size = length(_BANK_RANGE_NATURAL.store_f64) + length(_BANK_RANGE_NATURAL.store_f32) +
                     length(_BANK_RANGE_PERIODIC.store_f64) + length(_BANK_RANGE_PERIODIC.store_f32)
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
# Internal: BC-specific cache creation trait
# ===============================================================

@inline _build_cache(x, ::Val{:natural}) = CubicSplineCache(x)
@inline _build_cache(x, ::Val{:periodic}) = CubicSplineCache(x; bc=:periodic)

# ===============================================================
# Internal: Generic ring buffer add
# ===============================================================

@inline function _add_to_cache!(store::Vector{E}, ring::Base.RefValue{Int}, entry::E) where E
    max_size = _CACHE_SIZE_REF[]
    if length(store) < max_size
        push!(store, entry)
    else
        idx = ring[]
        store[idx] = entry
        ring[] = (idx % max_size) + 1
        _CACHE_EVICTIONS[] += 1
    end
    return nothing
end

# ===============================================================
# Internal: Generic lookup/insert core
# ===============================================================
#
# Type-stable because:
# - store::Vector{E} has concrete element type E
# - E determines spline return type via its field type
# - All dispatch is compile-time via BCVal

@inline function _lookup_or_insert!(
    store::Vector{E}, ring::Base.RefValue{Int}, x, ::Type{E}, bcval::BCVal
) where {E, BCVal}
    id = objectid(x)

    # [Pass 1] Identity Check - O(1) per entry
    @inbounds for entry in store
        if entry.id === id
            _CACHE_HITS[] += 1
            return entry.spline
        end
    end

    # [Pass 2] Equality Check - O(1) for Range, O(N) for Vector
    @inbounds for entry in store
        if isequal(entry.x, x)
            entry.id = id  # Self-healing
            _CACHE_HITS[] += 1
            return entry.spline
        end
    end

    # [Cache Miss] Create new entry
    _CACHE_MISSES[] += 1
    new_spline = _build_cache(x, bcval)
    new_entry = E(id, x, new_spline)
    _add_to_cache!(store, ring, new_entry)

    return new_spline
end

# ═══════════════════════════════════════════════════════════════════════
# Thin Wrappers: Store selection at compile time
# ═══════════════════════════════════════════════════════════════════════

# Vector F64
@inline _get_cache(x::Vector{Float64}, bank::CacheBank{E64,E32,BCVal}) where {E64,E32,BCVal} =
    _lookup_or_insert!(bank.store_f64, bank.ring_f64, x, E64, BCVal())

# Vector F32
@inline _get_cache(x::Vector{Float32}, bank::CacheBank{E64,E32,BCVal}) where {E64,E32,BCVal} =
    _lookup_or_insert!(bank.store_f32, bank.ring_f32, x, E32, BCVal())

# Range F64
@inline _get_cache(x::_StepRangeLen_F64, bank::CacheBank{E64,E32,BCVal}) where {E64,E32,BCVal} =
    _lookup_or_insert!(bank.store_f64, bank.ring_f64, x, E64, BCVal())

# Range F32
@inline _get_cache(x::_StepRangeLen_F32, bank::CacheBank{E64,E32,BCVal}) where {E64,E32,BCVal} =
    _lookup_or_insert!(bank.store_f32, bank.ring_f32, x, E32, BCVal())

# ═══════════════════════════════════════════════════════════════════════
# Public API: get_cubic_cache(x; bc=:natural)
# ═══════════════════════════════════════════════════════════════════════

"""
    get_cubic_cache(x; bc::Symbol=:natural)

Get or create a cached CubicSplineCache for the given x-grid.

# Arguments
- `x`: X-grid (Vector or Range)
- `bc`: Boundary condition - `:natural` (default) or `:periodic`

# Returns
Cached `CubicSplineCache` for zero-allocation repeated interpolation.

# Examples
```julia
cache = get_cubic_cache(x)                    # Natural BC
cache = get_cubic_cache(x; bc=:periodic)      # Periodic BC
```
"""
# Keyword convenience API (for users)
@inline function get_cubic_cache(x; bc::Symbol=:natural)
    bc === :natural  && return get_cubic_cache(x, Val(:natural))
    bc === :periodic && return get_cubic_cache(x, Val(:periodic))
    throw(ArgumentError("bc must be :natural or :periodic, got :$bc"))
end

# ═══════════════════════════════════════════════════════════════════════
# Val-based API (type-stable, for internal use and advanced users)
# ═══════════════════════════════════════════════════════════════════════

# Vector dispatch (accepts AbstractVector, collects if needed)
get_cubic_cache(x::AbstractVector{T}, ::Val{:natural}) where T<:Union{Float64,Float32} =
    _get_cubic_cache_impl(x isa Vector ? x : collect(x), _BANK_VEC_NATURAL)
get_cubic_cache(x::AbstractVector{T}, ::Val{:periodic}) where T<:Union{Float64,Float32} =
    _get_cubic_cache_impl(x isa Vector ? x : collect(x), _BANK_VEC_PERIODIC)

# Range dispatch (normalize to canonical StepRangeLen)
get_cubic_cache(x::AbstractRange{T}, ::Val{:natural}) where T<:Union{Float64,Float32} =
    _get_cubic_cache_impl(range(first(x), last(x), length(x)), _BANK_RANGE_NATURAL)
get_cubic_cache(x::AbstractRange{T}, ::Val{:periodic}) where T<:Union{Float64,Float32} =
    _get_cubic_cache_impl(range(first(x), last(x), length(x)), _BANK_RANGE_PERIODIC)

# Fallback: other Real types → convert to Float64
get_cubic_cache(x::AbstractVector{<:Real}, bcval::Val) =
    get_cubic_cache(Vector{Float64}(x), bcval)
get_cubic_cache(x::AbstractRange{<:Real}, bcval::Val) =
    get_cubic_cache(range(Float64(first(x)), Float64(last(x)), length(x)), bcval)

# Implementation with lock
@inline function _get_cubic_cache_impl(x, bank)
    lock(_CACHE_LOCK)
    try
        return _get_cache(x, bank)
    finally
        unlock(_CACHE_LOCK)
    end
end
