"""
    cubic_autocache.jl

Automatic caching layer for cubic spline interpolation.
Transparently reuses LU factorization for repeated x-grids.

# Implementation Details

- Fully parametric `CacheBank{T, L, R, X}` - supports any BC type combination
- Zero-allocation cache hit via 2-pass lookup (objectid → isequal)
- Ring buffer eviction for O(1) cache replacement
- Self-healing: updates objectid on content match for future fast-path hits
- Dynamic bank creation via IdDict keyed by type
- Thread-safe with ReentrantLock on cache access
"""

# ===============================================================
# Parametric Cache Types
# ===============================================================

"""
    CacheEntry{T, L, R, X}

A single cache entry storing an x-grid and its associated CubicSplineCache.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `L`: Left boundary condition type (Deriv1{T} or Deriv2{T})
- `R`: Right boundary condition type (Deriv1{T} or Deriv2{T})
- `X`: Grid type (Vector{T} or StepRangeLen)
"""
mutable struct CacheEntry{T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}}
    id::UInt
    x::X
    cache::CubicSplineCache{T, X, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, BCPair{T,L,R}}
end

"""
    CacheBank{T, L, R, X}

A cache bank holding entries for a specific (T, L, R, X) type combination.

# Type Parameters
- `T`: Float type
- `L`: Left BC type
- `R`: Right BC type
- `X`: Grid type
"""
mutable struct CacheBank{T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}}
    store::Vector{CacheEntry{T, L, R, X}}
    ring::Base.RefValue{Int}
end

CacheBank{T,L,R,X}() where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}} =
    CacheBank{T,L,R,X}(CacheEntry{T,L,R,X}[], Ref(1))

"""
    PeriodicCacheEntry{T, X}

Cache entry for periodic BC (uses PeriodicData instead of BCPair).
"""
mutable struct PeriodicCacheEntry{T<:AbstractFloat, X<:AbstractVector{T}}
    id::UInt
    x::X
    cache::CubicSplineCache{T, X, LinearAlgebra.LU{T, LinearAlgebra.Tridiagonal{T, Vector{T}}, Vector{Int64}}, PeriodicData{T}}
end

"""
    PeriodicCacheBank{T, X}

Cache bank for periodic BC.
"""
mutable struct PeriodicCacheBank{T<:AbstractFloat, X<:AbstractVector{T}}
    store::Vector{PeriodicCacheEntry{T, X}}
    ring::Base.RefValue{Int}
end

PeriodicCacheBank{T,X}() where {T<:AbstractFloat, X<:AbstractVector{T}} =
    PeriodicCacheBank{T,X}(PeriodicCacheEntry{T,X}[], Ref(1))

# ===============================================================
# Bank Registry (Dynamic/Lazy)
# ===============================================================

# NOTE: Value type is `Any` because banks are created dynamically for each
# (T, L, R, X) combination at runtime. Type-stable access is ensured by
# the typed getter functions (_get_derivative_bank, _get_periodic_bank).
const _DERIVATIVE_BANK_REGISTRY = IdDict{DataType, Any}()
const _PERIODIC_BANK_REGISTRY = IdDict{DataType, Any}()
const _CACHE_LOCK = ReentrantLock()
const _CACHE_SIZE_REF = Ref{Int}(16)


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
    lock(_CACHE_LOCK) do
        empty!(_DERIVATIVE_BANK_REGISTRY)
        empty!(_PERIODIC_BANK_REGISTRY)
    end
    return nothing
end

# ===============================================================
# Internal: Bank Retrieval (Lock-free Read Path)
# ===============================================================

"""
Get or create a derivative BC cache bank for the given (T, L, R, X) combination.
Uses lock-free read for fast path, lock only for first creation.
"""
@inline function _get_derivative_bank(::X, ::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}}
    BankType = CacheBank{T,L,R,X}

    # Fast path: lock-free read
    bank = get(_DERIVATIVE_BANK_REGISTRY, BankType, nothing)
    bank !== nothing && return bank::BankType

    # Slow path: lock for first creation (try-finally to avoid closure allocation)
    lock(_CACHE_LOCK)
    try
        if !haskey(_DERIVATIVE_BANK_REGISTRY, BankType)
            _DERIVATIVE_BANK_REGISTRY[BankType] = CacheBank{T,L,R,X}()
        end
    finally
        unlock(_CACHE_LOCK)
    end
    return _DERIVATIVE_BANK_REGISTRY[BankType]::BankType
end

"""
Get or create a periodic BC cache bank for the given (T, X) combination.
"""
@inline function _get_periodic_bank(::X) where {T<:AbstractFloat, X<:AbstractVector{T}}
    BankType = PeriodicCacheBank{T,X}

    # Fast path: lock-free read
    bank = get(_PERIODIC_BANK_REGISTRY, BankType, nothing)
    bank !== nothing && return bank::BankType

    # Slow path: lock for first creation (try-finally to avoid closure allocation)
    lock(_CACHE_LOCK)
    try
        if !haskey(_PERIODIC_BANK_REGISTRY, BankType)
            _PERIODIC_BANK_REGISTRY[BankType] = PeriodicCacheBank{T,X}()
        end
    finally
        unlock(_CACHE_LOCK)
    end
    return _PERIODIC_BANK_REGISTRY[BankType]::BankType
end

# ===============================================================
# Internal: Ring Buffer Add
# ===============================================================

@inline function _add_to_ring!(store::Vector{E}, ring::Base.RefValue{Int}, entry::E) where E
    max_size = _CACHE_SIZE_REF[]
    if length(store) < max_size
        push!(store, entry)
    else
        idx = ring[]
        store[idx] = entry
        ring[] = (idx % max_size) + 1
    end
    return nothing
end

# ===============================================================
# Internal: Lookup/Insert (Unified)
# ===============================================================

"""
Lookup or insert into a derivative BC cache bank.
2-pass lookup: identity check (fast) → equality check (slower) → insert.

On cache miss, creates entry with the provided `bc_pair` values. This ensures
the cache has meaningful bc_data for edge cases, though callers should still
use 3-arg `_solve_system!` for dynamic BC values.
"""
@inline function _lookup_or_insert!(bank::CacheBank{T,L,R,X}, x::X, bc_pair::BCPair{T,L,R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, X<:AbstractVector{T}}
    store = bank.store
    ring = bank.ring
    id = objectid(x)

    # [Pass 1] Identity check (fast path)
    @inbounds for entry in store
        if entry.id === id
            return entry.cache
        end
    end

    # [Pass 2] Equality check - O(1) for Range, O(N) for Vector
    @inbounds for entry in store
        if isequal(entry.x, x)
            entry.id = id  # Self-healing for future fast-path hits
            return entry.cache
        end
    end

    # [Cache Miss] Create new entry with the provided BC values.
    # Cache is keyed by BC *type* only (LU factorization depends on matrix structure).
    # On cache hit, bc_data may differ from caller's BC values, so callers should
    # use 3-arg _solve_system!(cache, y, bc_tuple) to apply their actual BC values.
    new_cache = _build_derivative_bc_cache(x, bc_pair.left, bc_pair.right)
    new_entry = CacheEntry{T,L,R,X}(id, x, new_cache)
    _add_to_ring!(store, ring, new_entry)

    return new_cache
end

"""
Lookup or insert into a periodic BC cache bank.
"""
@inline function _lookup_or_insert!(bank::PeriodicCacheBank{T,X}, x::X) where {T<:AbstractFloat, X<:AbstractVector{T}}
    store = bank.store
    ring = bank.ring
    id = objectid(x)

    # [Pass 1] Identity check (fast path)
    @inbounds for entry in store
        if entry.id === id
            return entry.cache
        end
    end

    # [Pass 2] Equality check
    @inbounds for entry in store
        if isequal(entry.x, x)
            entry.id = id  # Self-healing
            return entry.cache
        end
    end

    # [Cache Miss] Create new entry
    new_cache = _build_periodic_cache(x)
    new_entry = PeriodicCacheEntry{T,X}(id, x, new_cache)
    _add_to_ring!(store, ring, new_entry)

    return new_cache
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
# Internal: Cache Implementation with Locking
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
    return _lookup_or_insert!(bank, x_normalized)
end

@inline function _get_periodic_cache_impl(x::AbstractRange{T}) where {T<:Union{Float64,Float32}}
    # Normalize to StepRangeLen for consistent cache key type.
    # LinRange and other Range types are converted (minor overhead on first call).
    x_normalized = (x isa _StepRangeLen_F64 || x isa _StepRangeLen_F32) ? x : range(first(x), last(x), length(x))
    bank = _get_periodic_bank(x_normalized)
    return _lookup_or_insert!(bank, x_normalized)
end

# Fallback: other Real types → convert to Float64
@inline function _get_periodic_cache_impl(x::AbstractVector{<:Real})
    _get_periodic_cache_impl(Vector{Float64}(x))
end

@inline function _get_periodic_cache_impl(x::AbstractRange{<:Real})
    _get_periodic_cache_impl(range(Float64(first(x)), Float64(last(x)), length(x)))
end
