# ========================================
# _CachedRange: Normalized AbstractRange
# ========================================
#
# All AbstractRange inputs are normalized to _CachedRange{T} at system boundaries
# (via `_to_float`). After normalization, the grid type space is exactly
# {_CachedRange{T}, Vector{T}} — eliminating downstream dispatch explosion.
#
# Include order: grid_spacing.jl → cached_range.jl → search.jl → utils.jl

"""
    _CachedRange{T <: AbstractFloat} <: AbstractRange{T}

`AbstractRange` subtype that pre-caches `first`, `last`, `step`, and `inv(step)` as
plain `T` fields, avoiding TwicePrecision arithmetic overhead on Intel CPUs.

All `AbstractRange` inputs are normalized to `_CachedRange` via `_to_float` at public
API boundaries. This means downstream code only needs to handle two grid types:
`_CachedRange{T}` (uniform) and `Vector{T}` (non-uniform).

# Fields
- `lo::T`     — `first(x)`, cached as plain `T`
- `hi::T`     — `last(x)`, cached as plain `T`
- `h::T`      — `step(x)`, cached as plain `T`
- `inv_h::T`  — precomputed `1/step(x)`, enables multiply-not-divide in `_search_direct`
- `len::Int`  — `length(x)`

Because `_CachedRange <: AbstractRange`, all existing `AbstractRange` dispatch
(DirectSearch routing, `_resolve_search_policy`, `search_interval`, etc.) propagates
automatically without any changes at call sites.
"""
struct _CachedRange{T <: AbstractFloat} <: AbstractRange{T}
    lo::T
    hi::T
    h::T
    inv_h::T
    len::Int
end

Base.length(r::_CachedRange) = r.len
Base.first(r::_CachedRange) = r.lo
Base.last(r::_CachedRange) = r.hi
Base.step(r::_CachedRange) = r.h
Base.getindex(r::_CachedRange, i::Int) = muladd(i - 1, r.h, r.lo)

# ========================================
# _to_float: Range → _CachedRange conversion
# ========================================

"""
    _to_float(x::AbstractRange, ::Type{FT}) -> _CachedRange{FT}

Convert any `AbstractRange` to `_CachedRange{FT}`.

This is the universal normalizer for range grids. After `_to_float`, the grid type
space is exactly `{_CachedRange{FT}, Vector{FT}}`, eliminating downstream dispatch
on `StepRangeLen`, `LinRange`, `OrdinalRange`, etc.
"""
function _to_float(x::AbstractRange, ::Type{FT}) where {FT <: AbstractFloat}
    h = FT(step(x))
    return _CachedRange{FT}(FT(first(x)), FT(last(x)), h, inv(h), length(x))
end

# _CachedRange same-type pass-through: already normalized, return as-is.
_to_float(x::_CachedRange{T}, ::Type{T}) where {T <: AbstractFloat} = x

# _CachedRange type-mismatch (e.g. Float32 → Float64 via _convert_grid):
function _to_float(x::_CachedRange, ::Type{FT}) where {FT <: AbstractFloat}
    h = FT(x.h)
    return _CachedRange{FT}(FT(x.lo), FT(x.hi), h, inv(h), x.len)
end

"""
    _to_float_adding_endpoint(x::AbstractRange, ::Type{FT}) -> _CachedRange{FT}

Extend `x` by one step (for exclusive periodic grids: length n → n+1),
converting to `_CachedRange{FT}`.

Dispatch:
- `_CachedRange{FT}`: pure field copy + one addition — zero recomputation
- Other `AbstractRange` (OrdinalRange, StepRangeLen, LinRange, ...): convert + extend
"""
@inline function _to_float_adding_endpoint(x::_CachedRange{FT}, ::Type{FT}) where {FT <: AbstractFloat}
    return _CachedRange{FT}(x.lo, x.hi + x.h, x.h, x.inv_h, x.len + 1)
end

@inline function _to_float_adding_endpoint(x::AbstractRange, ::Type{FT}) where {FT <: AbstractFloat}
    h = FT(step(x))
    lo = FT(first(x))
    return _CachedRange{FT}(lo, FT(last(x)) + h, h, inv(h), length(x) + 1)
end

# ========================================
# _create_spacing: _CachedRange specialization
# ========================================

# _CachedRange already has h and inv_h cached — trivial field copy, no recomputation.
function _create_spacing(x::_CachedRange{T}) where {T <: AbstractFloat}
    return ScalarSpacing{T}(x.h, x.inv_h)
end
