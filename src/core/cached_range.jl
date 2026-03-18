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
plain `T` fields, enabling multiply-instead-of-divide search and avoiding
TwicePrecision dispatch overhead.

All `AbstractRange` inputs are normalized to `_CachedRange` via `_to_float` at public
API boundaries. This means downstream code only needs to handle two grid types:
`_CachedRange{T}` (uniform) and `Vector{T}` (non-uniform).

# Fields
- `lo::T`       — `first(x)`, cached as plain `T` (used for index computation)
- `hi::T`       — `last(x)`, cached as plain `T` (used for index computation)
- `h::T`        — `step(x)`, cached as plain `T`
- `inv_h::T`    — precomputed `1/step(x)`, enables multiply-not-divide in `_search_direct`
- `len::Int`    — `length(x)`
- `domain_lo::T` — safe lower bound for domain checks (≤ lo, prevents false DomainError)
- `domain_hi::T` — safe upper bound for domain checks (≥ hi, prevents false DomainError)

The `domain_lo`/`domain_hi` fields equal `lo`/`hi` in the exact path (ARM, non-TwicePrecision).
On x86_64 with TwicePrecision fast path, they are widened by 1 ULP to account for
possible rounding difference between fast plain-T arithmetic and exact TwicePrecision.

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
    domain_lo::T
    domain_hi::T
end

# Exact constructor: domain_lo == lo, domain_hi == hi (default for non-TwicePrecision paths)
@inline function _CachedRange{T}(lo::T, hi::T, h::T, inv_h::T, len::Int) where {T <: AbstractFloat}
    return _CachedRange{T}(lo, hi, h, inv_h, len, lo, hi)
end

Base.length(r::_CachedRange) = r.len
Base.size(r::_CachedRange) = (r.len,)
Base.first(r::_CachedRange) = r.lo
Base.last(r::_CachedRange) = r.hi
Base.step(r::_CachedRange) = r.h
function Base.getindex(r::_CachedRange, i::Int)
    @boundscheck checkbounds(r, i)
    i == 1 && return r.lo
    i == r.len && return r.hi
    return muladd(i - 1, r.h, r.lo)
end

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

# x86_64: TwicePrecision first()/last() ~9ns each on Intel — bypass via plain-T muladd.
# lo/hi may be ±1 ULP vs exact; domain_lo/domain_hi widened for safe _check_domain.
# ARM: TwicePrecision is fast, so generic AbstractRange path above is used instead.
@static if Sys.ARCH === :x86_64
    function _to_float(
            x::StepRangeLen{FT, Base.TwicePrecision{FT}, Base.TwicePrecision{FT}},
            ::Type{FT}
        ) where {FT <: AbstractFloat}
        h = FT(x.step)
        lo = muladd(1 - x.offset, h, FT(x.ref))
        hi = muladd(x.len - x.offset, h, FT(x.ref))

        domain_lo = prevfloat(lo)
        domain_hi = nextfloat(hi)
        return _CachedRange{FT}(lo, hi, h, inv(h), length(x), domain_lo, domain_hi)
    end
end

# _CachedRange same-type pass-through: already normalized, return as-is.
_to_float(x::_CachedRange{T}, ::Type{T}) where {T <: AbstractFloat} = x

# _CachedRange type-mismatch (e.g. Float32 → Float64 via _convert_grid):
# Uses 5-arg constructor (domain = exact). Any x86_64 domain widening from the source
# is intentionally dropped: the type conversion itself introduces fresh rounding,
# so re-widening would need to be based on the new FT, not the old T.
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
# domain_hi = hi_new (exact): the extension uses cached plain-T fields only
# (no TwicePrecision involved), so no additional rounding uncertainty.
@inline function _to_float_adding_endpoint(x::_CachedRange{FT}, ::Type{FT}) where {FT <: AbstractFloat}
    hi_new = x.hi + x.h
    return _CachedRange{FT}(
        x.lo, hi_new, x.h, x.inv_h, x.len + 1,
        x.domain_lo, hi_new
    )
end

@inline function _to_float_adding_endpoint(x::AbstractRange, ::Type{FT}) where {FT <: AbstractFloat}
    r = _to_float(x, FT)
    return _to_float_adding_endpoint(r, FT)
end

# ========================================
# _create_spacing: _CachedRange specialization
# ========================================

# _CachedRange already has h and inv_h cached — trivial field copy, no recomputation.
function _create_spacing(x::_CachedRange{T}) where {T <: AbstractFloat}
    return ScalarSpacing{T}(x.h, x.inv_h)
end
