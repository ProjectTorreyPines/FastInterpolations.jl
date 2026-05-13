# ========================================
# _CachedRange: Normalized AbstractRange
# ========================================
#
# All AbstractRange inputs are normalized to _CachedRange{T,Tinv} at system boundaries
# (via `_to_float`). After normalization, the grid type space is exactly
# {_CachedRange{T,Tinv}, Vector{T}} — eliminating downstream dispatch explosion.
# `Tinv = typeof(inv(oneunit(T)))` — equals `T` for `T <: AbstractFloat`, equals
# `Float64` for `T = Int` (mirrors the `_CachedVector{T, Tinv}` shape).
#
# Include order: grid_spacing.jl → cached_range.jl → search.jl → utils.jl

"""
    _CachedRange{T, Tinv} <: AbstractRange{T}

`AbstractRange` subtype that pre-caches `first`, `last`, `step`, and `inv(step)` as
plain `T`/`Tinv` fields, enabling multiply-instead-of-divide search and avoiding
TwicePrecision dispatch overhead.

All `AbstractRange` inputs are normalized to `_CachedRange` via `_to_float` at public
API boundaries. This means downstream code only needs to handle two grid types:
`_CachedRange{T,Tinv}` (uniform) and `Vector{T}` (non-uniform).

`Tinv == typeof(inv(oneunit(T)))` — equals `T` for `T <: AbstractFloat`, equals
`Float64` for raw `T = Int`. Mirrors `_CachedVector{T, Tinv}`.

# Fields
- `lo::T`        — `first(x)`, cached as plain `T` (used for index computation)
- `hi::T`        — `last(x)`, cached as plain `T` (used for index computation)
- `h::T`         — `step(x)`, cached as plain `T`
- `inv_h::Tinv`  — precomputed `1/step(x)`, enables multiply-not-divide in `_search_direct`
- `len::Int`     — `length(x)`
- `domain_lo::T` — safe lower bound for domain checks (≤ lo, prevents false DomainError)
- `domain_hi::T` — safe upper bound for domain checks (≥ hi, prevents false DomainError)

The `domain_lo`/`domain_hi` fields equal `lo`/`hi` in the exact path (ARM, non-TwicePrecision).
On x86_64 with TwicePrecision fast path, they are widened by 1 ULP to account for
possible rounding difference between fast plain-T arithmetic and exact TwicePrecision.

Because `_CachedRange <: AbstractRange`, all existing `AbstractRange` dispatch
(DirectSearch routing, `_resolve_search_policy`, `search_interval`, etc.) propagates
automatically without any changes at call sites.
"""
struct _CachedRange{T, Tinv} <: AbstractRange{T}
    lo::T
    hi::T
    h::T
    inv_h::Tinv
    len::Int
    domain_lo::T
    domain_hi::T
end

# Exact constructor: domain_lo == lo, domain_hi == hi (default for non-TwicePrecision paths)
@inline function _CachedRange{T, Tinv}(lo::T, hi::T, h::T, inv_h::Tinv, len::Int) where {T, Tinv}
    return _CachedRange{T, Tinv}(lo, hi, h, inv_h, len, lo, hi)
end

# Convenience: construct from any AbstractRange{T}, using its own eltype.
# Internal code should prefer _to_float(x, Tg) when the desired target type Tg differs from eltype(x).
_CachedRange(x::AbstractRange{T}) where {T} = _to_float(x, T)
_CachedRange(x::_CachedRange) = x

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

# Range slicing — return a new _CachedRange instead of falling back to a generic
# `Vector{T}` (from getindex) or `SubArray{T, 1, _CachedRange{T}, ...}` (from view).
# Both fallbacks would lose the `<: AbstractRange` type tag, breaking trait-based
# dispatch like `_can_infer_period(::AbstractRange) -> true` (see periodic.jl:161).
#
# This matches how Julia's built-in `StepRangeLen` handles `view`/`getindex` with
# range indices: the result stays a range, preserving uniformity and step caches.
# Without this method, any code that windows or slices a `_CachedRange` (e.g. the
# Hermite ND cell-local OnTheFly path in hetero_eval.jl) would silently degrade
# its grid to a non-range type.
@inline function Base.getindex(r::_CachedRange{T, Tinv}, idx::AbstractUnitRange{<:Integer}) where {T, Tinv}
    @boundscheck checkbounds(r, idx)
    new_len = length(idx)
    # Empty slice: return a length-0 _CachedRange anchored at r.lo (callers that
    # would dereference this hit the same checkbounds wall they would on r itself).
    new_len == 0 && return _CachedRange{T, Tinv}(r.lo, r.lo, r.h, r.inv_h, 0)
    i_lo = Int(first(idx))
    i_hi = Int(last(idx))
    # Reuse cached endpoints when the slice touches them — preserves the exact-bit
    # value of `r.lo`/`r.hi` for full-axis or boundary-touching slices, which keeps
    # any downstream Tg-precision comparison stable.
    new_lo = i_lo == 1 ? r.lo : muladd(i_lo - 1, r.h, r.lo)
    new_hi = i_hi == r.len ? r.hi : muladd(i_hi - 1, r.h, r.lo)
    return _CachedRange{T, Tinv}(new_lo, new_hi, r.h, r.inv_h, new_len)
end

# `view` follows `getindex` semantics for ranges — both return a fresh range, no
# parent reference is needed (the struct is small and immutable). This mirrors
# `Base.view(::StepRangeLen, ::AbstractUnitRange)` from Base.
@inline Base.view(r::_CachedRange, idx::AbstractUnitRange{<:Integer}) = r[idx]

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
function _to_float(x::AbstractRange, ::Type{T}) where {T}
    h = T(step(x))
    inv_h = inv(h)
    return _CachedRange{T, typeof(inv_h)}(T(first(x)), T(last(x)), h, inv_h, length(x))
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
        return _CachedRange{FT, FT}(lo, hi, h, inv(h), length(x), domain_lo, domain_hi)
    end
end

# _CachedRange same-type pass-through: already normalized, return as-is.
_to_float(x::_CachedRange{T, Tinv}, ::Type{T}) where {T, Tinv} = x

# _CachedRange type-mismatch (e.g. Float32 → Float64 via _convert_grid):
# Uses 5-arg constructor (domain = exact). Any x86_64 domain widening from the source
# is intentionally dropped: the type conversion itself introduces fresh rounding,
# so re-widening would need to be based on the new FT, not the old T.
function _to_float(x::_CachedRange, ::Type{T}) where {T}
    h = T(x.h)
    inv_h = inv(h)
    return _CachedRange{T, typeof(inv_h)}(T(x.lo), T(x.hi), h, inv_h, x.len)
end

"""
    _to_float_adding_endpoint(x::AbstractRange, ::Type{FT}) -> _CachedRange{FT, Tinv}

Extend `x` by one step (for exclusive periodic grids: length n → n+1),
converting to `_CachedRange{FT, Tinv}` (`Tinv = typeof(inv(oneunit(FT)))`).

Dispatch:
- `_CachedRange{FT, Tinv}`: pure field copy + one addition — zero recomputation
- Other `AbstractRange` (OrdinalRange, StepRangeLen, LinRange, ...): convert + extend
"""
# domain_hi = hi_new (exact): the extension uses cached plain-T fields only
# (no TwicePrecision involved), so no additional rounding uncertainty.
@inline function _to_float_adding_endpoint(x::_CachedRange{T, Tinv}, ::Type{T}) where {T, Tinv}
    hi_new = x.hi + x.h
    return _CachedRange{T, Tinv}(
        x.lo, hi_new, x.h, x.inv_h, x.len + 1,
        x.domain_lo, hi_new
    )
end

@inline function _to_float_adding_endpoint(x::AbstractRange, ::Type{T}) where {T}
    r = _to_float(x, T)
    return _to_float_adding_endpoint(r, T)
end

# ---------- Wrapper-aware `_convert_copy` ----------
# `_CachedRange{T}` is an immutable struct of scalar fields — no buffer to
# share with user code, so same-type "copy" is the same reference (free).
# Different-type → `_to_float(r, T)` rebuilds with the target eltype.
# Mirrors the pattern in cached_vector.jl / periodic_axis.jl for unified
# `map(g -> _convert_copy(g, Tg), grids)` use across grid wrapper types.
@inline _convert_copy(r::_CachedRange{T, Tinv}, ::Type{T}) where {T, Tinv} = r
@inline _convert_copy(r::_CachedRange, ::Type{T}) where {T} = _to_float(r, T)

# 3-arg grid-based accessors: _CachedRange has h/inv_h cached in the struct.
# Args are (x, xL, xR) — natural L→R order, matching AbstractVector fallbacks
# in grid_spacing.jl. The cached form ignores both endpoints (uniform step).
@inline _get_h(x::_CachedRange, ::Real, ::Real) = x.h
@inline _get_inv_h(x::_CachedRange, ::Real, ::Real) = x.inv_h
