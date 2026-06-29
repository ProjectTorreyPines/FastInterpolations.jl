# ========================================
# _CachedRange methods — normalized AbstractRange behavior
# ========================================
#
# Struct definition + 5-arg inner ctor + `_CachedRange(::_CachedRange)` identity
# live in `axis_types.jl` (loaded earlier). This file owns:
#   - `_to_float` (Range → `_CachedRange` normalizer)
#   - AbstractArray interface (`length`, `size`, `first`, `last`, `step`,
#     `getindex(::Int)`, `getindex(::AbstractUnitRange)`, `view`)
#   - `_convert_copy`
#   - `_get_h` / `_get_inv_h`
#   - `_resolve_axis` / `_cache_axis` / `_cache_axis_pooled` (all BC variants
#     for raw `AbstractRange` and pre-wrapped `_CachedRange` input)
#
# Include order: grid_spacing.jl → axis_types.jl → cached_range.jl → ...

# Convenience: construct from any AbstractRange{T}, using its own eltype.
# Prefer `_to_float(x, Tg)` at call sites when the desired target differs.
_CachedRange(x::AbstractRange{T}) where {T} = _to_float(x, T)

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
# Slice preserves the axis tag (a sub-range of a unit-step grid is still unit-step).
@inline function Base.getindex(r::_CachedRange{T, Tinv, Tag}, idx::AbstractUnitRange{<:Integer}) where {T, Tinv, Tag}
    @boundscheck checkbounds(r, idx)
    new_len = length(idx)
    # Empty slice: return a length-0 _CachedRange anchored at r.lo (callers that
    # would dereference this hit the same checkbounds wall they would on r itself).
    new_len == 0 && return _CachedRange{T, Tinv, Tag}(r.lo, r.lo, r.h, r.inv_h, 0)
    i_lo = Int(first(idx))
    i_hi = Int(last(idx))
    # Reuse cached endpoints when the slice touches them — preserves the exact-bit
    # value of `r.lo`/`r.hi` for full-axis or boundary-touching slices, which keeps
    # any downstream Tg-precision comparison stable.
    new_lo = i_lo == 1 ? r.lo : muladd(i_lo - 1, r.h, r.lo)
    new_hi = i_hi == r.len ? r.hi : muladd(i_hi - 1, r.h, r.lo)
    return _CachedRange{T, Tinv, Tag}(new_lo, new_hi, r.h, r.inv_h, new_len)
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

# Unit-step fast-path: `AbstractUnitRange` (`UnitRange`/`Base.OneTo`) has step ≡ 1 by
# type, so the grid is built from literal constants (`h = one(T)`, `inv_h = one(Tinv)`)
# with no runtime division — `inv` is only in the compile-time type
# `Tinv = typeof(inv(oneunit(T)))` (Float64 for `T=Int`, `T` for Float, per the
# `_CachedRange` contract; skips the generic path's runtime `inv(step(x))`).
@inline function _to_float(x::AbstractUnitRange, ::Type{T}) where {T}
    Tinv = typeof(inv(oneunit(T)))
    return _CachedRange{T, Tinv, _UnitStep}(T(first(x)), T(last(x)), one(T), one(Tinv), length(x))
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
function _to_float(x::_CachedRange{S, Si, Tag}, ::Type{T}) where {S, Si, Tag, T}
    h = T(x.h)
    inv_h = inv(h)
    return _CachedRange{T, typeof(inv_h), Tag}(T(x.lo), T(x.hi), h, inv_h, x.len)
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
@inline function _to_float_adding_endpoint(x::_CachedRange{T, Tinv, Tag}, ::Type{T}) where {T, Tinv, Tag}
    hi_new = x.hi + x.h
    return _CachedRange{T, Tinv, Tag}(
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

# `_get_h` / `_get_inv_h` accessors. A `_CachedRange` is uniform, so one cached
# `h`/`inv_h` answers every cell: all shapes delegate to the no-arg form, where the
# `_UnitStep` (h ≡ inv_h ≡ 1) literal `one(T)` lives — LLVM then folds every
# downstream `×h`/`×inv_h` to identity.
@inline _get_h(x::_CachedRange) = x.h
@inline _get_inv_h(x::_CachedRange) = x.inv_h
@inline _get_h(::_CachedRange{T, Tinv, _UnitStep}) where {T, Tinv} = one(T)
@inline _get_inv_h(::_CachedRange{T, Tinv, _UnitStep}) where {T, Tinv} = one(Tinv)
# Inverse of the 2-cell span x[i+1]-x[i-1] (central-difference / cardinal interior).
# A uniform range has span = 2h, so inv = inv_h/2; `_UnitStep` folds to one/2 = 0.5
# (no division). 0.5 is exact (power of two) → one cached load + one multiply.
@inline function _get_inv_2cell(x::_CachedRange, i::Int)
    inv_h = _get_inv_h(x, i)
    return inv_h * oftype(inv_h, 0.5)
end
# idx-shaped forms — `(x, idx)` (solver/coeff) and `(x, idx, xL, xR)` (from
# `search_interval`) — ignore the extra args and delegate to the no-arg form.
@inline _get_h(x::_CachedRange, ::Int) = _get_h(x)
@inline _get_inv_h(x::_CachedRange, ::Int) = _get_inv_h(x)
@inline _get_h(x::_CachedRange, ::Int, ::Real, ::Real) = _get_h(x)
@inline _get_inv_h(x::_CachedRange, ::Int, ::Real, ::Real) = _get_inv_h(x)

# Raw `AbstractRange` (non-_CachedRange) fallback via `step()` — pre-normalization paths only.
@inline _get_h(x::AbstractRange, ::Int) = step(x)
@inline _get_inv_h(x::AbstractRange, ::Int) = inv(step(x))

# ========================================
# `_resolve_axis` — one-shot Range wrapping
# ========================================
@inline _resolve_axis(x::AbstractRange) = _to_float(x, float(eltype(x)))
@inline _resolve_axis(x::AbstractRange, ::AbstractBC) = _to_float(x, float(eltype(x)))
@inline _resolve_axis(c::_CachedRange) = c

# `:exclusive` raw-input one-shot path — wrap into `_ExclusivePeriodicAxis`.
@inline function _resolve_axis(x::AbstractRange, bc::PeriodicBC{:exclusive})
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_to_float(x, float(eltype(x))), bc_resolved.period)
end

# ========================================
# `_cache_axis` — persistent-path Range wrapping
# ========================================
# Full DISPATCH TABLE for `_cache_axis(x, bc, Tg)` (cross-input-type) lives in
# `periodic_axis.jl` as the central reference. Owner-file entries below.
@inline _cache_axis(x::AbstractRange) = _to_float(x, float(eltype(x)))
@inline _cache_axis(x::AbstractRange, ::AbstractBC) = _to_float(x, float(eltype(x)))
@inline _cache_axis(c::_CachedRange) = c
@inline _cache_axis(c::_CachedRange, ::AbstractBC) = c

# `:exclusive` 2-arg variants — produce `_ExclusivePeriodicAxis`.
@inline function _cache_axis(x::AbstractRange, bc::PeriodicBC{:exclusive})
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_to_float(x, float(eltype(x))), bc_resolved.period)
end
@inline function _cache_axis(c::_CachedRange, bc::PeriodicBC{:exclusive})
    bc_resolved = _resolve_bc_period(c, bc)
    return _ExclusivePeriodicAxis(c, bc_resolved.period)
end

# 3-arg Tg-aware. Raw Range respects Tg via `_to_float`. Pre-wrapped passes
# through — downstream `_convert_copy(_, Tg)` enforces Tg (intentional contract;
# see DISPATCH TABLE in `periodic_axis.jl`).
@inline _cache_axis(x::AbstractRange, ::AbstractBC, ::Type{Tg}) where {Tg} = _to_float(x, Tg)
@inline _cache_axis(c::_CachedRange, bc::AbstractBC, ::Type{Tg}) where {Tg} = _cache_axis(c, bc)
@inline function _cache_axis(x::AbstractRange, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_to_float(x, Tg), bc_resolved.period)
end
@inline _cache_axis(c::_CachedRange, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg} = _cache_axis(c, bc)

# ========================================
# `_cache_axis_pooled` — one-shot Range wrapping
# ========================================
# Range types have stack-only `_CachedRange` — no pool buffer needed.
@inline _cache_axis_pooled(_, x::AbstractRange) = _to_float(x, float(eltype(x)))
@inline _cache_axis_pooled(_, x::_CachedRange) = x
