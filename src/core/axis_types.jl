# ========================================
# Axis Types — central struct definitions
# ========================================
#
# All grid wrapper struct definitions live here, loaded early in the include
# chain so downstream method files can freely cross-reference any axis type
# without ordering constraints.
#
# Method dispatches live in owner files:
#   - `cached_range.jl`    → all `_CachedRange` methods (`_to_float`, `_get_h`,
#                            `_resolve_axis`, `_cache_axis*`, view/slice)
#   - `cached_vector.jl`   → all `_CachedVector` methods (build ctor,
#                            `_get_h`, `_resolve_axis`, `_cache_axis*`)
#   - `periodic_axis.jl`   → all `_ExclusivePeriodicAxis` methods (seam-aware
#                            `_get_h`, search, view, fold, dispatch table doc)
#
# Include order (in `core/core.jl`):
#   ... → eval_ops.jl → bc_types.jl → ... → axis_types.jl
#                                            → cached_range.jl
#                                            → cached_vector.jl
#                                            → ... → periodic_axis.jl

# ─────────────────────────────────────────────────────────────────────────────
# _CachedRange — normalized AbstractRange
# ─────────────────────────────────────────────────────────────────────────────
"""
    _CachedRange{T, Tinv} <: AbstractRange{T}

`AbstractRange` subtype that pre-caches `first`, `last`, `step`, and
`inv(step)` as plain `T`/`Tinv` fields. All `AbstractRange` user inputs are
normalized to `_CachedRange` via `_to_float` at public API boundaries.

`Tinv == typeof(inv(oneunit(T)))` — equals `T` for `T<:AbstractFloat`,
`Float64` for `T=Int`. Mirrors `_CachedVector{T,Tinv}`.

# Fields
- `lo::T`, `hi::T` — cached `first`/`last`
- `h::T`, `inv_h::Tinv` — cached `step` and reciprocal
- `len::Int` — cached length
- `domain_lo::T`, `domain_hi::T` — safe bounds for domain checks (= `lo`/`hi`
  on the exact path; widened by ≈1 ULP on the x86_64 TwicePrecision fast path)
"""
# Grid tag (3rd type param): a generic type-level marker for grid properties,
# kept open so future kinds (log-spaced, reversed, …) — and `_CachedVector` —
# can reuse `_AxisTag`. Only `_search_direct`/`_alpha_of` specialize on `_UnitStep`
# (step ≡ 1, statically known → skip the ×inv_h/×h multiplies). Every other call
# site dispatches on `::_CachedRange{T,Tinv}` (= `…{T,Tinv,Tag} where Tag`), which
# matches all tags, so nothing else needs to change.
abstract type _AxisTag end
struct _Generic <: _AxisTag end    # default: step is a runtime field
struct _UnitStep <: _AxisTag end   # step ≡ 1 (from an AbstractUnitRange grid)

struct _CachedRange{T, Tinv, Tag <: _AxisTag} <: AbstractRange{T}
    lo::T
    hi::T
    h::T
    inv_h::Tinv
    len::Int
    domain_lo::T
    domain_hi::T
end

# Default-tag convenience ctors: keep every existing `_CachedRange{T,Tinv}(…)`
# call site (and `::_CachedRange{T,Tinv}` dispatch) working — fills `Tag=_Generic`.
@inline _CachedRange{T, Tinv}(lo::T, hi::T, h::T, inv_h::Tinv, len::Int) where {T, Tinv} =
    _CachedRange{T, Tinv, _Generic}(lo, hi, h, inv_h, len, lo, hi)
@inline _CachedRange{T, Tinv}(lo::T, hi::T, h::T, inv_h::Tinv, len::Int, dlo::T, dhi::T) where {T, Tinv} =
    _CachedRange{T, Tinv, _Generic}(lo, hi, h, inv_h, len, dlo, dhi)
# 5-arg with explicit tag (exact domain = lo/hi) — used by the unit-step fast path.
@inline _CachedRange{T, Tinv, Tag}(lo::T, hi::T, h::T, inv_h::Tinv, len::Int) where {T, Tinv, Tag} =
    _CachedRange{T, Tinv, Tag}(lo, hi, h, inv_h, len, lo, hi)

# Identity passthrough — re-wrapping would discard the cached fields.
_CachedRange(x::_CachedRange) = x

# ─────────────────────────────────────────────────────────────────────────────
# _CachedVector — Vector grid with cached spacing
# ─────────────────────────────────────────────────────────────────────────────
"""
    _CachedVector{T, Tinv} <: AbstractVector{T}

Vector grid wrapper that caches per-cell spacing (`h`) and reciprocal
(`inv_h`) for fast O(1) lookup. `inner` is always a concrete `Vector{T}`
(see "Design choice" in `cached_vector.jl` outer ctor doc).
"""
struct _CachedVector{T, Tinv} <: AbstractVector{T}
    inner::Vector{T}
    h::Vector{T}        # length n-1
    inv_h::Vector{Tinv} # length n-1
end

# Identity passthrough — re-wrapping would discard the cached `h`/`inv_h`.
_CachedVector(c::_CachedVector) = c

# ─────────────────────────────────────────────────────────────────────────────
# _ExclusivePeriodicAxis — virtual length-(n+1) wrap for `:exclusive` PeriodicBC
# ─────────────────────────────────────────────────────────────────────────────
"""
    _ExclusivePeriodicAxis{Tg, X<:AbstractVector{Tg}, Tp} <: AbstractVector{Tg}

Representation wrapper for `PeriodicBC{:exclusive, L}` non-uniform Vector
*axis* grids. Reports `length == n+1` so search algorithms find the seam
cell. `inner[i]` for `1 ≤ i ≤ n` returns the raw grid; the virtual seam
slot returns `inner[1] + period`.

# Fields
- `inner::X` — raw axis (length n). Typically `Vector{Tg}` / `_CachedVector` /
  `AbstractRange` / `_CachedRange`.
- `period::Tp` — period span (type independent of `Tg`).
- `_x_max::Tg` — precomputed `inner[1] + Tg(period)`; single-load `last(g)`.
"""
struct _ExclusivePeriodicAxis{Tg, X <: AbstractVector{Tg}, Tp} <: AbstractVector{Tg}
    inner::X
    period::Tp
    _x_max::Tg

    function _ExclusivePeriodicAxis{Tg, X, Tp}(inner::X, period::Tp) where {Tg, X <: AbstractVector{Tg}, Tp}
        x_max = @inbounds inner[1] + Tg(period)
        @inbounds inner[end] < x_max || _throw_excl_axis_period_too_small(period, x_max, inner[end])
        _validate_exclusive_period(inner, period)
        return new{Tg, X, Tp}(inner, period, x_max)
    end
end

# Convenience outer ctor — type params inferred from inputs.
@inline _ExclusivePeriodicAxis(inner::AbstractVector{Tg}, period) where {Tg} =
    _ExclusivePeriodicAxis{Tg, typeof(inner), typeof(period)}(inner, period)

# Inner-ctor validation. No-op for Vector inners (period unverifiable);
# Range inners cross-validate against `step × length` (= one period for the
# n-cell exclusive form). Same numerical contract as
# `_check_periodic_endpoints` in `periodic.jl`.
#
# Dependencies (`_PromotableValue`, `_extract_primal`) are resolved at first
# call, not at struct-def time — both are in scope before any ctor invocation.
@inline _validate_exclusive_period(::AbstractVector, _) = nothing
@inline _validate_exclusive_period(inner::AbstractRange, period) =
    _validate_exclusive_period_impl(inner, period, eltype(inner))

@inline function _validate_exclusive_period_impl(
        inner::AbstractRange, period, ::Type{T}
    ) where {T <: AbstractFloat}
    inferred = step(inner) * length(inner)
    isapprox(T(period), T(inferred); atol = 8 * eps(T), rtol = sqrt(eps(T))) ||
        _throw_excl_axis_period_mismatch(period, first(inner), inferred)
    return nothing
end

@inline function _validate_exclusive_period_impl(
        inner::AbstractRange, period, ::Type{Complex{T}}
    ) where {T <: AbstractFloat}
    inferred = step(inner) * length(inner)
    isapprox(Complex{T}(period), Complex{T}(inferred); atol = 8 * eps(T), rtol = sqrt(eps(T))) ||
        _throw_excl_axis_period_mismatch(period, first(inner), inferred)
    return nothing
end

@inline function _validate_exclusive_period_impl(
        inner::AbstractRange, period, ::Type{T}
    ) where {T <: _PromotableValue}
    # Integer / Rational grids: exact equality (default `isapprox` tolerance).
    inferred = step(inner) * length(inner)
    isapprox(period, inferred) ||
        _throw_excl_axis_period_mismatch(period, first(inner), inferred)
    return nothing
end

@inline function _validate_exclusive_period_impl(inner::AbstractRange, period, ::Type)
    # Duck-type fallback (Dual, Measurement, ...): strict equality on primal.
    inferred = step(inner) * length(inner)
    _extract_primal(period) == _extract_primal(inferred) ||
        _throw_excl_axis_period_mismatch(period, first(inner), inferred)
    return nothing
end

@noinline _throw_excl_axis_period_too_small(period, x_max, last_x) = throw(
    ArgumentError(
        "PeriodicBC(:exclusive) period=$period places virtual endpoint at $x_max, " *
            "not after last grid point x[end]=$last_x"
    )
)

@noinline _throw_excl_axis_period_mismatch(period, x0, inferred) = throw(
    ArgumentError(
        "PeriodicBC's period=$period conflicts with Range-inferred period = " *
            "$(x0 + inferred) - $x0 = $inferred. " *
            "Either adjust `period` or omit it for auto-inference."
    )
)
