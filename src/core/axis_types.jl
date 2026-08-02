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
- `domain_lo::T`, `domain_hi::T` — domain bracket read by `_domain_bounds` only for
  a `_WidenedDomain` grid (widened ≈1 ULP on the x86_64 TwicePrecision fast path);
  carried equal to `lo`/`hi` for `_Generic`/`_UnitStep`, which read `lo`/`hi` directly
"""
# Grid tag (3rd type param): a type-level marker for grid properties, kept open so
# future kinds (log-spaced, reversed, …) — and `_CachedVector` — can reuse
# `_AbstractAxisTag`. The `_AbstractUnitStep` family (mirrors Base's AbstractUnitRange)
# pins step ≡ 1, folding ×h/×inv_h through the accessors; `_OneTo` additionally pins
# `lo ≡ 1` (`first` returns a literal → `lo == 1` tests constant-fold). `_WidenedDomain`
# lets `_domain_bounds` read the widened x86_64 bracket instead of `lo`/`hi`. Every
# other call site dispatches on `::_CachedRange{T,Tinv}`, which matches all tags.
abstract type _AbstractAxisTag end
struct _Generic <: _AbstractAxisTag end       # default: step is a runtime field
abstract type _AbstractUnitStep <: _AbstractAxisTag end
struct _UnitStep <: _AbstractUnitStep end     # step ≡ 1 (from an AbstractUnitRange grid)
struct _OneTo <: _AbstractUnitStep end        # step ≡ 1 AND lo ≡ 1 (from a Base.OneTo grid)
# Widened 1-ULP domain bracket (x86_64 TwicePrecision reconstruction cushion);
# mutually exclusive with the unit-step family.
struct _WidenedDomain <: _AbstractAxisTag end

struct _CachedRange{T, Tinv, Tag <: _AbstractAxisTag} <: AbstractRange{T}
    lo::T
    hi::T
    h::T
    inv_h::Tinv
    len::Int
    domain_lo::T
    domain_hi::T
end

# Construction factory — the single home of domain-bracket construction: build a
# `_CachedRange` from raw geometry, deriving `domain_lo`/`domain_hi` from the tag
# *instance* (idiomatic trait dispatch, as with `NoExtrap()` / `EvalValue()`). Fresh
# conversion, slicing, and diff-type rebuilds all route through it, so a tag's domain
# invariant always holds — a `_WidenedDomain` range is *always* widened, never an
# exact bracket under a widened tag. The generic method recovers the concrete tag via
# `typeof(tag)` (constant-folded under specialization); `T`/`Tinv` are inferred from
# the value arguments, so call sites carry no type parameters. The raw 7-arg inner
# ctor stays for explicit-domain rebuilds (e.g. the asymmetric bracket in
# `_to_float_adding_endpoint`).
@inline function _cached_range(tag::_AbstractAxisTag, lo::T, hi::T, h::T, inv_h::Tinv, len::Int) where {T, Tinv}
    return _CachedRange{T, Tinv, typeof(tag)}(lo, hi, h, inv_h, len, lo, hi)
end
@inline function _cached_range(::_WidenedDomain, lo::T, hi::T, h::T, inv_h::Tinv, len::Int) where {T, Tinv}
    return _CachedRange{T, Tinv, _WidenedDomain}(lo, hi, h, inv_h, len, prevfloat(lo), nextfloat(hi))
end

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

# ─────────────────────────────────────────────────────────────────────────────
# _ReparamAxis — lazy dimensionless twin of a non-Real axis
# ─────────────────────────────────────────────────────────────────────────────
"""
    _ReparamAxis{T, X<:AbstractVector, W, U} <: AbstractVector{T}

Lazy `_reparam_op` view of a unit-carrying axis: element `i` is
`inner[i] * scale` with `scale = inv(oneunit(eltype(inner)))`.

The solver-family scaled store keeps `[Y]`-homogeneous partials, which pair
with DIMENSIONLESS cell widths — and both consumers (the fiber solve and the
separable integrate engine) only ever ask an axis for `getindex` and
`_get_h`/`_get_inv_h`. So the twin needs no array of its own: it is a view,
and its widths come from the parent's cache times a scalar.

Methods + the outer ctor live in `nd_utils.jl` beside `_reparam_grids`, the
only producer.

# Fields
- `inner::X` — the physical axis (possibly itself wrapped).
- `scale::W` — the canonical `_deriv_oneunit(…, DerivOp(1))` witness.
- `unit::U` — `oneunit(eltype(inner))`; restores `inv_h` without a division.
"""
struct _ReparamAxis{T, X <: AbstractVector, W, U} <: AbstractVector{T}
    inner::X
    scale::W
    unit::U
end

# Convenience outer ctor — type params inferred from inputs. The element type
# must hold the virtual seam point `inner[1] + period`, so widen a narrow grid
# eltype against the period type before wrapping (an Int grid with a float period
# would otherwise force `Int(period)` and throw). Zero-copy in the common
# float-grid case (the period is float, so the widen is a no-op there).
@inline function _ExclusivePeriodicAxis(inner::AbstractVector{Tg}, period) where {Tg}
    Te = promote_type(Tg, typeof(period))
    inner_e = _widen_axis_inner(inner, Te)
    return _ExclusivePeriodicAxis{Te, typeof(inner_e), typeof(period)}(inner_e, period)
end

# Widen the wrapped grid to `Te` only when needed — dispatch (not a `?:`) keeps
# the no-op case zero-copy and type-stable for the per-query one-shot path.
@inline _widen_axis_inner(inner::AbstractVector{Te}, ::Type{Te}) where {Te} = inner
@inline _widen_axis_inner(inner::AbstractVector, ::Type{Te}) where {Te} = _convert_copy(inner, Te)

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
