# ========================================
# Cardinal Spline 1D Oneshot API
# ========================================
# Public API: cardinal_interp(x, y, xq; coeffs=AutoCoeffs(), tension=0.0, ...)
# Routes to internal _cardinal_interp_precompute (bulk slopes + pool)
# or _cardinal_interp_onthefly (local slopes, no pool).

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: PreCompute (bulk slopes via @with_pool)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar — periodic BC physically extends grid via `_periodic_extend_1d`
# (length n+1 with closed cycle); the public API has already normalized the
# `_get_h(x, idx)` reads cached h. Slope kernels need physical data of length
# n+1 — wrapper-based axis (`_ExclusivePeriodicAxis`) only supports virtual
# n+1 access via `_getindex`, which the slope kernels don't use.
@inline @with_pool pool function _cardinal_interp_precompute(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        bc::AbstractBC,
        tension::Real,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    # `_periodic_extend_1d` already returns a normalized grid (Range or
    # `_CachedRange`/Vector) — the public `cardinal_interp` API pre-resolved
    # via `_resolve_axis(x)` before dispatching here, so no extra prep needed.
    x_eff, y_ext, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)
    # Value-matched width: dy buffer + slope arithmetic (incl. the `1 - tension`
    # scale) run at `Tw` — see pchip_oneshot.jl.
    Tw = _promote_grid_float(eltype(x_eff), Tv)
    Tdy = _promote_eltype(_coeff_op, Tw, Tv)
    dy = acquire!(pool, Tdy, length(y_ext))
    _cardinal_slopes!(dy, x_eff, y_ext, tension, Tw; bc = bc_eff)
    searcher = _resolve_search(x_eff, xq, search, hint)
    return _hermite_eval_at_point(x_eff, y_ext, dy, xq, extrap_eff, deriv, searcher)
end

# Vector in-place — periodic BC follows the same extend-then-eval pattern.
@inline @with_pool pool function _cardinal_interp_precompute!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector,
        bc::AbstractBC,
        tension::Real,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x_eff, y_ext, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)

    # Value-matched width: dy buffer + slope arithmetic (incl. the `1 - tension`
    # scale) run at `Tw` — see pchip_oneshot.jl.
    Tw = _promote_grid_float(eltype(x_eff), Tv)
    Tdy = _promote_eltype(_coeff_op, Tw, Tv)
    dy = acquire!(pool, Tdy, length(y_ext))
    _cardinal_slopes!(dy, x_eff, y_ext, tension, Tw; bc = bc_eff)
    searcher = _resolve_search(x_eff, x_query, search, hint)
    return _hermite_vector_loop!(output, x_eff, y_ext, dy, x_query, extrap_eff, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: OnTheFly (local slopes, no pool)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar — axis-as-truth: `_resolve_axis(x, bc)` wraps the axis so
# `last(x_eff) = first(x) + period` for periodic exclusive, giving the eval
# kernel correct wrap-domain bounds via `_wrap_to_domain(xq, x_eff)` and
# correct seam-cell `_get_h(x_eff, idx)`. Slopes use `_data_length(x_eff)`
# (raw n) for boundary detection, so `CardinalSlopes(tension, bc)`'s bc-aware
# wrap-formulas fire at i==n_raw without any `x[n+1]` access.
@inline function _cardinal_interp_onthefly(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        bc::AbstractBC,
        tension::Real,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
    # Wrap axis + data (axis-as-truth: `last(x_eff) == first(x) + period`,
    # so `_wrap_to_domain(xq, x_eff)` and `_get_h(x_eff, idx)` correctly hit
    # the seam without needing bc as a side-channel). Keep `bc` flowing into
    # `CardinalSlopes(tension, bc)` so the `:exclusive` slope dispatch fires
    # the seam formula via `bc.period` — the slope helper does NOT touch
    # `x[n+1]`, so `Base.getindex` raw passthrough on the wrapper is safe.
    x_eff = _resolve_axis(x, bc)
    y_eff = _resolve_data(y, bc)
    searcher = _resolve_search(x_eff, xq, search, hint)
    return _hermite_eval_at_point(x_eff, y_eff, CardinalSlopes(tension, bc), xq, extrap, deriv, searcher)
end

# Vector in-place — same axis-as-truth pattern.
@inline function _cardinal_interp_onthefly!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector,
        bc::AbstractBC,
        tension::Real,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x_eff = _resolve_axis(x, bc)
    y_eff = _resolve_data(y, bc)
    searcher = _resolve_search(x_eff, x_query, search, hint)
    return _hermite_vector_loop!(output, x_eff, y_eff, CardinalSlopes(tension, bc), x_query, extrap, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    TYPED CORE — all grid types                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    cardinal_interp(x, y, xq; coeffs=AutoCoeffs(), tension=0.0, extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

Cardinal spline interpolation at a single query point.
Default `tension=0` is Catmull-Rom. C\$^1\$ continuous.

# Keyword Arguments
- `coeffs`: Coefficient strategy — `PreCompute()` (bulk slopes) or `OnTheFly()` (local slopes per cell)
- `tension`: Cardinal tension parameter (0 = Catmull-Rom, 1 = linear)
"""
@inline function cardinal_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq;
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    # Value-matched Tg: Int/OneTo grid + Float32 data → Float32 axis (tension follows).
    x = _resolve_axis(x, _promote_grid_float(Tg, Tv))
    tension_f = float(eltype(x))(tension)
    extrap_eff = _resolve_extrap(extrap, bc, x, y)
    resolved = _resolve_coeffs(coeffs, x, xq)
    if resolved isa OnTheFly
        return _cardinal_interp_onthefly(x, y, xq, bc, tension_f, extrap_eff, deriv, search, hint)
    end
    return _cardinal_interp_precompute(x, y, xq, bc, tension_f, extrap_eff, deriv, search, hint)
end

"""
    cardinal_interp!(output, x, y, x_query; coeffs=AutoCoeffs(), tension=0.0, ...)

In-place cardinal spline interpolation.
"""
@inline function cardinal_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tq};
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x = _resolve_axis(x, _promote_grid_float(Tg, Tv))
    tension_f = float(eltype(x))(tension)
    extrap_eff = _resolve_extrap(extrap, bc, x, y)
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _cardinal_interp_onthefly!(output, x, y, x_query, bc, tension_f, extrap_eff, deriv, search, hint)
    end
    return _cardinal_interp_precompute!(output, x, y, x_query, bc, tension_f, extrap_eff, deriv, search, hint)
end

"""
    cardinal_interp(x, y, x_query; coeffs=AutoCoeffs(), tension=0.0, ...)

Cardinal spline interpolation at multiple query points. Returns `Vector`.
"""
function cardinal_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tq};
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    Tr = _promote_eltype(_interp_op, _promote_grid_float(Tg, Tv), Tv, Tq)
    output = Vector{Tr}(undef, length(x_query))
    cardinal_interp!(output, x, y, x_query; bc = bc, coeffs = coeffs, tension = tension, extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end
