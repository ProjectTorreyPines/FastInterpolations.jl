# ========================================
# Akima 1D Oneshot API
# ========================================
# Public API: akima_interp(x, y, xq; coeffs=PreCompute(), ...)
# Routes to internal _akima_interp_precompute (bulk slopes + pool)
# or _akima_interp_onthefly (local slopes, no pool).

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: PreCompute (bulk slopes via @with_pool)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar — bc-aware unified path. Axis-as-truth via `_resolve_axis`.
@inline @with_pool pool function _akima_interp_precompute(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        bc::AbstractBC,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    # Grid pre-normalized by the public `akima_interp` API via `_resolve_axis(x)`
    # before dispatching here; `_periodic_extend_1d` preserves the normalization.
    x_eff, y_ext, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)
    # Value-matched width: dy buffer + slope arithmetic run at `Tw` — see pchip_oneshot.jl.
    Tw = _promote_grid_float(eltype(x_eff), Tv)
    Tdy = _promote_eltype(_coeff_op, Tw, Tv)
    dy = acquire!(pool, Tdy, length(y_ext))
    _akima_slopes!(dy, x_eff, y_ext, Tw; bc = bc_eff)
    searcher = _resolve_search(x_eff, xq, search, hint)
    return _hermite_eval_at_point(x_eff, y_ext, dy, xq, extrap_eff, deriv, searcher)
end

# Vector in-place — bc-aware unified path.
@inline @with_pool pool function _akima_interp_precompute!(
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractArray,
        bc::AbstractBC,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    _check_query_output_size(output, x_query)
    x_eff, y_ext, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)

    # Value-matched width: dy buffer + slope arithmetic run at `Tw` — see pchip_oneshot.jl.
    Tw = _promote_grid_float(eltype(x_eff), Tv)
    Tdy = _promote_eltype(_coeff_op, Tw, Tv)
    dy = acquire!(pool, Tdy, length(y_ext))
    _akima_slopes!(dy, x_eff, y_ext, Tw; bc = bc_eff)
    searcher = _resolve_search(x_eff, x_query, search, hint)
    return _hermite_vector_loop!(output, x_eff, y_ext, dy, x_query, extrap_eff, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║         INTERNAL: OnTheFly (local slopes, no pool, bc-aware)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar — bc-aware unified path.
@inline function _akima_interp_onthefly(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        bc::AbstractBC,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Akima interpolation requires at least 2 points, got $(length(x))"))
    x_eff = _resolve_axis(x, bc)
    y_eff = _resolve_data(y, bc)

    searcher = _resolve_search(x_eff, xq, search, hint)
    return _hermite_eval_at_point(x_eff, y_eff, AkimaSlopes(bc), xq, extrap, deriv, searcher)
end

# Vector in-place — bc-aware unified path.
@inline function _akima_interp_onthefly!(
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractArray,
        bc::AbstractBC,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Akima interpolation requires at least 2 points, got $(length(x))"))
    _check_query_output_size(output, x_query)
    x_eff = _resolve_axis(x, bc)
    y_eff = _resolve_data(y, bc)


    searcher = _resolve_search(x_eff, x_query, search, hint)
    return _hermite_vector_loop!(output, x_eff, y_eff, AkimaSlopes(bc), x_query, extrap, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    PUBLIC API — typed entry points                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    akima_interp(x, y, xq; coeffs=PreCompute(), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

Akima interpolation at a single query point.

Outlier-robust, C\$^1\$ continuous.

# Keyword Arguments
- `coeffs`: Coefficient strategy — `PreCompute()` (bulk slopes) or `OnTheFly()` (local slopes per cell)
"""
@inline function akima_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq;
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    # Value-matched Tg: Int/OneTo grid + Float32 data → Float32 axis.
    x = _resolve_axis(x, _promote_grid_float(Tg, Tv))
    extrap_eff = _resolve_extrap(extrap, bc, x, y)
    resolved = _resolve_coeffs(coeffs, x, xq)
    if resolved isa OnTheFly
        return _akima_interp_onthefly(x, y, xq, bc, extrap_eff, deriv, search, hint)
    end
    return _akima_interp_precompute(x, y, xq, bc, extrap_eff, deriv, search, hint)
end

"""
    akima_interp!(output, x, y, x_query; coeffs=PreCompute(), ...)

In-place Akima interpolation with outlier-robust slopes.
"""
@inline function akima_interp!(
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractArray{Tq};
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x = _resolve_axis(x, _promote_grid_float(Tg, Tv))
    extrap_eff = _resolve_extrap(extrap, bc, x, y)
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _akima_interp_onthefly!(output, x, y, x_query, bc, extrap_eff, deriv, search, hint)
    end
    return _akima_interp_precompute!(output, x, y, x_query, bc, extrap_eff, deriv, search, hint)
end

"""
    akima_interp(x, y, x_query; coeffs=PreCompute(), ...)

Akima interpolation at multiple query points. Returns `Vector`.
"""
function akima_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractArray{Tq};
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    Tr = _promote_eltype(_interp_op, _promote_grid_float(Tg, Tv), Tv, Tq)
    output = _alloc_query_output(Tr, x_query)
    akima_interp!(output, x, y, x_query; bc = bc, coeffs = coeffs, extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end
