# ========================================
# PCHIP 1D Oneshot API
# ========================================
# Public API: pchip_interp(x, y, xq; coeffs=PreCompute(), ...)
# Routes to internal _pchip_interp_precompute (bulk slopes + pool)
# or _pchip_interp_onthefly (local slopes, no pool).

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║       INTERNAL: PreCompute (bulk slopes via @with_pool, bc-aware)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Single unified path. `bc` flows into the bulk slope routine and
# `_resolve_search`'s seam dispatch handles eval-time wrap. No grid extension
# is needed — same approach as Linear's zero-copy oneshot.

# Scalar — axis-as-truth via `_resolve_axis`/`_resolve_data`.
@inline @with_pool pool function _pchip_interp_precompute(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        bc::AbstractBC,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    # Grid pre-normalized by the public `pchip_interp` API via `_resolve_axis(x)`
    # before dispatching here; `_periodic_extend_1d` preserves the normalization.
    x_eff, y_ext, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)
    # Value-matched width: the dy buffer AND the slope arithmetic inside the
    # filler both run at `Tw` (raw Int axes stop minting `inv(Int)::Float64`).
    Tw = _promote_grid_float(eltype(x_eff), Tv)
    Tdy = _promote_eltype(_coeff_op, Tw, Tv)
    dy = acquire!(pool, Tdy, length(y_ext))
    _pchip_slopes!(dy, x_eff, y_ext, Tw; bc = bc_eff)
    searcher = _resolve_search(x_eff, xq, search, hint)
    return _hermite_eval_at_point(x_eff, y_ext, dy, xq, extrap_eff, deriv, searcher)
end

# Vector in-place
@inline @with_pool pool function _pchip_interp_precompute!(
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

    # Value-matched width: the dy buffer AND the slope arithmetic inside the
    # filler both run at `Tw` (raw Int axes stop minting `inv(Int)::Float64`).
    Tw = _promote_grid_float(eltype(x_eff), Tv)
    Tdy = _promote_eltype(_coeff_op, Tw, Tv)
    dy = acquire!(pool, Tdy, length(y_ext))
    _pchip_slopes!(dy, x_eff, y_ext, Tw; bc = bc_eff)
    searcher = _resolve_search(x_eff, x_query, search, hint)
    return _hermite_vector_loop!(output, x_eff, y_ext, dy, x_query, extrap_eff, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              INTERNAL: OnTheFly (local slopes, no pool, bc-aware)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# bc flows into PchipSlopes(bc), which carries the dispatch tag for boundary
# slope formulas. _resolve_search handles seam wrap.

# Scalar
@inline function _pchip_interp_onthefly(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        bc::AbstractBC,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("PCHIP interpolation requires at least 2 points, got $(length(x))"))
    x_eff = _resolve_axis(x, bc)
    y_eff = _resolve_data(y, bc)

    searcher = _resolve_search(x_eff, xq, search, hint)
    return _hermite_eval_at_point(x_eff, y_eff, PchipSlopes(bc), xq, extrap, deriv, searcher)
end

# Vector in-place
@inline function _pchip_interp_onthefly!(
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
    x_eff = _resolve_axis(x, bc)
    y_eff = _resolve_data(y, bc)


    searcher = _resolve_search(x_eff, x_query, search, hint)
    return _hermite_vector_loop!(output, x_eff, y_eff, PchipSlopes(bc), x_query, extrap, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    TYPED CORE — all grid types                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    pchip_interp(x, y, xq; coeffs=PreCompute(), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

PCHIP (monotone-preserving) interpolation at a single query point.

Computes Fritsch-Carlson slopes internally, then evaluates via cubic Hermite kernel.
C\$^1\$ continuous, monotonicity guaranteed for monotone input data.

# Keyword Arguments
- `coeffs`: Coefficient strategy — `PreCompute()` (bulk slopes) or `OnTheFly()` (local slopes per cell)
"""
@inline function pchip_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq;
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Number, Tv, Tq <: Number}
    _check_grid_orderable(Tg)
    # Unified entry — bc flows through the BC-aware extrap/search resolvers and
    # into the slope routines. No `_is_periodic_bc` branch, no extension copy:
    # `_resolve_search`'s seam dispatch + bc-aware slope formulas handle the
    # closed-cycle on the user's n-length grid (Linear pattern). Value-matched Tg:
    # Int/OneTo grid + Float32 data → Float32 axis.
    x = _resolve_axis(x, _promote_grid_float(Tg, Tv))
    extrap_eff = _resolve_extrap(extrap, bc, x, y)
    resolved = _resolve_coeffs(coeffs, x, xq)
    if resolved isa OnTheFly
        return _pchip_interp_onthefly(x, y, xq, bc, extrap_eff, deriv, search, hint)
    end
    return _pchip_interp_precompute(x, y, xq, bc, extrap_eff, deriv, search, hint)
end

"""
    pchip_interp!(output, x, y, x_query; coeffs=PreCompute(), ...)

In-place PCHIP interpolation with monotone-preserving slopes.
"""
@inline function pchip_interp!(
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
    ) where {Tg <: Number, Tv, Tq <: Number}
    _check_grid_orderable(Tg)
    x = _resolve_axis(x, _promote_grid_float(Tg, Tv))
    extrap_eff = _resolve_extrap(extrap, bc, x, y)
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _pchip_interp_onthefly!(output, x, y, x_query, bc, extrap_eff, deriv, search, hint)
    end
    return _pchip_interp_precompute!(output, x, y, x_query, bc, extrap_eff, deriv, search, hint)
end

"""
    pchip_interp(x, y, x_query; coeffs=PreCompute(), ...)

PCHIP interpolation at multiple query points. Returns an `Array`
matching the query's shape (a `Vector` for a vector query).
"""
function pchip_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractArray{Tq};
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Number, Tv, Tq <: Number}
    # Deriv-aware: an nth derivative lives in value/gridᴺ space (identity for `EvalValue`).
    Tw = _promote_grid_float(Tg, Tv)
    Tr = _deriv_eltype(_promote_eltype(_interp_op, Tw, Tv, Tq), Tw, deriv)
    output = _alloc_query_output(Tr, x_query)
    pchip_interp!(output, x, y, x_query; bc = bc, coeffs = coeffs, extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end
