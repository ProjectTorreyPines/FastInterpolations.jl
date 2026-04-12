# ========================================
# PCHIP 1D Oneshot API
# ========================================
# Public API: pchip_interp(x, y, xq; coeffs=PreCompute(), ...)
# Routes to internal _pchip_interp_precompute (bulk slopes + pool)
# or _pchip_interp_onthefly (local slopes, no pool).

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: PreCompute (bulk slopes via @with_pool)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar
@inline @with_pool pool function _pchip_interp_precompute(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    Tg_f = float(Tg)
    x = _to_float(x, Tg_f)
    Tdy = _output_eltype(Tv, Tg)
    dy = acquire!(pool, Tdy, length(y))
    _pchip_slopes!(dy, x, y)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# Vector in-place
@inline @with_pool pool function _pchip_interp_precompute!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    Tg_f = float(Tg)
    x = _to_float(x, Tg_f)

    Tdy = _output_eltype(Tv, Tg)
    dy = acquire!(pool, Tdy, length(y))
    _pchip_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# Range disambiguation for in-place
@inline @with_pool pool function _pchip_interp_precompute!(
        output::AbstractVector,
        x::AbstractRange{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    Tg_f = float(Tg)
    x = _to_float(x, Tg_f)

    Tdy = _output_eltype(Tv, Tg)
    dy = acquire!(pool, Tdy, length(y))
    _pchip_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: OnTheFly (local slopes, no pool)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar
@inline function _pchip_interp_onthefly(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("PCHIP interpolation requires at least 2 points, got $(length(x))"))
    x = _to_float(x, Tg)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, PchipSlopes(), xq, extrap, deriv, searcher)
end

# Vector in-place
@inline function _pchip_interp_onthefly!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x = _to_float(x, Tg)

    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, PchipSlopes(), x_query, extrap, deriv, searcher)
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
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x, y = _promote_itp_inputs(x, y)
    resolved = _resolve_coeffs(coeffs, x, xq)
    if resolved isa OnTheFly
        return _pchip_interp_onthefly(x, y, xq, extrap, deriv, search, hint)
    end
    return _pchip_interp_precompute(x, y, xq, extrap, deriv, search, hint)
end

"""
    pchip_interp!(output, x, y, x_query; coeffs=PreCompute(), ...)

In-place PCHIP interpolation with monotone-preserving slopes.
"""
@inline function pchip_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tq};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x = _prepare_grid(x)
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _pchip_interp_onthefly!(output, x, y, x_query, extrap, deriv, search, hint)
    end
    return _pchip_interp_precompute!(output, x, y, x_query, extrap, deriv, search, hint)
end

"""
    pchip_interp(x, y, x_query; coeffs=PreCompute(), ...)

PCHIP interpolation at multiple query points. Returns `Vector`.
"""
function pchip_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tq};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    Tr = _output_eltype(Tv, _promote_grid_float(Tg, Tv), Tq)
    output = Vector{Tr}(undef, length(x_query))
    pchip_interp!(output, x, y, x_query; coeffs = coeffs, extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end
