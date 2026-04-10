# ========================================
# Akima 1D Oneshot API
# ========================================
# Public API: akima_interp(x, y, xq; coeffs=PreCompute(), ...)
# Routes to internal _akima_interp_precompute (bulk slopes + pool)
# or _akima_interp_onthefly (local slopes, no pool).

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: PreCompute (bulk slopes via @with_pool)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar
@inline @with_pool pool function _akima_interp_precompute(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    x = _to_float(x, Tg)
    Tdy = _output_eltype(Tv, Tg)
    dy = acquire!(pool, Tdy, length(y))
    _akima_slopes!(dy, x, y)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# Vector in-place
@inline @with_pool pool function _akima_interp_precompute!(
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

    Tdy = _output_eltype(Tv, Tg)
    dy = acquire!(pool, Tdy, length(y))
    _akima_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# Range disambiguation for in-place
@inline @with_pool pool function _akima_interp_precompute!(
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
    x = _to_float(x, Tg)

    Tdy = _output_eltype(Tv, Tg)
    dy = acquire!(pool, Tdy, length(y))
    _akima_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: OnTheFly (local slopes, no pool)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar
@inline function _akima_interp_onthefly(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Akima interpolation requires at least 2 points, got $(length(x))"))
    x = _to_float(x, Tg)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, AkimaSlopes(), xq, extrap, deriv, searcher)
end

# Vector in-place
@inline function _akima_interp_onthefly!(
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
    length(x) >= 2 || throw(ArgumentError("Akima interpolation requires at least 2 points, got $(length(x))"))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x = _to_float(x, Tg)

    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, AkimaSlopes(), x_query, extrap, deriv, searcher)
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
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x, y = _promote_itp_inputs(x, y)
    resolved = _resolve_coeffs(coeffs, x, xq)
    if resolved isa OnTheFly
        return _akima_interp_onthefly(x, y, xq, extrap, deriv, search, hint)
    end
    return _akima_interp_precompute(x, y, xq, extrap, deriv, search, hint)
end

"""
    akima_interp!(output, x, y, x_query; coeffs=PreCompute(), ...)

In-place Akima interpolation with outlier-robust slopes.
"""
@inline function akima_interp!(
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
    x, y = _promote_itp_inputs(x, y)
    Tg_actual = eltype(x)
    Tq_float = Tg_actual <: AbstractFloat ? Tg_actual : float(Tq)
    xq_p = _to_float(x_query, Tq_float)
    resolved = _resolve_coeffs(coeffs, x, xq_p)
    if resolved isa OnTheFly
        return _akima_interp_onthefly!(output, x, y, xq_p, extrap, deriv, search, hint)
    end
    return _akima_interp_precompute!(output, x, y, xq_p, extrap, deriv, search, hint)
end

"""
    akima_interp(x, y, x_query; coeffs=PreCompute(), ...)

Akima interpolation at multiple query points. Returns `Vector`.
"""
function akima_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tq};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x, y = _promote_itp_inputs(x, y)
    Tg_actual = eltype(x)
    Tq_float = Tg_actual <: AbstractFloat ? Tg_actual : float(Tq)
    xq_p = _to_float(x_query, Tq_float)
    Tr = _output_eltype(eltype(y), Tg_actual, Tq)
    output = Vector{Tr}(undef, length(xq_p))
    akima_interp!(output, x, y, xq_p; coeffs = coeffs, extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end
