# ========================================
# Cardinal Spline 1D Oneshot API
# ========================================
# Public API: cardinal_interp(x, y, xq; coeffs=AutoCoeffs(), tension=0.0, ...)
# Routes to internal _cardinal_interp_precompute (bulk slopes + pool)
# or _cardinal_interp_onthefly (local slopes, no pool).

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: PreCompute (bulk slopes via @with_pool)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar
@inline @with_pool pool function _cardinal_interp_precompute(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        tension::Real,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    x = _prepare_grid(x)
    Tdy = _output_eltype(Tv, float(eltype(x)))
    dy = acquire!(pool, Tdy, length(y))
    _cardinal_slopes!(dy, x, y, tension)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# Vector in-place
@inline @with_pool pool function _cardinal_interp_precompute!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector,
        tension::Real,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x = _prepare_grid(x)

    Tdy = _output_eltype(Tv, float(eltype(x)))
    dy = acquire!(pool, Tdy, length(y))
    _cardinal_slopes!(dy, x, y, tension)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: OnTheFly (local slopes, no pool)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar
@inline function _cardinal_interp_onthefly(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        tension::Real,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
    x = _prepare_grid(x)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, CardinalSlopes(tension), xq, extrap, deriv, searcher)
end

# Vector in-place
@inline function _cardinal_interp_onthefly!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector,
        tension::Real,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x = _prepare_grid(x)

    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, CardinalSlopes(tension), x_query, extrap, deriv, searcher)
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
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x = _prepare_grid(x)
    tension_f = float(eltype(x))(tension)
    resolved = _resolve_coeffs(coeffs, x, xq)
    if resolved isa OnTheFly
        return _cardinal_interp_onthefly(x, y, xq, tension_f, extrap, deriv, search, hint)
    end
    return _cardinal_interp_precompute(x, y, xq, tension_f, extrap, deriv, search, hint)
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
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x = _prepare_grid(x)
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _cardinal_interp_onthefly!(output, x, y, x_query, float(eltype(x))(tension), extrap, deriv, search, hint)
    end
    return _cardinal_interp_precompute!(output, x, y, x_query, float(eltype(x))(tension), extrap, deriv, search, hint)
end

"""
    cardinal_interp(x, y, x_query; coeffs=AutoCoeffs(), tension=0.0, ...)

Cardinal spline interpolation at multiple query points. Returns `Vector`.
"""
function cardinal_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tq};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    Tr = _output_eltype(Tv, _promote_grid_float(Tg, Tv), Tq)
    output = Vector{Tr}(undef, length(x_query))
    cardinal_interp!(output, x, y, x_query; coeffs = coeffs, tension = tension, extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end
