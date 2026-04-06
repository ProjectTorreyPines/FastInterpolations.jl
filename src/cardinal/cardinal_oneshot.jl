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
        tension::Tg,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    x = _to_float(x, Tg)
    dy = similar!(pool, y)
    _cardinal_slopes!(dy, x, y, tension)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# Vector in-place
@inline @with_pool pool function _cardinal_interp_precompute!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg},
        tension::Tg,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg <: AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    dy = similar!(pool, y)
    _cardinal_slopes!(dy, x, y, tension)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# Range disambiguation for in-place
@inline @with_pool pool function _cardinal_interp_precompute!(
        output::AbstractVector,
        x::AbstractRange{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg},
        tension::Tg,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg <: AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    dy = similar!(pool, y)
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
        tension::Tg,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
    x = _to_float(x, Tg)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, CardinalSlopes(tension), xq, extrap, deriv, searcher)
end

# Vector in-place
@inline function _cardinal_interp_onthefly!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg},
        tension::Tg,
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg <: AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, CardinalSlopes(tension), x_query, extrap, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    PUBLIC API — typed entry points                         ║
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
        tension::Real = zero(Tg),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    resolved = _resolve_coeffs(coeffs, x, xq)
    if resolved isa OnTheFly
        return _cardinal_interp_onthefly(x, y, xq, Tg(tension), extrap, deriv, search, hint)
    end
    return _cardinal_interp_precompute(x, y, xq, Tg(tension), extrap, deriv, search, hint)
end

"""
    cardinal_interp!(output, x, y, x_query; coeffs=AutoCoeffs(), tension=0.0, ...)

In-place cardinal spline interpolation.
"""
@inline function cardinal_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = zero(Tg),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _cardinal_interp_onthefly!(output, x, y, x_query, Tg(tension), extrap, deriv, search, hint)
    end
    return _cardinal_interp_precompute!(output, x, y, x_query, Tg(tension), extrap, deriv, search, hint)
end

# Range disambiguation for in-place
@inline function cardinal_interp!(
        output::AbstractVector,
        x::AbstractRange{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = zero(Tg),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _cardinal_interp_onthefly!(output, x, y, x_query, Tg(tension), extrap, deriv, search, hint)
    end
    return _cardinal_interp_precompute!(output, x, y, x_query, Tg(tension), extrap, deriv, search, hint)
end

"""
    cardinal_interp(x, y, x_query; coeffs=AutoCoeffs(), tension=0.0, ...)

Cardinal spline interpolation at multiple query points. Returns `Vector{Tv}`.
"""
function cardinal_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        tension::Real = zero(Tg),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    output = Vector{Tv}(undef, length(x_query))
    cardinal_interp!(output, x, y, x_query; coeffs = coeffs, tension = tension, extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  GENERIC WRAPPERS — Real type promotion                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar — promotes x, y to Float; passes xq directly for AD support
@inline function cardinal_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        xq::Tq;
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    x_p, y_p = _promote_itp_inputs(x, y)
    return cardinal_interp(x_p, y_p, xq; kwargs...)
end

# Vector — allocating
function cardinal_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    x_p, y_p, xq_p = _promote_itp_inputs(x, y, x_query)
    Tv_float = eltype(y_p)
    output = Vector{Tv_float}(undef, length(x_query))
    cardinal_interp!(output, x_p, y_p, xq_p; kwargs...)
    return output
end

# Vector — in-place
function cardinal_interp!(
        output::AbstractVector,
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x_p, y_p, xq_p = _promote_itp_inputs(x, y, x_query)
    Tv_float = eltype(y_p)

    Tout = eltype(output)
    if promote_type(Tout, Tv_float) !== Tout
        throw(
            ArgumentError(
                "output eltype $Tout cannot hold interpolation result type $Tv_float. " *
                    "Use Vector{$Tv_float} or a wider type."
            )
        )
    end

    return cardinal_interp!(output, x_p, y_p, xq_p; kwargs...)
end
