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
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    x = _to_float(x, Tg)
    dy = similar!(pool, y)
    _pchip_slopes!(dy, x, y)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# Vector in-place
@inline @with_pool pool function _pchip_interp_precompute!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg},
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
    _pchip_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# Range disambiguation for in-place
@inline @with_pool pool function _pchip_interp_precompute!(
        output::AbstractVector,
        x::AbstractRange{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg},
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
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    x = _to_float(x, Tg)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, PchipSlopes(), xq, extrap, deriv, searcher)
end

# Vector in-place
@inline function _pchip_interp_onthefly!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg},
        extrap::AbstractExtrap,
        deriv::DerivOp,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}}
    ) where {Tg <: AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")
    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, PchipSlopes(), x_query, extrap, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    PUBLIC API — typed entry points                         ║
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
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
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
        x_query::AbstractVector{Tg};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _pchip_interp_onthefly!(output, x, y, x_query, extrap, deriv, search, hint)
    end
    return _pchip_interp_precompute!(output, x, y, x_query, extrap, deriv, search, hint)
end

# Range disambiguation for in-place
@inline function pchip_interp!(
        output::AbstractVector,
        x::AbstractRange{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    resolved = _resolve_coeffs(coeffs, x, x_query)
    if resolved isa OnTheFly
        return _pchip_interp_onthefly!(output, x, y, x_query, extrap, deriv, search, hint)
    end
    return _pchip_interp_precompute!(output, x, y, x_query, extrap, deriv, search, hint)
end

"""
    pchip_interp(x, y, x_query; coeffs=PreCompute(), ...)

PCHIP interpolation at multiple query points. Returns `Vector{Tv}`.
"""
function pchip_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    output = Vector{Tv}(undef, length(x_query))
    pchip_interp!(output, x, y, x_query; coeffs = coeffs, extrap = extrap, deriv = deriv, search = search, hint = hint)
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  GENERIC WRAPPERS — Real type promotion                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar — promotes x, y to Float; passes xq directly for AD support
@inline function pchip_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        xq::Tq;
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    x_p, y_p = _promote_itp_inputs(x, y)
    return pchip_interp(x_p, y_p, xq; kwargs...)
end

# Vector — allocating
function pchip_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    x_p, y_p, xq_p = _promote_itp_inputs(x, y, x_query)
    Tv_float = eltype(y_p)
    output = Vector{Tv_float}(undef, length(x_query))
    pchip_interp!(output, x_p, y_p, xq_p; kwargs...)
    return output
end

# Vector — in-place
function pchip_interp!(
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

    return pchip_interp!(output, x_p, y_p, xq_p; kwargs...)
end
