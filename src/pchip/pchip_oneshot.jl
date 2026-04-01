# ========================================
# PCHIP 1D Oneshot API
# ========================================
# Standalone pchip_interp / pchip_interp! functions.
# Computes Fritsch-Carlson slopes via @with_pool, then delegates
# to _cubic_hermite_eval_at_point / _cubic_hermite_vector_loop! from Phase 1.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH — AbstractFloat grid                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar oneshot
# ========================================

"""
    pchip_interp(x, y, xq; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

PCHIP (monotone-preserving) interpolation at a single query point.

Computes Fritsch-Carlson slopes internally, then evaluates via cubic Hermite kernel.
C\$^1\$ continuous, monotonicity guaranteed for monotone input data.
"""
@inline @with_pool pool function pchip_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))

    x = _to_float(x, Tg)
    dy = similar!(pool, y)
    _pchip_slopes!(dy, x, y)
    searcher = _resolve_search(x, xq, search, hint)
    return _cubic_hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — in-place
# ========================================

"""
    pchip_interp!(output, x, y, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

In-place PCHIP interpolation with monotone-preserving slopes.
"""
@inline @with_pool pool function pchip_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    dy = similar!(pool, y)
    _pchip_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, hint)
    return _cubic_hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# Range disambiguation for in-place
@inline @with_pool pool function pchip_interp!(
        output::AbstractVector,
        x::AbstractRange{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    dy = similar!(pool, y)
    _pchip_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, hint)
    return _cubic_hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — allocating
# ========================================

"""
    pchip_interp(x, y, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

PCHIP interpolation at multiple query points. Returns `Vector{Tv}`.
"""
function pchip_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    output = Vector{Tv}(undef, length(x_query))
    pchip_interp!(output, x, y, x_query; extrap, deriv, search, hint)
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
