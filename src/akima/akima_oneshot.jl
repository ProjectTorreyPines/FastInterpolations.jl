# ========================================
# Akima 1D Oneshot API
# ========================================
# Standalone akima_interp / akima_interp! functions.
# Computes Akima slopes via @with_pool, then delegates to
# _cubic_hermite_eval_at_point / _cubic_hermite_vector_loop!.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH — AbstractFloat grid                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar oneshot
# ========================================

"""
    akima_interp(x, y, xq; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Akima interpolation at a single query point.
Outlier-robust, C\$^1\$ continuous.
"""
@inline @with_pool pool function akima_interp(
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
    _akima_slopes!(dy, x, y)
    searcher = _resolve_search(x, xq, search, hint)
    return _cubic_hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — in-place
# ========================================

"""
    akima_interp!(output, x, y, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

In-place Akima interpolation.
"""
@inline @with_pool pool function akima_interp!(
        output::AbstractVector{Tv},
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tv}
    @assert length(y) == length(x) "y length must match x"
    @assert length(output) == length(x_query) "output length must match x_query"

    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    dy = similar!(pool, y)
    _akima_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, nothing)
    return _cubic_hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# Range disambiguation for in-place
@inline @with_pool pool function akima_interp!(
        output::AbstractVector{Tv},
        x::AbstractRange{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tv}
    @assert length(y) == length(x) "y length must match x"
    @assert length(output) == length(x_query) "output length must match x_query"

    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    dy = similar!(pool, y)
    _akima_slopes!(dy, x, y)
    searcher = _resolve_search(x, x_query, search, nothing)
    return _cubic_hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — allocating
# ========================================

"""
    akima_interp(x, y, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Akima interpolation at multiple query points. Returns `Vector{Tv}`.
"""
function akima_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tv}
    output = Vector{Tv}(undef, length(x_query))
    akima_interp!(output, x, y, x_query; extrap, deriv, search)
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  GENERIC WRAPPERS — Real type promotion                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar
@inline function akima_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        xq::Tq;
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    x_p, y_p = _promote_itp_inputs(x, y)
    return akima_interp(x_p, y_p, xq; kwargs...)
end

# Vector — allocating
function akima_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    x_p, y_p, xq_p = _promote_itp_inputs(x, y, x_query)
    Tv_float = eltype(y_p)
    output = Vector{Tv_float}(undef, length(x_query))
    akima_interp!(output, x_p, y_p, xq_p; kwargs...)
    return output
end

# Vector — in-place
function akima_interp!(
        output::AbstractVector,
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    @assert length(y) == length(x) "y length must match x"
    @assert length(output) == length(x_query) "output length must match x_query"

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

    return akima_interp!(output, x_p, y_p, xq_p; kwargs...)
end
