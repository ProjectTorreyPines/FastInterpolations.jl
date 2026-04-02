# ========================================
# Hermite 1D Oneshot API
# ========================================
# hermite_interp / hermite_interp! — user-supplied slopes go directly to kernel.
# No cache, no solve, no wrapper.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH — AbstractFloat grid                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar oneshot
# ========================================

"""
    hermite_interp(x, y, dy, xq; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

Cubic Hermite interpolation at a single query point using user-supplied slopes.

Returns interpolated value (or derivative, if `deriv` is set).
C\$^1\$ continuous — slopes are used directly, no global spline solve.
"""
@inline function hermite_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(dy) == length(x) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
    @boundscheck length(x) >= 2 || throw(ArgumentError("Hermite interpolation requires at least 2 points, got $(length(x))"))

    x = _to_float(x, Tg)
    searcher = _resolve_search(x, xq, search, hint)
    return _hermite_eval_at_point(x, y, dy, xq, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — in-place
# ========================================

"""
    hermite_interp!(output, x, y, dy, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

In-place cubic Hermite interpolation using user-supplied slopes.
"""
function hermite_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg <: AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(dy) == length(x) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# Range disambiguation for in-place
function hermite_interp!(
        output::AbstractVector,
        x::AbstractRange{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg <: AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(dy) == length(x) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    searcher = _resolve_search(x, x_query, search, hint)
    return _hermite_vector_loop!(output, x, y, dy, x_query, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — allocating
# ========================================

"""
    hermite_interp(x, y, dy, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch(), hint=nothing)

Cubic Hermite interpolation at multiple query points using user-supplied slopes.
Returns `Vector{Tv}` of interpolated values.
"""
function hermite_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg <: AbstractFloat, Tv}
    output = Vector{_value_type(Tv, Tg)}(undef, length(x_query))
    hermite_interp!(output, x, y, dy, x_query; extrap, deriv, search, hint)
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  GENERIC WRAPPERS — Real type promotion                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar — promotes x, y, dy to Float; passes xq directly for AD support
@inline function hermite_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        dy::AbstractVector,
        xq::Tq;
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    x_p, y_p = _promote_itp_inputs(x, y)
    Tg = eltype(x_p)
    dy_p = eltype(dy) <: _PromotableValue ? _promote_value_type(dy, eltype(x_p))[2] : dy
    return hermite_interp(x_p, y_p, dy_p, xq; kwargs...)
end

# Vector — allocating
function hermite_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        dy::AbstractVector,
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    x_p, y_p = _promote_itp_inputs(x, y)
    Tg = eltype(x_p)
    dy_p = eltype(dy) <: _PromotableValue ? _promote_value_type(dy, eltype(x_p))[2] : dy
    xq_p = _to_float(x_query, Tg)
    output = Vector{eltype(y_p)}(undef, length(x_query))
    hermite_interp!(output, x_p, y_p, dy_p, xq_p; kwargs...)
    return output
end

# Vector — in-place
function hermite_interp!(
        output::AbstractVector,
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        dy::AbstractVector,
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, TY, Tq <: Real}
    @boundscheck length(y) == length(x) || _throw_length_mismatch(length(x), length(y))
    @boundscheck length(dy) == length(x) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x_p, y_p = _promote_itp_inputs(x, y)
    Tg = eltype(x_p)
    dy_p = eltype(dy) <: _PromotableValue ? _promote_value_type(dy, eltype(x_p))[2] : dy
    xq_p = _to_float(x_query, Tg)

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

    return hermite_interp!(output, x_p, y_p, dy_p, xq_p; kwargs...)
end
