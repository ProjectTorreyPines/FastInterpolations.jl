# ========================================
# Cubic Hermite 1D Oneshot API
# ========================================
# Extends cubic_interp / cubic_interp! with Hermite(y, dy) dispatch.
# No cache, no solve — user-supplied slopes go directly to _hermite_kernel_1d.

# ========================================
# Keyword Rejection Helper
# ========================================

@noinline _throw_hermite_bc_error() =
    throw(ArgumentError("bc is not applicable for Hermite data — slopes are user-provided"))
@noinline _throw_hermite_autocache_error() =
    throw(ArgumentError("autocache is not applicable for Hermite data — no global solve needed"))

@inline function _reject_hermite_kwargs(bc, autocache)
    bc !== nothing && _throw_hermite_bc_error()
    autocache !== nothing && _throw_hermite_autocache_error()
    return nothing
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH — AbstractFloat grid                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar oneshot
# ========================================

"""
    cubic_interp(x, Hermite(y, dy), xq; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Cubic Hermite interpolation at a single query point using user-supplied slopes.

Returns interpolated value (or derivative, if `deriv` is set).
C\$^1\$ continuous — slopes are used directly, no global spline solve.
"""
@inline function cubic_interp(
        x::AbstractVector{Tg},
        h::Hermite{<:AbstractVector{Tv}, <:AbstractVector{Tv}},
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
        bc::Union{Nothing, AbstractBC} = nothing,
        autocache::Union{Nothing, Bool} = nothing
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    _reject_hermite_kwargs(bc, autocache)
    @boundscheck length(h.y) == length(x) || _throw_length_mismatch(length(x), length(h.y))

    x = _to_float(x, Tg)
    searcher = _resolve_search(x, xq, search, hint)
    return _cubic_hermite_eval_at_point(x, h.y, h.dy, xq, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — in-place
# ========================================

"""
    cubic_interp!(output, x, Hermite(y, dy), x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

In-place cubic Hermite interpolation using user-supplied slopes.
"""
function cubic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        h::Hermite{<:AbstractVector{Tv}, <:AbstractVector{Tv}},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
        bc::Union{Nothing, AbstractBC} = nothing,
        autocache::Union{Nothing, Bool} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    _reject_hermite_kwargs(bc, autocache)
    @boundscheck length(h.y) == length(x) || _throw_length_mismatch(length(x), length(h.y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    searcher = _resolve_search(x, x_query, search, hint)
    return _cubic_hermite_vector_loop!(output, x, h.y, h.dy, x_query, extrap, deriv, searcher)
end

# Range disambiguation for in-place
function cubic_interp!(
        output::AbstractVector,
        x::AbstractRange{Tg},
        h::Hermite{<:AbstractVector{Tv}, <:AbstractVector{Tv}},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
        bc::Union{Nothing, AbstractBC} = nothing,
        autocache::Union{Nothing, Bool} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    _reject_hermite_kwargs(bc, autocache)
    @boundscheck length(h.y) == length(x) || _throw_length_mismatch(length(x), length(h.y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x = _to_float(x, Tg)
    x_query = _to_float(x_query, Tg)
    searcher = _resolve_search(x, x_query, search, hint)
    return _cubic_hermite_vector_loop!(output, x, h.y, h.dy, x_query, extrap, deriv, searcher)
end

# ========================================
# Vector oneshot — allocating
# ========================================

"""
    cubic_interp(x, Hermite(y, dy), x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Cubic Hermite interpolation at multiple query points using user-supplied slopes.
Returns `Vector{Tv}` of interpolated values.
"""
function cubic_interp(
        x::AbstractVector{Tg},
        h::Hermite{<:AbstractVector{Tv}, <:AbstractVector{Tv}},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
        bc::Union{Nothing, AbstractBC} = nothing,
        autocache::Union{Nothing, Bool} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    _reject_hermite_kwargs(bc, autocache)
    output = Vector{promote_type(Tg, Tv)}(undef, length(x_query))
    cubic_interp!(output, x, h, x_query; extrap, deriv, search, hint)
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  GENERIC WRAPPERS — Real type promotion                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar — promotes x, y, dy to Float; passes xq directly for AD support
@inline function cubic_interp(
        x::AbstractVector{TX},
        h::Hermite,
        xq::Tq;
        kwargs...
    ) where {TX <: Real, Tq <: Real}
    x_p, h_p = _promote_itp_inputs(x, h)
    return cubic_interp(x_p, h_p, xq; kwargs...)
end

# Vector — allocating
function cubic_interp(
        x::AbstractVector{TX},
        h::Hermite,
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, Tq <: Real}
    x_p, h_p, xq_p = _promote_itp_inputs(x, h, x_query)
    Tv_float = eltype(h_p.y)
    output = Vector{Tv_float}(undef, length(x_query))
    cubic_interp!(output, x_p, h_p, xq_p; kwargs...)
    return output
end

# Vector — in-place
function cubic_interp!(
        output::AbstractVector,
        x::AbstractVector{TX},
        h::Hermite,
        x_query::AbstractVector{Tq};
        kwargs...
    ) where {TX <: Real, Tq <: Real}
    @boundscheck length(h.y) == length(x) || _throw_length_mismatch(length(x), length(h.y))
    @boundscheck length(output) == length(x_query) || _throw_length_mismatch(length(x_query), length(output), "x_query", "output")

    x_p, h_p, xq_p = _promote_itp_inputs(x, h, x_query)
    Tv_float = eltype(h_p.y)

    Tout = eltype(output)
    if promote_type(Tout, Tv_float) !== Tout
        throw(
            ArgumentError(
                "output eltype $Tout cannot hold interpolation result type $Tv_float. " *
                    "Use Vector{$Tv_float} or a wider type."
            )
        )
    end

    return cubic_interp!(output, x_p, h_p, xq_p; kwargs...)
end
