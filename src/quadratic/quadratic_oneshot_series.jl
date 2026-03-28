# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║               QUADRATIC ONE-SHOT SERIES INTERPOLATION                    ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: ... → quadratic_anchor.jl → quadratic_oneshot_series.jl → ...
# Shared anchor eval: _quadratic_eval_at_anchor(y, a, d, aq, op, extrap) in quadratic_anchor.jl

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Internal: Tuple NTuple return (zero heap alloc) ─────────────────────────
@inline @with_pool pool function _quadratic_oneshot_series_ntuple(
        x::AbstractVector{Tg},
        s::Series{<:Tuple},
        xq::Tq;
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    nx = length(x)
    spacing = _create_spacing_pooled(pool, x)
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:quadratic), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    K = n_series(s)
    Tv_out = _value_type(eltype(first(vecs)), Tg)
    d = acquire!(pool, Tv_out, nx)
    a = acquire!(pool, Tv_out, nx - 1)
    y_buf = acquire!(pool, Tv_out, nx)
    return ntuple(Val(K)) do k
        copyto!(y_buf, 1, vecs[k], 1, nx)
        bc_promoted = _normalize_bc(bc, first(y_buf))
        _compute_quadratic_coeffs!(d, a, spacing, x, y_buf, bc_promoted)
        _quadratic_eval_at_anchor(y_buf, a, d, aq, deriv, extrap)
    end
end

# ─── Scalar Series → Vector return (consistent with SeriesInterpolant) ───────

@inline @with_pool pool function quadratic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    nx = length(x)
    K = n_series(s)
    vecs = _series_vectors(s)
    spacing = _create_spacing_pooled(pool, x)
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:quadratic), extrap isa WrapExtrap, searcher)
    Tv_out = _value_type(_series_eltype(s), Tg)
    output = Vector{promote_type(Tv_out, typeof(aq.dL))}(undef, K)
    d = acquire!(pool, Tv_out, nx)
    a = acquire!(pool, Tv_out, nx - 1)
    y_buf = acquire!(pool, Tv_out, nx)
    @inbounds for k in 1:K
        copyto!(y_buf, 1, vecs[k], 1, nx)
        bc_promoted = _normalize_bc(bc, first(y_buf))
        _compute_quadratic_coeffs!(d, a, spacing, x, y_buf, bc_promoted)
        output[k] = _quadratic_eval_at_anchor(y_buf, a, d, aq, deriv, extrap)
    end
    return output
end

# ─── In-place scalar ─────────────────────────────────────────────────────────

@inline @with_pool pool function quadratic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    nx = length(x)
    vecs = _series_vectors(s)
    spacing = _create_spacing_pooled(pool, x)
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:quadratic), extrap isa WrapExtrap, searcher)
    Tv_out = _value_type(_series_eltype(s), Tg)
    d = acquire!(pool, Tv_out, nx)
    a = acquire!(pool, Tv_out, nx - 1)
    y_buf = acquire!(pool, Tv_out, nx)
    @inbounds for k in eachindex(output)
        copyto!(y_buf, 1, vecs[k], 1, nx)
        bc_promoted = _normalize_bc(bc, first(y_buf))
        _compute_quadratic_coeffs!(d, a, spacing, x, y_buf, bc_promoted)
        output[k] = _quadratic_eval_at_anchor(y_buf, a, d, aq, deriv, extrap)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@with_pool pool function quadratic_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    K = n_series(s)
    length(outputs) == K || _throw_series_dim_mismatch(length(outputs), K)
    nx = length(x)
    vecs = _series_vectors(s)
    spacing = _create_spacing_pooled(pool, x)
    Tv_out = _value_type(_series_eltype(s), Tg)

    # Pre-compute coefficients for all series
    ds = [acquire!(pool, Tv_out, nx) for _ in 1:K]
    as = [acquire!(pool, Tv_out, nx - 1) for _ in 1:K]
    y_buf = acquire!(pool, Tv_out, nx)
    for k in 1:K
        copyto!(y_buf, 1, vecs[k], 1, nx)
        bc_promoted = _normalize_bc(bc, first(y_buf))
        _compute_quadratic_coeffs!(ds[k], as[k], spacing, x, y_buf, bc_promoted)
    end

    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap isa WrapExtrap
    @inbounds for j in eachindex(xqs)
        aq = _anchor_query(x, xqs[j], Val(:quadratic), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _quadratic_eval_at_anchor(vecs[k], as[k], ds[k], aq, deriv, extrap)
        end
    end
    return outputs
end

function quadratic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tq <: Real}
    K = n_series(s)
    Tv_out = promote_type(_value_type(_series_eltype(s), Tg), Tq)
    outputs = [Vector{Tv_out}(undef, length(xqs)) for _ in 1:K]
    quadratic_interp!(outputs, x, s, xqs; bc, extrap, deriv, search)
    return outputs
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     REAL TYPE PROMOTION WRAPPERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@inline function quadratic_interp(
        x::AbstractVector{Tg}, s::Series, xq::Tq; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    x_typed = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    return quadratic_interp(x_typed, s, xq; kwargs...)
end

@inline function quadratic_interp!(
        output::AbstractVector, x::AbstractVector{Tg}, s::Series, xq::Tq; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    x_typed = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    return quadratic_interp!(output, x_typed, s, xq; kwargs...)
end

function quadratic_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg}, s::Series, xqs::AbstractVector{Tq}; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return quadratic_interp!(outputs, _to_float(x, Tg_float), s, _to_float(xqs, Tg_float); kwargs...)
end

function quadratic_interp(
        x::AbstractVector{Tg}, s::Series, xqs::AbstractVector{Tq}; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return quadratic_interp(_to_float(x, Tg_float), s, _to_float(xqs, Tg_float); kwargs...)
end
