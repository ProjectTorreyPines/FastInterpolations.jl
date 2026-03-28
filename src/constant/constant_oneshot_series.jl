# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                CONSTANT ONE-SHOT SERIES INTERPOLATION                    ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: ... → constant_anchor.jl → constant_oneshot_series.jl → ...
# Shared anchor eval: _constant_eval_at_anchor(y, x_last, aq, op, side, extrap) in constant_anchor.jl

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Internal: Tuple NTuple return (zero heap alloc) ─────────────────────────
@inline function _constant_oneshot_series_ntuple(
        x::AbstractVector{Tg},
        s::Series{<:Tuple},
        xq::Tq;
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    K = n_series(s)
    return ntuple(k -> _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap), Val(K))
end

# ─── Scalar Series → Vector return (consistent with SeriesInterpolant) ───────

@inline function constant_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    K = n_series(s)
    Tv_out = _value_type(_series_eltype(s), Tg)
    output = Vector{Tv_out}(undef, K)
    @inbounds for k in 1:K
        output[k] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
    end
    return output
end

# ─── In-place scalar ─────────────────────────────────────────────────────────

@inline function constant_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    @inbounds for k in eachindex(output)
        output[k] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function constant_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    K = n_series(s)
    length(outputs) == K || _throw_series_dim_mismatch(length(outputs), K)
    vecs = _series_vectors(s)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap isa WrapExtrap
    x_last = Tg(last(x))
    @inbounds for j in eachindex(xqs)
        aq = _anchor_query(x, xqs[j], Val(:constant), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
        end
    end
    return outputs
end

function constant_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tq <: Real}
    K = n_series(s)
    Tv_out = _value_type(_series_eltype(s), Tg)
    outputs = [Vector{Tv_out}(undef, length(xqs)) for _ in 1:K]
    constant_interp!(outputs, x, s, xqs; side, extrap, deriv, search)
    return outputs
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     REAL TYPE PROMOTION WRAPPERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@inline function constant_interp(
        x::AbstractVector{Tg}, s::Series, xq::Tq; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    x_typed = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    return constant_interp(x_typed, s, xq; kwargs...)
end

@inline function constant_interp!(
        output::AbstractVector, x::AbstractVector{Tg}, s::Series, xq::Tq; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    x_typed = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    return constant_interp!(output, x_typed, s, xq; kwargs...)
end

function constant_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg}, s::Series, xqs::AbstractVector{Tq}; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return constant_interp!(outputs, _to_float(x, Tg_float), s, _to_float(xqs, Tg_float); kwargs...)
end

function constant_interp(
        x::AbstractVector{Tg}, s::Series, xqs::AbstractVector{Tq}; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return constant_interp(_to_float(x, Tg_float), s, _to_float(xqs, Tg_float); kwargs...)
end
