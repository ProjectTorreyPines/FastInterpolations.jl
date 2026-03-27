# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                CONSTANT ONE-SHOT SERIES INTERPOLATION                    ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: ... → constant_anchor.jl → constant_oneshot_series.jl → ...

# ─── Decoupled anchor dispatch (reads raw y, not interpolant) ────────────────

# Inside domain or extension mode → kernel
@inline function _constant_series_eval_at_anchor(
        y::AbstractVector{Tv},
        x_last::Tg,
        aq::_ConstantAnchoredQuery{Tg},
        op::AbstractEvalOp,
        side_param::AbstractSide,
        ::AbstractExtrap
    ) where {Tg <: AbstractFloat, Tv}
    if aq.xq == x_last
        return op isa EvalValue ? (@inbounds y[end]) : 0 * first(y)
    end
    @inbounds return _constant_kernel(op, y[aq.idx], y[aq.idx + 1], aq.h, aq.dL, side_param)
end

# NoExtrap → throw if OOB
@inline function _constant_series_eval_at_anchor(
        y::AbstractVector{Tv},
        x_last::Tg,
        aq::_ConstantAnchoredQuery{Tg},
        op::AbstractEvalOp,
        side_param::AbstractSide,
        ::NoExtrap
    ) where {Tg <: AbstractFloat, Tv}
    if aq.side != 0x00
        throw(DomainError(aq.xq, "query point outside domain"))
    end
    if aq.xq == x_last
        return op isa EvalValue ? (@inbounds y[end]) : 0 * first(y)
    end
    @inbounds return _constant_kernel(op, y[aq.idx], y[aq.idx + 1], aq.h, aq.dL, side_param)
end

# ClampExtrap / FillExtrap → boundary value if OOB
@inline function _constant_series_eval_at_anchor(
        y::AbstractVector{Tv},
        x_last::Tg,
        aq::_ConstantAnchoredQuery{Tg},
        op::AbstractEvalOp,
        side_param::AbstractSide,
        extrap::_ClampOrFill
    ) where {Tg <: AbstractFloat, Tv}
    if aq.side == 0x01
        return _eval_extrapolation(op, first(y), extrap, aq.xq)
    elseif aq.side == 0x02
        return _eval_extrapolation(op, last(y), extrap, aq.xq)
    end
    if aq.xq == x_last
        return op isa EvalValue ? (@inbounds y[end]) : 0 * first(y)
    end
    @inbounds return _constant_kernel(op, y[aq.idx], y[aq.idx + 1], aq.h, aq.dL, side_param)
end

# ExtendExtrap → ClampExtrap for constant (zero slope → extend = clamp)
@inline function _constant_series_eval_at_anchor(
        y::AbstractVector{Tv},
        x_last::Tg,
        aq::_ConstantAnchoredQuery{Tg},
        op::AbstractEvalOp,
        side_param::AbstractSide,
        ::ExtendExtrap
    ) where {Tg <: AbstractFloat, Tv}
    return _constant_series_eval_at_anchor(y, x_last, aq, op, side_param, ClampExtrap())
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Tuple Series → NTuple return ────────────────────────────────────────────

"""
    constant_interp(x, Series(y1, y2, ...), xq; side, extrap, deriv, search, hint) → NTuple

One-shot constant interpolation of multiple y-series at a single query point.
"""
@inline function constant_interp(
        x::AbstractVector{Tg},
        s::Series{<:Tuple},
        xq::Tq;
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    K = n_series(s)
    return ntuple(k -> _constant_series_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap), Val(K))
end

# ─── Dynamic Series → Vector return ──────────────────────────────────────────

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
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    K = n_series(s)
    Tv = _series_eltype(s)
    output = Vector{Tv}(undef, K)
    @inbounds for k in 1:K
        output[k] = _constant_series_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
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
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    @assert length(output) == n_series(s) "output length must match number of series"
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    @inbounds for k in eachindex(output)
        output[k] = _constant_series_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
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
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    K = n_series(s)
    @assert length(outputs) == K "outputs length must match number of series"
    vecs = _series_vectors(s)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap isa WrapExtrap
    x_last = Tg(last(x))
    @inbounds for j in eachindex(xqs)
        aq = _anchor_query(x, xqs[j], Val(:constant), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _constant_series_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
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
    Tv = _series_eltype(s)
    outputs = [Vector{Tv}(undef, length(xqs)) for _ in 1:K]
    constant_interp!(outputs, x, s, xqs; side, extrap, deriv, search)
    return outputs
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     REAL TYPE PROMOTION WRAPPERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@inline function constant_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    return constant_interp(x_typed, s, xq; side, extrap, deriv, search, hint)
end

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
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    return constant_interp!(output, x_typed, s, xq; side, extrap, deriv, search, hint)
end

function constant_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    xqs_typed = _to_float(xqs, Tg_float)
    return constant_interp!(outputs, x_typed, s, xqs_typed; side, extrap, deriv, search)
end

function constant_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    xqs_typed = _to_float(xqs, Tg_float)
    return constant_interp(x_typed, s, xqs_typed; side, extrap, deriv, search)
end
