# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 LINEAR ONE-SHOT SERIES INTERPOLATION                     ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides `linear_interp(x, Series(y1,y2,...), xq)` without constructing a
# SeriesInterpolant. Zero-allocation for scalar queries.
#
# Include order: ... → linear_anchor.jl → linear_oneshot_series.jl → ...

# ─── Decoupled anchor dispatch (reads raw y, not interpolant) ────────────────

# Inside domain or ExtendExtrap → just kernel
@inline function _linear_series_eval_at_anchor(
        y::AbstractVector{Tv},
        aq::_LinearAnchoredQuery{Tg, Tq},
        op::AbstractEvalOp,
        ::AbstractExtrap
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    @inbounds return _linear_kernel(op, y[aq.idx], y[aq.idx + 1], aq)
end

# NoExtrap → throw if OOB
@inline function _linear_series_eval_at_anchor(
        y::AbstractVector{Tv},
        aq::_LinearAnchoredQuery{Tg, Tq},
        op::AbstractEvalOp,
        ::NoExtrap
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    if aq.side != 0x00
        throw(DomainError(aq.xq, "query point outside domain"))
    end
    @inbounds return _linear_kernel(op, y[aq.idx], y[aq.idx + 1], aq)
end

# ClampExtrap / FillExtrap → boundary value if OOB
@inline function _linear_series_eval_at_anchor(
        y::AbstractVector{Tv},
        aq::_LinearAnchoredQuery{Tg, Tq},
        op::AbstractEvalOp,
        extrap::_ClampOrFill
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    if aq.side != 0x00
        y_bnd = aq.side == 0x01 ? first(y) : last(y)
        return _eval_extrapolation(op, y_bnd, extrap, aq.xq)
    end
    @inbounds return _linear_kernel(op, y[aq.idx], y[aq.idx + 1], aq)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Tuple Series → NTuple return (compile-time K) ───────────────────────────

"""
    linear_interp(x, Series(y1, y2, ...), xq; extrap, deriv, search, hint) → NTuple

One-shot linear interpolation of multiple y-series at a single query point.
Returns an `NTuple{K}` when `Series` is constructed from varargs (compile-time K).

# Strategy
Search once → anchor once → evaluate kernel per y-vector.
Zero allocation for scalar queries.

# Example
```julia
x = 0.0:0.01:1.0
y_sin, y_cos = sin.(x), cos.(x)
vals = linear_interp(x, Series(y_sin, y_cos), 0.5)  # → (sin(0.5), cos(0.5))
```
"""
@inline function linear_interp(
        x::AbstractVector{Tg},
        s::Series{<:Tuple},
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:linear), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    K = n_series(s)
    return ntuple(k -> _linear_series_eval_at_anchor(vecs[k], aq, deriv, extrap), Val(K))
end

# ─── Dynamic Series → Vector return ──────────────────────────────────────────

"""
    linear_interp(x, Series(Y::Matrix), xq; ...) → Vector
    linear_interp(x, Series([y1, y2, ...]), xq; ...) → Vector

One-shot linear interpolation of multiple y-series at a single query point.
Returns a `Vector` when `Series` wraps a matrix or vector-of-vectors.
"""
@inline function linear_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:linear), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    K = n_series(s)
    output = Vector{promote_type(eltype(first(vecs)), typeof(aq.alpha))}(undef, K)
    @inbounds for k in 1:K
        output[k] = _linear_series_eval_at_anchor(vecs[k], aq, deriv, extrap)
    end
    return output
end

# ─── In-place scalar ─────────────────────────────────────────────────────────

"""
    linear_interp!(output, x, Series(...), xq; ...) → output

In-place one-shot linear interpolation of multiple y-series at a single query point.
"""
@inline function linear_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    @assert length(output) == n_series(s) "output length must match number of series"
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:linear), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    @inbounds for k in eachindex(output)
        output[k] = _linear_series_eval_at_anchor(vecs[k], aq, deriv, extrap)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    linear_interp!(outputs, x, Series(...), xqs; ...) → outputs

In-place one-shot linear interpolation at multiple query points.
`outputs` is a `Vector{<:AbstractVector}` of length `n_series`, each of length `length(xqs)`.
"""
function linear_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
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
    @inbounds for j in eachindex(xqs)
        aq = _anchor_query(x, xqs[j], Val(:linear), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _linear_series_eval_at_anchor(vecs[k], aq, deriv, extrap)
        end
    end
    return outputs
end

"""
    linear_interp(x, Series(...), xqs::AbstractVector; ...) → Vector{Vector}

Allocating one-shot linear interpolation at multiple query points.
Returns `Vector{Vector{Tv}}` of length `n_series`, each of length `length(xqs)`.
"""
function linear_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tq <: Real}
    K = n_series(s)
    Tv = _series_eltype(s)
    Tv_out = promote_type(Tv, Tq)
    outputs = [Vector{Tv_out}(undef, length(xqs)) for _ in 1:K]
    linear_interp!(outputs, x, s, xqs; extrap, deriv, search)
    return outputs
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     REAL TYPE PROMOTION WRAPPERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Scalar: Real grid → promote x, pass xq directly (preserves Dual for AD)
@inline function linear_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    return linear_interp(x_typed, s, xq; extrap, deriv, search, hint)
end

# In-place scalar: Real grid
@inline function linear_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    return linear_interp!(output, x_typed, s, xq; extrap, deriv, search, hint)
end

# Vector: Real grid
function linear_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    xqs_typed = _to_float(xqs, Tg_float)
    return linear_interp!(outputs, x_typed, s, xqs_typed; extrap, deriv, search)
end

# Vector allocating: Real grid
function linear_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    xqs_typed = _to_float(xqs, Tg_float)
    return linear_interp(x_typed, s, xqs_typed; extrap, deriv, search)
end
