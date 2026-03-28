# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 LINEAR ONE-SHOT SERIES INTERPOLATION                     ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides `linear_interp(x, Series(y1,y2,...), xq)` without constructing a
# SeriesInterpolant. Zero-allocation for scalar queries.
#
# Include order: ... → linear_anchor.jl → linear_oneshot_series.jl → ...
# Shared anchor eval: _linear_eval_at_anchor(y, aq, op, extrap) in linear_anchor.jl

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Scalar Series → Vector return (consistent with SeriesInterpolant) ───────

"""
    linear_interp(x, Series(y1, y2, ...), xq; ...) → Vector
    linear_interp(x, Series(Y::Matrix), xq; ...) → Vector

One-shot linear interpolation of multiple y-series at a single query point.
Returns a `Vector`, consistent with `SeriesInterpolant` output format.

# Strategy
Search once → anchor once → evaluate kernel per y-vector.

# Example
```julia
x = 0.0:0.01:1.0
y_sin, y_cos = sin.(x), cos.(x)
vals = linear_interp(x, Series(y_sin, y_cos), 0.5)  # → [sin(0.5), cos(0.5)]
```
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
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:linear), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    K = n_series(s)
    Tv = promote_type(_series_eltype(s), typeof(aq.alpha))
    output = Vector{Tv}(undef, K)
    @inbounds for k in 1:K
        output[k] = _linear_eval_at_anchor(vecs[k], aq, deriv, extrap)
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
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:linear), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    @inbounds for k in eachindex(output)
        output[k] = _linear_eval_at_anchor(vecs[k], aq, deriv, extrap)
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
    _validate_series_lengths(s, length(x))
    K = n_series(s)
    length(outputs) == K || _throw_series_dim_mismatch(length(outputs), K)
    vecs = _series_vectors(s)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap isa WrapExtrap
    @inbounds for j in eachindex(xqs)
        aq = _anchor_query(x, xqs[j], Val(:linear), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _linear_eval_at_anchor(vecs[k], aq, deriv, extrap)
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

# Scalar: Real grid → promote x, forward kwargs
@inline function linear_interp(
        x::AbstractVector{Tg}, s::Series, xq::Tq; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    x_typed = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    return linear_interp(x_typed, s, xq; kwargs...)
end

# In-place scalar: Real grid
@inline function linear_interp!(
        output::AbstractVector, x::AbstractVector{Tg}, s::Series, xq::Tq; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    x_typed = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    return linear_interp!(output, x_typed, s, xq; kwargs...)
end

# Vector in-place: Real grid
function linear_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg}, s::Series, xqs::AbstractVector{Tq}; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return linear_interp!(outputs, _to_float(x, Tg_float), s, xqs; kwargs...)
end

# Vector allocating: Real grid
function linear_interp(
        x::AbstractVector{Tg}, s::Series, xqs::AbstractVector{Tq}; kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return linear_interp(_to_float(x, Tg_float), s, xqs; kwargs...)
end
