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
# ║                 INTERNAL: PERIODIC CORE (zero-copy)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Zero-copy scalar-series periodic anchor + K-loop eval. One search, K evals.
# No pool, no grid/value extension — the anchor carries pair indices
# (`idxL`/`idxR`) so `y[aq.idxR]` reads `vecs[k][1]` automatically at the
# exclusive seam.
#
# Mirrors the non-series scalar periodic pattern in `linear_oneshot.jl`:
# resolve a `WrapExtrap` carrying the BC's period, wrap `xq` into the periodic
# domain, then call `search_interval` on the wrapped query. For exclusive,
# the searcher's seam branch fires when the wrapped `xq` lands in
# `[x[n], x[1]+period)` and returns the `(n, 1)` seam pair. For inclusive,
# the wrap uses `x[n] - x[1]` (= period) and search returns a regular pair
# guarded by the user's `y[1] ≈ y[end]` invariant.
#
# The only `bc`-distinguished work is the per-series endpoint validation for
# inclusive — a compile-time `bc isa PeriodicBC{:inclusive}` branch that LLVM
# drops in the exclusive specialization.
@inline function _linear_oneshot_series_periodic!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq,
        bc::PeriodicBC,
        op::AbstractEvalOp,
        searcher
    ) where {Tg}
    vecs = _series_vectors(s)
    K = length(output)

    if bc isa PeriodicBC{:inclusive}
        @inbounds for k in 1:K
            _check_periodic_endpoints(bc, vecs[k])
        end
    end

    # Materialize WrapExtrap with BC-correct period (inclusive: x[n]-x[1];
    # exclusive: bc.period). `first(vecs)` only contributes element-type info.
    extrap_p = _resolve_extrap(NoExtrap(), bc, x, first(vecs))
    xq_wrapped = _wrap_to_domain(xq, extrap_p)
    idxL, idxR, xL, xR = search_interval(searcher, x, xq_wrapped)
    h     = xR - xL
    inv_h = inv(h)
    alpha = (xq_wrapped - xL) * inv_h
    aq = _LinearAnchoredQuery(idxL, idxR, xq_wrapped, IN_DOMAIN, xL, h, inv_h, alpha)

    @inbounds for k in 1:K
        output[k] = _linear_eval_at_anchor(vecs[k], aq, op, extrap_p)
    end
    return output
end

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
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    Tg_p = _promote_grid_float(Tg, _series_eltype(s))
    x = _to_float(x, Tg_p)
    K = n_series(s)
    Tg_actual = eltype(x)
    Tv = _series_output_type(_output_eltype(_series_eltype(s), Tg_actual), Tq)
    output = Vector{Tv}(undef, K)
    if _is_periodic_bc(bc)
        # Thread `bc` into the Searcher type param so `search_interval` performs
        # the seam-wrap for `PeriodicBC{:exclusive}` inside the helper.
        searcher = _resolve_search(x, xq, search, hint, bc)
        return _linear_oneshot_series_periodic!(output, x, s, xq, bc, deriv, searcher)
    end
    _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:linear), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
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
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    Tg_p = _promote_grid_float(Tg, _series_eltype(s))
    x = _to_float(x, Tg_p)
    if _is_periodic_bc(bc)
        searcher = _resolve_search(x, xq, search, hint, bc)
        return _linear_oneshot_series_periodic!(output, x, s, xq, bc, deriv, searcher)
    end
    _check_domain(x, xq, extrap)
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
@with_pool pool function linear_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    Tg_p = _promote_grid_float(Tg, _series_eltype(s))
    x = _to_float(x, Tg_p)
    Tg_actual = eltype(x)
    K = n_series(s)
    _validate_series_outputs(outputs, K, length(xqs))
    vecs = _series_vectors(s)

    # Periodic vector-xqs: extend x once, per-series extend y (exclusive) or
    # reuse + validate (inclusive), pre-compute anchors on extended grid, then
    # K outer × Q inner eval loop.
    if _is_periodic_bc(bc)
        x_p, y_p_first, extrap_p = _periodic_extend_1d_pooled!(pool, x, first(vecs), bc, WrapExtrap())
        Tg_p_actual = eltype(x_p)
        Tq_promoted = promote_type(Tq, Tg_p_actual)
        searcher = _resolve_search(x_p, xqs, search, nothing)
        aq_vec = acquire!(pool, _LinearAnchoredQuery{Tg_p_actual, Tq_promoted}, length(xqs))
        _fill_anchors!(aq_vec, x_p, xqs, Val(:linear), true, searcher)
        @inbounds for j in eachindex(xqs)
            outputs[1][j] = _linear_eval_at_anchor(y_p_first, aq_vec[j], deriv, extrap_p)
        end
        if K > 1
            if bc isa PeriodicBC{:exclusive}
                n = length(x)
                Tv_buf = eltype(y_p_first)
                y_p = acquire!(pool, Tv_buf, length(x_p))
                @inbounds for k in 2:K
                    copyto!(y_p, 1, vecs[k], 1, n)
                    y_p[n + 1] = vecs[k][1]
                    for j in eachindex(xqs)
                        outputs[k][j] = _linear_eval_at_anchor(y_p, aq_vec[j], deriv, extrap_p)
                    end
                end
            else
                @inbounds for k in 2:K
                    _check_periodic_endpoints(bc, vecs[k])
                    for j in eachindex(xqs)
                        outputs[k][j] = _linear_eval_at_anchor(vecs[k], aq_vec[j], deriv, extrap_p)
                    end
                end
            end
        end
        return outputs
    end

    # Domain check: NoExtrap → throws if OOB, returns InBounds(); others → pass-through
    extrap_eff = _check_domain(x, xqs, extrap)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap_eff isa WrapExtrap
    # Pre-compute anchors via pool, then K outer × Q inner for cache locality
    Tq_promoted = promote_type(Tq, Tg_actual)
    aq_vec = acquire!(pool, _LinearAnchoredQuery{Tg_actual, Tq_promoted}, length(xqs))
    _fill_anchors!(aq_vec, x, xqs, Val(:linear), wrap, searcher)
    @inbounds for k in 1:K
        for j in eachindex(xqs)
            outputs[k][j] = _linear_eval_at_anchor(vecs[k], aq_vec[j], deriv, extrap_eff)
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
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    K = n_series(s)
    Tg_p = _promote_grid_float(Tg, _series_eltype(s))
    Tv_out = _series_output_type(_output_eltype(_series_eltype(s), Tg_p), Tq)
    outputs = [Vector{Tv_out}(undef, length(xqs)) for _ in 1:K]
    linear_interp!(outputs, x, s, xqs; bc, extrap, deriv, search)
    return outputs
end

# NOTE: the former Real type promotion wrappers (scalar, in-place scalar, vector
# in-place, vector allocating) have been removed. Each typed method above now
# handles promotion internally via _promote_grid_float + _to_float, same as the
# non-series linear API. This prevents infinite recursion on duck grids and avoids
# dispatch ambiguity between {Tg} and {Tg<:Real} signatures on Julia 1.10.
