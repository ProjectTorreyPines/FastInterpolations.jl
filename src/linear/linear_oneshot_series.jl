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

    # Surface-API axis resolution: for `:exclusive` Vector / Range, `x_eff`
    # becomes `_ExclusivePeriodicAxis` carrying the period and virtual endpoint.
    # `last(x_eff)` then gives the canonical wrap domain max (`first + period`),
    # and the wrapper's specialized `search_interval` returns post-fold idx_R=1
    # at the seam. For `:inclusive`, axis stays unchanged.
    x_eff = _resolve_axis(x, bc)
    extrap_p = _resolve_extrap(NoExtrap(), bc, x_eff)   # PeriodicBC → WrapExtrap()
    xq_wrapped = _wrap_to_domain(xq, x_eff)
    idxL, idxR, xL, xR = search_interval(searcher, x_eff, xq_wrapped)
    # Index-based 2-arg `_get_h(x, idx)` reads `_CachedRange.h` (exact cached
    # step) and the wrapper's seam width directly — avoids `xR - xL` cancellation
    # on large-offset Ranges and matches the persistent interpolant path.
    h = _get_h(x_eff, idxL)
    inv_h = _get_inv_h(x_eff, idxL)
    alpha = (xq_wrapped - xL) * inv_h
    aq = _LinearAnchoredQuery(_IdxPair(idxL, idxR), xq_wrapped, IN_DOMAIN, xL, h, inv_h, alpha)

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
        # Helper wraps `x` via `_resolve_axis(x, bc)` and searches against the
        # wrapped axis — axis dispatch handles seam, no `bc` thread needed.
        searcher = _resolve_search(x, xq, search, hint)
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
        searcher = _resolve_search(x, xq, search, hint)
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
# Q outer × K inner — small-NQ fast path. Anchor stack-resident across the
# K-inner loop. No pool acquired so tiny `xqs` doesn't pay pool-setup
# overhead. `_is_periodic_bc(bc)` is compile-time-resolved.
@inline function _linear_series_batch_qk!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector,
        vecs,
        xqs::AbstractVector,
        bc::AbstractBC,
        op::AbstractEvalOp,
        extrap::AbstractExtrap,
        search::AbstractSearchPolicy
    )
    K = length(vecs)
    NQ = length(xqs)

    if _is_periodic_bc(bc)
        x_eff = _resolve_axis(x, bc)
        extrap_p = _resolve_extrap(NoExtrap(), bc, x_eff)
        searcher = _resolve_search(x_eff, xqs, search, nothing)
        @inbounds for j in 1:NQ
            xq_wrapped = _wrap_to_domain(xqs[j], x_eff)
            idxL, idxR, xL, _ = search_interval(searcher, x_eff, xq_wrapped)
            h = _get_h(x_eff, idxL)
            inv_h = _get_inv_h(x_eff, idxL)
            alpha = (xq_wrapped - xL) * inv_h
            aq = _LinearAnchoredQuery(_IdxPair(idxL, idxR), xq_wrapped, IN_DOMAIN, xL, h, inv_h, alpha)
            for k in 1:K
                outputs[k][j] = _linear_eval_at_anchor(vecs[k], aq, op, extrap_p)
            end
        end
        return outputs
    end

    extrap_eff = _check_domain(x, xqs, extrap)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap_eff isa WrapExtrap
    @inbounds for j in 1:NQ
        aq = _anchor_query(x, xqs[j], Val(:linear), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _linear_eval_at_anchor(vecs[k], aq, op, extrap_eff)
        end
    end
    return outputs
end

# K outer × Q inner with pool-acquired anchor vector — large-NQ fast path.
# Inner loop streams a single `outputs[k]` Vector — LLVM auto-SIMDs and
# avoids the K-cache-line write jump per query of the Q×K shape.
@inline @with_pool pool function _linear_series_batch_kq!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        vecs,
        xqs::AbstractVector{Tq},
        bc::AbstractBC,
        op::AbstractEvalOp,
        extrap::AbstractExtrap,
        search::AbstractSearchPolicy
    ) where {Tg, Tq <: Real}
    K = length(vecs)
    NQ = length(xqs)
    Tg_actual = eltype(x)
    Tqp = promote_type(Tg_actual, Tq)

    if _is_periodic_bc(bc)
        x_eff = _resolve_axis(x, bc)
        extrap_p = _resolve_extrap(NoExtrap(), bc, x_eff)
        searcher = _resolve_search(x_eff, xqs, search, nothing)
        aq_vec = acquire!(pool, _LinearAnchoredQuery{Tg_actual, Tqp}, NQ)
        @inbounds for j in 1:NQ
            xq_wrapped = _wrap_to_domain(xqs[j], x_eff)
            idxL, idxR, xL, _ = search_interval(searcher, x_eff, xq_wrapped)
            h = _get_h(x_eff, idxL)
            inv_h = _get_inv_h(x_eff, idxL)
            alpha = (xq_wrapped - xL) * inv_h
            aq_vec[j] = _LinearAnchoredQuery(_IdxPair(idxL, idxR), xq_wrapped, IN_DOMAIN, xL, h, inv_h, alpha)
        end
        @inbounds for k in 1:K
            for j in 1:NQ
                outputs[k][j] = _linear_eval_at_anchor(vecs[k], aq_vec[j], op, extrap_p)
            end
        end
        return outputs
    end

    extrap_eff = _check_domain(x, xqs, extrap)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap_eff isa WrapExtrap
    aq_vec = acquire!(pool, _LinearAnchoredQuery{Tg_actual, Tqp}, NQ)
    _fill_anchors!(aq_vec, x, xqs, Val(:linear), wrap, searcher)
    @inbounds for k in 1:K
        for j in 1:NQ
            outputs[k][j] = _linear_eval_at_anchor(vecs[k], aq_vec[j], op, extrap_eff)
        end
    end
    return outputs
end

# Adaptive entry. `length(xqs)` and `K` feed `_series_use_kq_loop` to select
# the loop order — see `src/core/series_utils.jl`.
@inline function linear_interp!(
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
    K = n_series(s)
    _validate_series_outputs(outputs, K, length(xqs))
    vecs = _series_vectors(s)
    # Per-series `:inclusive` endpoint guarantee. `:exclusive` needs no
    # per-series check; the anchor's seam pair handles wrap.
    if bc isa PeriodicBC{:inclusive}
        @inbounds for k in 1:K
            _check_periodic_endpoints(bc, vecs[k])
        end
    end

    if _series_use_kq_loop(length(xqs), K)
        return _linear_series_batch_kq!(outputs, x, vecs, xqs, bc, deriv, extrap, search)
    else
        return _linear_series_batch_qk!(outputs, x, vecs, xqs, bc, deriv, extrap, search)
    end
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
