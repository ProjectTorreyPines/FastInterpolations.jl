# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  CUBIC ONE-SHOT SERIES INTERPOLATION                     ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides `cubic_interp(x, Series(y1,y2,...), xq; bc=...)` without constructing
# a CubicSeriesInterpolant. Uses pool allocation for z-buffer reuse.
#
# Include order: ... → cubic_anchor.jl → cubic_oneshot_series.jl → ...
# Shared kernel: _cubic_eval_kernel(y, z, aq, op) in cubic_anchor.jl
# Shared extrap: _cubic_eval_at_anchor(y, z, aq, op, extrap) in cubic_anchor.jl

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      INTERNAL: NON-PERIODIC CORE                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Core BCPair scalar: solve per y-vector reusing z buffer
@inline @with_pool pool function _cubic_oneshot_series_bcpair!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq,
        bc::BCPair,
        extrap::AbstractExtrap,
        autocache::Bool,
        op::AbstractEvalOp,
        searcher
    ) where {Tg}
    cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, Tg))
    aq = _anchor_query(cache.x, xq, Val(:cubic), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    Tv_out = _value_type(_series_eltype(s), Tg)
    Tz = _output_eltype(_series_eltype(s), eltype(cache.x))
    n = length(first(vecs))
    z = acquire!(pool, Tz, n)
    y_buf = acquire!(pool, Tv_out, n)
    @inbounds for k in eachindex(output)
        copyto!(y_buf, 1, vecs[k], 1, n)
        _solve_system!(z, cache, y_buf, bc)
        output[k] = _cubic_eval_at_anchor(y_buf, z, aq, op, extrap)
    end
    return output
end

# Build a seam-aware cubic anchor for one query against a (possibly raw
# n-size) periodic cache. Bypasses `_anchor_query_impl` because that helper's
# `_anchor_loc` discards `idx_R`, so it cannot represent the periodic-exclusive
# seam pair `(n, 1)`. Search returns the 4-tuple directly; `_periodic_cell_h`
# supplies the seam-aware width (`bc.h_n` at the seam, spacing accessor
# elsewhere). Used by both scalar and vector periodic series helpers.
@inline function _build_periodic_cubic_anchor(
        cache::CubicSplineCache,
        xq,
        extrap_p::AbstractExtrap,
        searcher::Searcher,
    )
    # `cache.x` is the wrapped axis: `_CachedRange`/`_CachedVector` for
    # `:inclusive`, `_ExclusivePeriodicAxis` for `:exclusive` (virtual length n+1
    # with cached `_x_max`). `_wrap_to_domain(xq, cache.x)` reads `(first, last)`
    # uniformly. The wrapper's `search_interval` returns `idx_R = 1` at seam so
    # raw `y[idx_R]` indexing in the eval kernel works without a data wrapper.
    xq_wrapped = _wrap_to_domain(xq, cache.x)
    idxL, idxR, xL, xR = search_interval(searcher, cache.x, xq_wrapped)
    h = _get_h(cache.x, idxL)
    inv_h = _get_inv_h(cache.x, idxL)
    dL = xq_wrapped - xL
    dR = xR - xq_wrapped
    w0 = _compute_anchor_weights(EvalValue(), h, inv_h, dL, dR)
    w1 = _compute_anchor_weights(EvalDeriv1(), h, inv_h, dL, dR)
    w2 = _compute_anchor_weights(EvalDeriv2(), h, inv_h, dL, dR)
    w3 = _compute_anchor_weights(EvalDeriv3(), h, inv_h, dL, dR)
    return _CubicAnchoredQuery(_IdxPair(idxL, idxR), xq_wrapped, IN_DOMAIN, w0, w1, w2, w3, eltype(cache.x))
end

# Periodic scalar: zero-copy. One search → seam-aware `_IdxPair` anchor → loop
# solve+eval per series. No grid extension, no `y_p` rebuild.
#
# Mirrors `_linear_oneshot_series_periodic!`: wrap query, search once, anchor
# carries seam pair, K-loop solves and evaluates. Caller passes a bc-threaded
# `searcher` so the seam dispatch fires for `:exclusive`.
@inline @with_pool pool function _cubic_oneshot_series_periodic!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq,
        bc::PeriodicBC,
        op::AbstractEvalOp,
        autocache::Bool,
        searcher
    ) where {Tg}
    vecs = _series_vectors(s)
    n = length(x)
    K = length(output)

    # `:inclusive` requires `y[1] ≈ y[end]` per series; `:exclusive` is no-op.
    if bc isa PeriodicBC{:inclusive}
        @inbounds for k in 1:K
            _check_periodic_endpoints(bc, vecs[k])
        end
    end

    # Build cache on the user's grid (BC-aware: `_build_periodic_cache`).
    cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, Tg))
    extrap_p = _resolve_extrap(NoExtrap(), bc, cache.x, first(vecs))
    aq = _build_periodic_cubic_anchor(cache, xq, extrap_p, searcher)

    # Solve + eval per series. For `:exclusive` periodic, wrap each `vecs[k]`
    # with `_ExclusivePeriodicData` so it reports virtual length n+1 to match
    # `length(cache.x)`; the solver and kernel see uniform indexing.
    Tz = _output_eltype(_series_eltype(s), eltype(cache.x))
    z = acquire!(pool, Tz, length(cache.x))
    @inbounds for k in 1:K
        y_eff = _resolve_data(vecs[k], bc)
        _solve_system!(z, cache, y_eff, cache.bc)
        output[k] = _cubic_eval_kernel(y_eff, z, aq, op)
    end
    return output
end

# Periodic vector: zero-copy. Build cache once, pre-fill seam-aware anchors
# for all queries, then solve+eval per series. No grid extension.
@inline function _cubic_oneshot_series_periodic_vec!(
        pool::AbstractArrayPool,
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq},
        bc::PeriodicBC,
        op::AbstractEvalOp,
        autocache::Bool,
        search::AbstractSearchPolicy
    ) where {Tg, Tq <: Real}
    vecs = _series_vectors(s)
    n = length(x)
    K = n_series(s)

    # `:inclusive` validation per-series (`:exclusive` no-op).
    if bc isa PeriodicBC{:inclusive}
        @inbounds for k in 1:K
            _check_periodic_endpoints(bc, vecs[k])
        end
    end

    cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, Tg))
    extrap_p = _resolve_extrap(NoExtrap(), bc, cache.x, first(vecs))

    # Pre-fill seam-aware anchors via `_build_periodic_cubic_anchor`.
    Tg_c = eltype(cache.x)
    Tq_w = promote_type(Tq, Tg_c)
    aq_vec = acquire!(pool, _CubicAnchoredQuery{Tg_c, Tq_w}, length(xqs))
    # `cache.x` is wrapped (`_ExclusivePeriodicAxis(_CachedVector, period)` for
    # `:exclusive`) — axis-level seam dispatch fires via `g.period`. No `bc` thread.
    searcher = _resolve_search(cache.x, xqs, search, nothing)
    @inbounds for j in eachindex(xqs)
        aq_vec[j] = _build_periodic_cubic_anchor(cache, xqs[j], extrap_p, searcher)
    end

    # Solve per series, eval at all queries. For `:exclusive`, wrap `vecs[k]`
    # via `_ExclusivePeriodicData` so it reports virtual n+1 like `cache.x`.
    Tz = _output_eltype(_series_eltype(s), Tg_c)
    z = acquire!(pool, Tz, length(cache.x))
    @inbounds for k in 1:K
        y_eff = _resolve_data(vecs[k], bc)
        _solve_system!(z, cache, y_eff, cache.bc)
        for j in eachindex(xqs)
            outputs[k][j] = _cubic_eval_kernel(y_eff, z, aq_vec[j], op)
        end
    end
    return outputs
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Scalar Series → Vector return (consistent with SeriesInterpolant) ───────

"""
    cubic_interp(x, Series(y1, y2, ...), xq; ...) → Vector
    cubic_interp(x, Series(Y::Matrix), xq; ...) → Vector

One-shot cubic spline interpolation of multiple y-series at a single query point.
Returns `Vector`, consistent with `SeriesInterpolant` output format.

# Strategy
Build cache once → anchor once → solve+eval per y-vector with z-buffer reuse.
"""
@inline function cubic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    x = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    _is_periodic_bc(bc) || _check_domain(x, xq, extrap)
    K = n_series(s)
    Tg_actual = eltype(x)
    output = Vector{_output_eltype(_series_eltype(s), Tg_actual, Tq)}(undef, K)
    # Periodic helper searches against `cache.x` (wrapped from the cache pool),
    # so axis-level dispatch handles seam — no `bc` thread into the Searcher.
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        _cubic_oneshot_series_periodic!(output, x, s, xq, bc, deriv, autocache, searcher)
        return output
    end
    bc_pair = _normalize_bc(bc)
    _cubic_oneshot_series_bcpair!(output, x, s, xq, bc_pair, extrap, autocache, deriv, searcher)
    return output
end

# ─── In-place scalar ─────────────────────────────────────────────────────────

@inline function cubic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    x = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    _is_periodic_bc(bc) || _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        _cubic_oneshot_series_periodic!(output, x, s, xq, bc, deriv, autocache, searcher)
        return output
    end
    bc_pair = _normalize_bc(bc)
    _cubic_oneshot_series_bcpair!(output, x, s, xq, bc_pair, extrap, autocache, deriv, searcher)
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Series-outer loop: pre-compute anchors once, then solve+eval per series.
# O(n + Q) memory (single z buffer + anchor vector) — search done once for all series.
@with_pool pool function cubic_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    x = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    K = n_series(s)
    _validate_series_outputs(outputs, K, length(xqs))
    vecs = _series_vectors(s)

    Tv_out = _value_type(_series_eltype(s), Tg)
    n = length(first(vecs))

    if _is_periodic_bc(bc)
        return _cubic_oneshot_series_periodic_vec!(pool, outputs, x, s, xqs, bc, deriv, autocache, search)
    end

    # Domain check: NoExtrap → throws if OOB, returns InBounds(); others → pass-through
    extrap_eff = _check_domain(x, xqs, extrap)

    bc_pair = _normalize_bc(bc)
    cache = _get_cubic_cache(x, bc_pair, _effective_autocache(autocache, Tg))

    # Pre-compute anchors once (search Q times, not K×Q)
    Tq_w = promote_type(Tq, eltype(cache.x))
    aq_vec = acquire!(pool, _CubicAnchoredQuery{eltype(cache.x), Tq_w}, length(xqs))
    searcher = _resolve_search(cache.x, xqs, search, nothing)
    _fill_anchors!(aq_vec, cache.x, xqs, Val(:cubic), extrap_eff isa WrapExtrap, searcher)

    Tz = _output_eltype(_series_eltype(s), eltype(cache.x))
    z = acquire!(pool, Tz, n)
    y_buf = acquire!(pool, Tv_out, n)

    # Solve z for series k, then eval at all query points before moving to k+1.
    # Eval uses original vecs[k] (not y_buf): _solve_system! mutates y_buf as
    # tridiagonal scratch; the kernel reads y[idx]/y[idx+1] from the original data.
    @inbounds for k in 1:K
        copyto!(y_buf, 1, vecs[k], 1, n)
        _solve_system!(z, cache, y_buf, bc_pair)
        for j in eachindex(xqs)
            outputs[k][j] = _cubic_eval_at_anchor(vecs[k], z, aq_vec[j], deriv, extrap_eff)
        end
    end
    return outputs
end

function cubic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    K = n_series(s)
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    Tv = _output_eltype(_series_eltype(s), Tg_float, Tq)
    outputs = [Vector{Tv}(undef, length(xqs)) for _ in 1:K]
    cubic_interp!(outputs, x, s, xqs; bc, extrap, autocache, deriv, search)
    return outputs
end


# Note: Real wrappers (Tg <: Real) removed — typed methods above handle
# all grid types including ForwardDiff.Dual via _to_float + _promote_grid_float.
