# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  CUBIC ONE-SHOT SERIES INTERPOLATION                     ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides `cubic_interp(x, Series(y1,y2,...), xq; bc=...)` without constructing
# a CubicSeriesInterpolant. Uses pool allocation for z-buffer reuse.
#
# Include order: ... → cubic_anchor.jl → cubic_series_payloads.jl → this file.
# Eval goes through the lean payload adapters (`_cubic_series_eval`,
# `_cubic_payload_kernel`) in cubic_series_payloads.jl.

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
    # One lean op/extrap-aware anchor, built from the statically-typed `extrap`
    # (same helpers as the vector one-shot path); raw-vector eval per series.
    A = _cubic_series_anchor_type(op, extrap, cache.x, _coord_eltype(typeof(xq), eltype(cache.x)))
    a = _build_series_anchor(CubicInterp(), A, cache.x, xq, extrap, extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    Tv_out = _value_type(_series_eltype(s), Tg)
    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), _series_eltype(s))
    n = length(first(vecs))
    z = acquire!(pool, Tz, n)
    y_buf = acquire!(pool, Tv_out, n)
    @inbounds for k in eachindex(output)
        copyto!(y_buf, 1, vecs[k], 1, n)
        _solve_system!(z, cache, y_buf, bc)
        output[k] = _cubic_series_eval(y_buf, z, a, extrap)
    end
    return output
end

# Periodic scalar: zero-copy. One search → seam-aware `_ExplicitIndices` anchor → loop
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
    # Seam-aware LEAN anchor (bare payload — periodic eval always wraps in-domain).
    A = _AxisAnchor{_interval_type(cache.x), _cubic_series_payload_type(op, _coord_eltype(typeof(xq), eltype(cache.x)))}
    a = _build_periodic_series_anchor(A, cache, xq, searcher)

    # Solve + eval per series. For `:exclusive` periodic, wrap each `vecs[k]`
    # with `_ExclusivePeriodicData` so it reports virtual length n+1 to match
    # `length(cache.x)`; the solver and kernel see uniform indexing.
    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), _series_eltype(s))
    z = acquire!(pool, Tz, length(cache.x))
    @inbounds for k in 1:K
        y_eff = _resolve_data(vecs[k], bc)
        _solve_system!(z, cache, y_eff, cache.bc)
        output[k] = _cubic_payload_kernel(y_eff, z, a)
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

    # Pre-fill seam-aware LEAN anchors (bare payload — periodic eval has no
    # extrap dispatch, queries always wrap in-domain).
    Tg_c = eltype(cache.x)
    Tq_w = _coord_eltype(Tq, Tg_c)
    A = _AxisAnchor{_interval_type(cache.x), _cubic_series_payload_type(op, Tq_w)}
    anchors = acquire!(pool, A, length(xqs))
    # `cache.x` is wrapped (`_ExclusivePeriodicAxis(_CachedVector, period)` for
    # `:exclusive`) — axis-level seam dispatch fires via `g.period`. No `bc` thread.
    searcher = _resolve_search(cache.x, xqs, search, nothing)
    @inbounds for j in eachindex(xqs)
        anchors[j] = _build_periodic_series_anchor(A, cache, xqs[j], searcher)
    end

    # Solve per series, eval at all queries. For `:exclusive`, wrap `vecs[k]`
    # via `_ExclusivePeriodicData` so it reports virtual n+1 like `cache.x`.
    Tz = _promote_eltype(_coeff_op2, Tg_c, _series_eltype(s))
    z = acquire!(pool, Tz, length(cache.x))
    @inbounds for k in 1:K
        y_eff = _resolve_data(vecs[k], bc)
        _solve_system!(z, cache, y_eff, cache.bc)
        for j in eachindex(xqs)
            outputs[k][j] = _cubic_payload_kernel(y_eff, z, anchors[j])
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
    output = Vector{_promote_eltype(_interp_op, Tg_actual, _series_eltype(s), Tq)}(undef, K)
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

    bc_pair = _normalize_bc(bc)
    cache = _get_cubic_cache(x, bc_pair, _effective_autocache(autocache, Tg))

    # Pre-compute lean op/extrap-aware anchors once (search Q times, not K×Q),
    # built from the statically-typed `extrap` so the pooled anchor vector stays
    # concretely typed. `_check_domain`'s in-domain promotion returns a Union for
    # Clamp/Fill/Wrap; deriving the anchor type from it would box the pool acquire.
    # Mirrors the persistent batch entry: per-query state comes from the search and
    # NoExtrap throws OOB inside this build, before any output is written.
    Tq_w = _coord_eltype(Tq, eltype(cache.x))
    A = _cubic_series_anchor_type(deriv, extrap, cache.x, Tq_w)
    anchors = acquire!(pool, A, length(xqs))
    _fill_series_anchors_resolved!(CubicInterp(), anchors, cache.x, xqs, extrap, extrap isa WrapExtrap, search, nothing)

    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), _series_eltype(s))
    z = acquire!(pool, Tz, n)
    y_buf = acquire!(pool, Tv_out, n)

    # Solve z for series k, then eval at all query points before moving to k+1.
    # Eval uses original vecs[k] (not y_buf): _solve_system! mutates y_buf as
    # tridiagonal scratch; the kernel reads y[idx]/y[idx+1] from the original data.
    @inbounds for k in 1:K
        copyto!(y_buf, 1, vecs[k], 1, n)
        _solve_system!(z, cache, y_buf, bc_pair)
        for j in eachindex(xqs)
            outputs[k][j] = _cubic_series_eval(vecs[k], z, anchors[j], extrap)
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
    Tv = _promote_eltype(_interp_op, Tg_float, _series_eltype(s), Tq)
    outputs = _alloc_series_batch_outputs(Tv, K, length(xqs))
    cubic_interp!(outputs, x, s, xqs; bc, extrap, autocache, deriv, search)
    return outputs
end


# Note: Real wrappers (Tg <: Real) removed — typed methods above handle
# all grid types including ForwardDiff.Dual via _to_float + _promote_grid_float.
