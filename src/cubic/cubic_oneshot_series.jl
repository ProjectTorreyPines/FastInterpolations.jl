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

# Periodic: extend grid ONCE, then solve+eval per y-vector reusing buffers.
# Phase 1: _cubic_periodic_solve! for first series → establishes cache, x_p, y_p, z
# Phase 2: reuse y_p + z for remaining series (only y-data changes, no grid re-extension)
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
    Tv_out = _value_type(_series_eltype(s), Tg)

    # Promote first series to Tv_out so z/y_p buffers match all series (mixed-type safety)
    y1_promoted = acquire!(pool, Tv_out, n)
    copyto!(y1_promoted, 1, first(vecs), 1, n)

    # ── Phase 1: Extend grid + solve first series (establishes cache + x_p) ──
    cache, y_p_first, z = _cubic_periodic_solve!(pool, x, y1_promoted, bc, autocache)
    n_p = length(y_p_first)  # inclusive length (n or n+1 depending on BC mode)

    # Build anchor on the extended (inclusive) grid — once for all series
    aq = _anchor_query(cache.x, xq, Val(:cubic), true, searcher)
    output[1] = _cubic_eval_kernel(y_p_first, z, aq, op)

    # ── Phase 2: Reuse z buffer for remaining series (no grid re-extension) ──
    # For exclusive BC: y_p_first is a pool buffer of length n+1, safe to overwrite
    # For inclusive BC: y_p_first IS the user's vector, must NOT mutate it
    is_exclusive = bc isa PeriodicBC{:exclusive}
    y_p = if is_exclusive
        y_p_first  # pool buffer, safe to reuse
    else
        acquire!(pool, Tv_out, n_p)  # separate buffer for inclusive BC
    end

    for k in 2:length(output)
        if is_exclusive
            @inbounds copyto!(y_p, 1, vecs[k], 1, n)
            @inbounds y_p[n + 1] = vecs[k][1]
        else
            @inbounds copyto!(y_p, 1, vecs[k], 1, n_p)
        end
        _check_periodic_endpoints(bc, y_p)
        _solve_system!(z, cache, y_p, cache.bc_config)
        @inbounds output[k] = _cubic_eval_kernel(y_p, z, aq, op)
    end
    return output
end

# Periodic vector: extend grid once, pre-compute anchors, then solve+eval per series.
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
    ) where {Tg,Tq <: Real}
    vecs = _series_vectors(s)
    n = length(x)
    K = n_series(s)
    Tv_out = _value_type(_series_eltype(s), Tg)

    # Promote first series to Tv_out so z/y_p buffers match all series (mixed-type safety)
    y1_promoted = acquire!(pool, Tv_out, n)
    copyto!(y1_promoted, 1, first(vecs), 1, n)

    # Phase 1: Extend grid + solve first series (establishes cache + x_p)
    cache, y_p_first, z = _cubic_periodic_solve!(pool, x, y1_promoted, bc, autocache)
    n_p = length(y_p_first)

    # Pre-compute anchors on the extended (inclusive) grid — once for all series
    # Tq widens when grid is Dual: promote_type(Float64, Dual) = Dual
    Tq_w = promote_type(Tq, eltype(cache.x))
    aq_vec = acquire!(pool, _CubicAnchoredQuery{eltype(cache.x), Tq_w}, length(xqs))
    searcher = _resolve_search(cache.x, xqs, search, nothing)
    _fill_anchors!(aq_vec, cache.x, xqs, Val(:cubic), true, searcher)

    # Eval first series (already solved)
    @inbounds for j in eachindex(xqs)
        outputs[1][j] = _cubic_eval_kernel(y_p_first, z, aq_vec[j], op)
    end

    # Phase 2: Reuse z buffer for remaining series
    is_exclusive = bc isa PeriodicBC{:exclusive}
    y_p = if is_exclusive
        y_p_first  # pool buffer, safe to reuse
    else
        acquire!(pool, Tv_out, n_p)  # separate buffer for inclusive BC
    end

    for k in 2:K
        if is_exclusive
            @inbounds copyto!(y_p, 1, vecs[k], 1, n)
            @inbounds y_p[n + 1] = vecs[k][1]
        else
            @inbounds copyto!(y_p, 1, vecs[k], 1, n_p)
        end
        _check_periodic_endpoints(bc, y_p)
        _solve_system!(z, cache, y_p, cache.bc_config)
        @inbounds for j in eachindex(xqs)
            outputs[k][j] = _cubic_eval_kernel(y_p, z, aq_vec[j], op)
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
    ) where {Tg,Tq <: Real}
    _validate_series_lengths(s, length(x))
    x = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    _is_periodic_bc(bc) || _check_domain(x, xq, extrap)
    K = n_series(s)
    Tg_actual = eltype(x)
    output = Vector{_series_output_type(_output_eltype(_series_eltype(s), Tg_actual), Tq)}(undef, K)
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        _cubic_oneshot_series_periodic!(output, x, s, xq, bc, deriv, autocache, searcher)
        return output
    end
    bc_pair = _normalize_bc(bc, _series_eltype(s))
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
    ) where {Tg,Tq <: Real}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    x = _to_float(x, _promote_grid_float(Tg, _series_eltype(s)))
    _is_periodic_bc(bc) || _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        _cubic_oneshot_series_periodic!(output, x, s, xq, bc, deriv, autocache, searcher)
        return output
    end
    bc_pair = _normalize_bc(bc, _series_eltype(s))
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
    ) where {Tg,Tq <: Real}
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

    bc_pair = _normalize_bc(bc, _series_eltype(s))
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
    ) where {Tg,Tq <: Real}
    K = n_series(s)
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    Tv = _series_output_type(_output_eltype(_series_eltype(s), Tg_float), Tq)
    outputs = [Vector{Tv}(undef, length(xqs)) for _ in 1:K]
    cubic_interp!(outputs, x, s, xqs; bc, extrap, autocache, deriv, search)
    return outputs
end


# Note: Real wrappers (Tg <: Real) removed — typed methods above handle
# all grid types including ForwardDiff.Dual via _to_float + _promote_grid_float.
