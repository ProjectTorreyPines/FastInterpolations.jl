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
    ) where {Tg <: AbstractFloat}
    cache = _get_cubic_cache(x, bc, autocache)
    aq = _anchor_query(cache.x, xq, Val(:cubic), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    Tv_out = _value_type(_series_eltype(s), Tg)
    n = length(first(vecs))
    z = acquire!(pool, Tv_out, n)
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
    ) where {Tg <: AbstractFloat}
    vecs = _series_vectors(s)
    n = length(x)
    Tv_out = _value_type(eltype(first(vecs)), Tg)

    # ── Phase 1: Extend grid + solve first series (establishes cache + x_p) ──
    cache, y_p_first, z = _cubic_periodic_solve!(pool, x, first(vecs), bc, autocache)
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

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Internal: Tuple NTuple return (zero heap alloc) ─────────────────────────
@inline function _cubic_oneshot_series_ntuple(
        x::AbstractVector{Tg},
        s::Series{<:Tuple},
        xq::Tq;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        return _cubic_oneshot_series_periodic_ntuple(x, s, xq, bc, deriv, autocache, searcher)
    end
    bc_pair = _normalize_bc(bc, _series_eltype(s))
    return _cubic_oneshot_series_bcpair_ntuple(x, s, xq, bc_pair, extrap, autocache, deriv, searcher)
end

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
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    K = n_series(s)
    output = Vector{promote_type(_value_type(_series_eltype(s), Tg), Tq)}(undef, K)
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
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
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
    ) where {Tg <: AbstractFloat, Tq <: Real}
    _validate_series_lengths(s, length(x))
    K = n_series(s)
    length(outputs) == K || _throw_series_dim_mismatch(length(outputs), K)
    vecs = _series_vectors(s)
    searcher = _resolve_search(x, xqs, search, nothing)

    # Solve z for each y-vector upfront (need all z for the query loop)
    if _is_periodic_bc(bc)
        throw(ArgumentError("Vector query with PeriodicBC not yet supported for one-shot Series. Use pre-built interpolant."))
    end
    Tv_out = _value_type(_series_eltype(s), Tg)
    n = length(first(vecs))
    zs = [acquire!(pool, Tv_out, n) for _ in 1:K]
    y_buf = acquire!(pool, Tv_out, n)
    bc_pair = _normalize_bc(bc, _series_eltype(s))
    cache = _get_cubic_cache(x, bc_pair, autocache)
    for k in 1:K
        copyto!(y_buf, 1, vecs[k], 1, n)
        _solve_system!(zs[k], cache, y_buf, bc_pair)
    end

    # Eval uses original vecs[k] (not y_buf): _solve_system! only needs y for the
    # tridiagonal solve scratch; the kernel reads y[idx]/y[idx+1] from the original data.
    wrap = extrap isa WrapExtrap
    @inbounds for j in eachindex(xqs)
        aq = _anchor_query(cache.x, xqs[j], Val(:cubic), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _cubic_eval_at_anchor(vecs[k], zs[k], aq, deriv, extrap)
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
    ) where {Tg <: AbstractFloat, Tq <: Real}
    K = n_series(s)
    Tv = promote_type(_value_type(_series_eltype(s), Tg), Tq)
    outputs = [Vector{Tv}(undef, length(xqs)) for _ in 1:K]
    cubic_interp!(outputs, x, s, xqs; bc, extrap, autocache, deriv, search)
    return outputs
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     REAL TYPE PROMOTION WRAPPERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@inline function cubic_interp(
        x::AbstractVector{Tg}, s::Series, xq::Tq; bc::AbstractBC = CubicFit(), kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return cubic_interp(_to_float(x, Tg_float), s, xq; bc = _promote_bc(bc, Tg_float), kwargs...)
end

@inline function cubic_interp!(
        output::AbstractVector, x::AbstractVector{Tg}, s::Series, xq::Tq;
        bc::AbstractBC = CubicFit(), kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return cubic_interp!(output, _to_float(x, Tg_float), s, xq; bc = _promote_bc(bc, Tg_float), kwargs...)
end

function cubic_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg}, s::Series, xqs::AbstractVector{Tq};
        bc::AbstractBC = CubicFit(), kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return cubic_interp!(
        outputs, _to_float(x, Tg_float), s, _to_float(xqs, Tg_float);
        bc = _promote_bc(bc, Tg_float), kwargs...
    )
end

function cubic_interp(
        x::AbstractVector{Tg}, s::Series, xqs::AbstractVector{Tq};
        bc::AbstractBC = CubicFit(), kwargs...
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    return cubic_interp(
        _to_float(x, Tg_float), s, _to_float(xqs, Tg_float);
        bc = _promote_bc(bc, Tg_float), kwargs...
    )
end
