# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                CONSTANT ONE-SHOT SERIES INTERPOLATION                    ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: ... → constant_anchor.jl → constant_oneshot_series.jl → ...
# Shared anchor eval: _constant_eval_at_anchor(y, x_last, aq, op, side, extrap) in constant_anchor.jl

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: PERIODIC CORE                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Zero-copy scalar-series periodic anchor + K-loop eval. Mirrors the Linear
# counterpart: wrap `xq` using a BC-materialized `WrapExtrap`, search once,
# reuse the pair-aware anchor for all K series. The `x_last` passed to
# `_constant_eval_at_anchor` is the ORIGINAL grid's right endpoint (x[n]) —
# it preserves the right-continuous special case for inclusive queries at
# exactly `xq == x[n]` (returns `y[end]`), matching the non-series constant
# convention.
@inline function _constant_oneshot_series_periodic!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq,
        bc::PeriodicBC,
        op::AbstractEvalOp,
        side::AbstractSide,
        searcher
    ) where {Tg}
    vecs = _series_vectors(s)
    K = length(output)

    if bc isa PeriodicBC{:inclusive}
        @inbounds for k in 1:K
            _check_periodic_endpoints(bc, vecs[k])
        end
    end

    # Surface-API axis resolution: `:exclusive` axis → `_ExclusivePeriodicAxis`
    # carrying period + virtual endpoint; specialized search returns post-fold
    # `idx_R` so raw `vecs[k][aq.idxR]` reads the wrapped corner directly.
    x_eff = _resolve_axis(x, bc)
    extrap_p = _resolve_extrap(NoExtrap(), bc, x_eff)   # PeriodicBC → WrapExtrap()
    xq_wrapped = _wrap_to_domain(xq, x_eff)
    idxL, idxR, xL, _ = search_interval(searcher, x_eff, xq_wrapped)
    # 2-arg cached geometry (`_get_h(x_eff, idxL)`), seam pair `(idxL, idxR)`, and
    # domain-max at the ORIGINAL right endpoint (`last(x_eff)`) — see the periodic
    # builder. Bare payload (periodic always wraps in-domain).
    A = _constant_series_anchor_type(op, extrap_p, x_eff, _coord_eltype(typeof(xq), eltype(x_eff)))
    a = _build_constant_periodic_series_anchor(A, x_eff, xq_wrapped, idxL, idxR, xL, ConstantInterp(side))
    @inbounds for k in 1:K
        output[k] = _constant_series_eval(vecs[k], a, extrap_p)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Scalar Series → Vector return (consistent with SeriesInterpolant) ───────

@inline function constant_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    x = _to_float(x, Tg)
    K = n_series(s)
    Tv = _series_eltype(s)
    Tv_out = _promote_eltype(_select_op, Tg, Tv, Tq)
    output = Vector{Tv_out}(undef, K)
    if _is_periodic_bc(bc)
        # Helper wraps `x` via `_resolve_axis(x, bc)` and searches against the
        # wrapped axis — axis dispatch handles seam, no `bc` thread needed.
        searcher = _resolve_search(x, xq, search, hint)
        return _constant_oneshot_series_periodic!(output, x, s, xq, bc, deriv, side, searcher)
    end
    _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    A = _constant_series_anchor_type(deriv, extrap, x, _coord_eltype(Tq, eltype(x)))
    a = _build_series_anchor(ConstantInterp(side), A, x, xq, extrap, extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    @inbounds for k in 1:K
        output[k] = _constant_series_eval(vecs[k], a, extrap)
    end
    return output
end

# ─── In-place scalar ─────────────────────────────────────────────────────────

@inline function constant_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    x = _to_float(x, Tg)
    if _is_periodic_bc(bc)
        searcher = _resolve_search(x, xq, search, hint)
        return _constant_oneshot_series_periodic!(output, x, s, xq, bc, deriv, side, searcher)
    end
    _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    A = _constant_series_anchor_type(deriv, extrap, x, _coord_eltype(Tq, eltype(x)))
    a = _build_series_anchor(ConstantInterp(side), A, x, xq, extrap, extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    @inbounds for k in eachindex(output)
        output[k] = _constant_series_eval(vecs[k], a, extrap)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Q outer × K inner — small-NQ fast path. Anchor is stack-resident across
# the K-inner loop. No pool acquired, so callers with tiny `xqs` don't pay
# pool-setup overhead. `_is_periodic_bc(bc)` is compile-time-resolved.
@inline function _constant_series_batch_qk!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector,
        vecs,
        xqs::AbstractVector,
        bc::AbstractBC,
        op::AbstractEvalOp,
        side::AbstractSide,
        extrap::AbstractExtrap,
        search::AbstractSearchPolicy
    )
    K = length(vecs)
    NQ = length(xqs)

    if _is_periodic_bc(bc)
        x_eff = _resolve_axis(x, bc)
        extrap_p = _resolve_extrap(NoExtrap(), bc, x_eff)
        searcher = _resolve_search(x_eff, xqs, search, nothing)
        m = ConstantInterp(side)
        A = _constant_series_anchor_type(op, extrap_p, x_eff, _coord_eltype(eltype(xqs), eltype(x_eff)))
        @inbounds for j in 1:NQ
            xq_wrapped = _wrap_to_domain(xqs[j], x_eff)
            idxL, idxR, xL, _ = search_interval(searcher, x_eff, xq_wrapped)
            a = _build_constant_periodic_series_anchor(A, x_eff, xq_wrapped, idxL, idxR, xL, m)
            for k in 1:K
                outputs[k][j] = _constant_series_eval(vecs[k], a, extrap_p)
            end
        end
        return outputs
    end

    extrap_eff = _check_domain(x, xqs, extrap)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap_eff isa WrapExtrap
    m = ConstantInterp(side)
    A = _constant_series_anchor_type(op, extrap_eff, x, _coord_eltype(eltype(xqs), eltype(x)))
    @inbounds for j in 1:NQ
        a = _build_series_anchor(m, A, x, xqs[j], extrap_eff, wrap, searcher)
        for k in 1:K
            outputs[k][j] = _constant_series_eval(vecs[k], a, extrap_eff)
        end
    end
    return outputs
end

# K outer × Q inner with pool-acquired anchor vector — large-NQ fast path.
# Inner loop streams a single `outputs[k]` Vector, which LLVM auto-SIMDs
# (~K× fewer cache-line jumps on the write side than the Q×K shape).
@inline @with_pool pool function _constant_series_batch_kq!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        vecs,
        xqs::AbstractVector{Tq},
        bc::AbstractBC,
        op::AbstractEvalOp,
        side::AbstractSide,
        extrap::AbstractExtrap,
        search::AbstractSearchPolicy
    ) where {Tg, Tq <: Real}
    K = length(vecs)
    NQ = length(xqs)

    if _is_periodic_bc(bc)
        x_eff = _resolve_axis(x, bc)
        extrap_p = _resolve_extrap(NoExtrap(), bc, x_eff)
        searcher = _resolve_search(x_eff, xqs, search, nothing)
        m = ConstantInterp(side)
        A = _constant_series_anchor_type(op, extrap_p, x_eff, _coord_eltype(Tq, eltype(x_eff)))
        anchors = acquire!(pool, A, NQ)
        @inbounds for j in 1:NQ
            xq_wrapped = _wrap_to_domain(xqs[j], x_eff)
            idxL, idxR, xL, _ = search_interval(searcher, x_eff, xq_wrapped)
            anchors[j] = _build_constant_periodic_series_anchor(A, x_eff, xq_wrapped, idxL, idxR, xL, m)
        end
        @inbounds for k in 1:K
            for j in 1:NQ
                outputs[k][j] = _constant_series_eval(vecs[k], anchors[j], extrap_p)
            end
        end
        return outputs
    end

    extrap_eff = _check_domain(x, xqs, extrap)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap_eff isa WrapExtrap
    m = ConstantInterp(side)
    A = _constant_series_anchor_type(op, extrap_eff, x, _coord_eltype(Tq, eltype(x)))
    anchors = acquire!(pool, A, NQ)
    _fill_series_anchors!(m, anchors, x, xqs, extrap_eff, wrap, searcher)
    @inbounds for k in 1:K
        for j in 1:NQ
            outputs[k][j] = _constant_series_eval(vecs[k], anchors[j], extrap_eff)
        end
    end
    return outputs
end

# Adaptive entry. `length(xqs)` and `K` feed `_series_use_kq_loop` to select
# the loop order — see `src/core/series_utils.jl`.
# `x_last = Tg(last(x))` is computed inside each helper from the same
# `x_eff` it uses (preserves the right-continuous short-circuit inside
# `_constant_eval_at_anchor` for inclusive queries at `xq == x[n]`).
@inline function constant_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    x = _to_float(x, Tg)
    K = n_series(s)
    _validate_series_outputs(outputs, K, length(xqs))
    vecs = _series_vectors(s)
    if bc isa PeriodicBC{:inclusive}
        @inbounds for k in 1:K
            _check_periodic_endpoints(bc, vecs[k])
        end
    end

    if _series_use_kq_loop(length(xqs), K)
        return _constant_series_batch_kq!(outputs, x, vecs, xqs, bc, deriv, side, extrap, search)
    else
        return _constant_series_batch_qk!(outputs, x, vecs, xqs, bc, deriv, side, extrap, search)
    end
end

function constant_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    K = n_series(s)
    Tv = _series_eltype(s)
    Tv_out = _promote_eltype(_select_op, Tg, Tv, Tq)
    outputs = _alloc_series_batch_outputs(Tv_out, K, length(xqs))
    constant_interp!(outputs, x, s, xqs; bc, side, extrap, deriv, search)
    return outputs
end
