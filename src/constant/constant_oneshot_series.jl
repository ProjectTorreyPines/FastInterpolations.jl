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
    h = _get_h(x_eff, idxL)
    dL = xq_wrapped - xL
    # Promote xq to match dL type (Float64 query + Dual grid → dL is Dual).
    xq_promoted = oftype(dL, xq_wrapped)
    aq = _ConstantAnchoredQuery(_IdxPair(idxL, idxR), xq_promoted, IN_DOMAIN, h, dL)

    x_last = @inbounds Tg(last(x_eff))

    @inbounds for k in 1:K
        output[k] = _constant_eval_at_anchor(vecs[k], x_last, aq, op, side, extrap_p)
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
    Tv_out = _series_eltype(s)
    output = Vector{Tv_out}(undef, K)
    if _is_periodic_bc(bc)
        searcher = _resolve_search(x, xq, search, hint, bc)
        return _constant_oneshot_series_periodic!(output, x, s, xq, bc, deriv, side, searcher)
    end
    _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    @inbounds for k in 1:K
        output[k] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
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
        searcher = _resolve_search(x, xq, search, hint, bc)
        return _constant_oneshot_series_periodic!(output, x, s, xq, bc, deriv, side, searcher)
    end
    _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    @inbounds for k in eachindex(output)
        output[k] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Zero-pool vector-batch: Q outer × K inner, mirrors the Linear counterpart
# with `side` + `x_last` propagation. `x_last = Tg(last(x))` on the ORIGINAL
# grid — preserves the right-continuous short-circuit inside
# `_constant_eval_at_anchor` for inclusive queries at `xq == x[n]`.
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
    Tg_actual = eltype(x)

    if _is_periodic_bc(bc)
        if bc isa PeriodicBC{:inclusive}
            @inbounds for k in 1:K
                _check_periodic_endpoints(bc, vecs[k])
            end
        end
        x_eff = _resolve_axis(x, bc)
        extrap_p = _resolve_extrap(NoExtrap(), bc, x_eff)
        searcher = _resolve_search(x_eff, xqs, search, nothing, NoBC())
        x_last = @inbounds Tg_actual(last(x_eff))
        @inbounds for j in eachindex(xqs)
            xq_wrapped = _wrap_to_domain(xqs[j], x_eff)
            idxL, idxR, xL, _ = search_interval(searcher, x_eff, xq_wrapped)
            # Index-based dispatch (cached step on Range, seam width on wrapper).
            h = _get_h(x_eff, idxL)
            dL = xq_wrapped - xL
            xq_promoted = oftype(dL, xq_wrapped)
            aq = _ConstantAnchoredQuery(_IdxPair(idxL, idxR), xq_promoted, IN_DOMAIN, h, dL)
            for k in 1:K
                outputs[k][j] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap_p)
            end
        end
        return outputs
    end

    # Non-periodic: `_anchor_query` handles WrapExtrap query-wrap + OOB state.
    extrap_eff = _check_domain(x, xqs, extrap)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap_eff isa WrapExtrap
    x_last = @inbounds Tg_actual(last(x))
    @inbounds for j in eachindex(xqs)
        aq = _anchor_query(x, xqs[j], Val(:constant), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap_eff)
        end
    end
    return outputs
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
    Tv_out = _series_eltype(s)
    outputs = [Vector{Tv_out}(undef, length(xqs)) for _ in 1:K]
    constant_interp!(outputs, x, s, xqs; bc, side, extrap, deriv, search)
    return outputs
end
