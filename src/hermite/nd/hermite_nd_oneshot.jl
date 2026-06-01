# ========================================
# CubicHermiteInterpolantND — One-shot Evaluation (Pool-based)
# ========================================
#
# Zero-allocation one-shot API. The packed `_NodalDerivativesND`-shaped buffer
# and any Vector-grid extension are acquired from a thread-local
# `AbstractArrayPool` via `@with_pool` (mirrors `_cubic_interp_nd_oneshot`).
# After the first warmup call, every subsequent one-shot — scalar or batch —
# re-uses the same pool buffers, so user loops over many queries with
# changing data / partials see zero per-call allocation.
#
# Internal contract: the pool buffer returned from
# `_pack_and_extend_nodal_derivs_pooled` MUST NOT escape the enclosing
# `@with_pool` block. We satisfy this by consuming it in-line through
# `_eval_nd_cell` and returning a scalar value (or writing into a
# pre-allocated caller-owned `output` for batch).

# ========================================
# Public API entries (scalar + batch + in-place)
# ========================================

"""
    hermite_interp(grids, data, partials, query::Tuple) -> value

ND cubic Hermite one-shot at a single query point.

Zero-allocation after warmup: the packed partials buffer is acquired from a
thread-local pool and reused across calls. Equivalent in result to
`hermite_interp(grids, data, partials)(query)`.

Keywords: same as the persistent-build form (`bc`, `extrap`, `search`, `deriv`).
"""
function hermite_interp(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{<:Any, N},
        partials::HermitePartials{N},
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {N}
    grids_typed, data_typed, partials_typed, bcs, extraps_val, searches, ops =
        _hermite_oneshot_prepare(grids, data, partials, bc, extrap, search, deriv)
    return _hermite_interp_nd_oneshot(
        grids_typed, data_typed, partials_typed, query,
        bcs, extraps_val, searches, ops, hint,
    )
end

"""
    hermite_interp(grids, data, partials, queries) -> Vector

ND cubic Hermite one-shot at multiple query points. Allocates `output`
internally; the packed partials buffer is pool-based so per-query alloc
is zero after warmup.
"""
function hermite_interp(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv_part},
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {Tv, N, Tv_part}
    _, Tg, _, _ = _nd_promote_grids(grids, data)
    Tq = _query_eltype(queries)
    Tr = _output_eltype(_arithmetic_kernel_shape, Tg, promote_type(Tv, Tv_part), Tq)
    output = Vector{Tr}(undef, _query_length(queries))
    hermite_interp!(
        output, grids, data, partials, queries;
        deriv, bc, extrap, search, hint
    )
    return output
end

"""
    hermite_interp!(output, grids, data, partials, queries) -> output

In-place batch ND cubic Hermite one-shot. Zero-alloc workspace after warmup.
"""
function hermite_interp!(
        output::AbstractVector,
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{<:Any, N},
        partials::HermitePartials{N},
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {N}
    grids_typed, data_typed, partials_typed, bcs, extraps_val, searches, ops =
        _hermite_oneshot_prepare(grids, data, partials, bc, extrap, search, deriv)
    return _hermite_interp_nd_oneshot_batch!(
        output, grids_typed, data_typed, partials_typed, queries,
        bcs, extraps_val, ops, searches, hint,
    )
end

# ========================================
# Shared prepare path (kwarg normalization + validation)
# ========================================
#
# Pulled out so scalar/batch APIs share the same validation order and any
# type promotion happens exactly once per call. The actual pool-acquiring
# work happens later inside `_hermite_interp_nd_oneshot[_batch!]`.
@inline function _hermite_oneshot_prepare(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{<:Any, N},
        partials::HermitePartials{N, Tv_part, K},
        bc, extrap, search, deriv,
    ) where {N, Tv_part, K}
    K == (1 << N) - 1 || _throw_partials_not_full_mixed(N, K)

    grids_typed, _, Tv_promoted, _ = _nd_promote_grids(grids, data)
    Tv = promote_type(Tv_promoted, Tv_part)
    data_typed = _coerce_data_eltype(data, Tv, Val(N))
    partials_typed = _coerce_partials_eltype(partials, Tv, Val(N))

    _validate_nd_grids(grids_typed, data_typed)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    _validate_hermite_nd_bcs_phase1a(bcs)
    _validate_partial_sizes(data_typed, partials_typed)
    _validate_inclusive_seams(data_typed, partials_typed, bcs)

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))

    return grids_typed, data_typed, partials_typed, bcs, extraps_val, searches, ops
end

# ========================================
# Pool-based oneshot — scalar
# ========================================
#
# Mirrors `_cubic_interp_nd_oneshot`: validate domain → fill-OOB short
# circuit → pool-acquire packed buffer + extend grids/data wrap rows → run
# the shared search + eval pipeline → return scalar. Pool buffers are
# reclaimed on `@with_pool` rewind.
@with_pool pool function _hermite_interp_nd_oneshot(
        grids::Tuple{Vararg{AbstractVector{Tg}, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        query::Tuple{Vararg{Real, N}},
        bcs::Tuple{Vararg{AbstractBC, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}},
    ) where {Tg, Tv, N, K}
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    grids_p, buf, bcs_p = _pack_and_extend_nodal_derivs_pooled(pool, grids, data, partials, bcs)
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)

    # Use the shared hint helper so the per-axis Ref allocation stays local
    # (stack-elidable). `mono` is `(true,…,true)` for a single point — no
    # monotone-hint optimization needed for one query.
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _scalar_mono(hint, Val(N))

    q_evals = _handle_all_extraps(query, grids_p, extraps_eff)
    indices, Ls, _ = _search_all_intervals(q_evals, grids_p, searches, hints, mono)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids_p, indices, Ls)
    return _eval_nd_cell(buf, indices, hs, inv_hs, dLs, ops)
end

# ========================================
# Pool-based oneshot — batch in-place
# ========================================
#
# Build pass (pack + extend + per-axis extrap resolution + InBounds promotion)
# happens ONCE; then a tight per-query eval loop writes into `output`.
@with_pool pool function _hermite_interp_nd_oneshot_batch!(
        output::AbstractVector,
        grids::Tuple{Vararg{AbstractVector{Tg}, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        queries,
        bcs::Tuple{Vararg{AbstractBC, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        ops::NTuple{N, AbstractEvalOp},
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}},
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}},
    ) where {Tg, Tv, N, K}
    policies, hints = _resolve_oneshot_search_nd(search, queries, hint, Val(N))
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)

    grids_p, buf, bcs_p = _pack_and_extend_nodal_derivs_pooled(pool, grids, data, partials, bcs)
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)
    extraps_eff = _check_domain_nd(grids_p, queries, extraps_eff)

    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids_p, extraps_eff, ops, first(data))
        if oob_val !== nothing
            output[k] = oob_val
            continue
        end
        q_evals = _handle_all_extraps(query_k, grids_p, extraps_eff)
        indices, Ls, _ = _search_all_intervals(q_evals, grids_p, policies, hints)
        hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids_p, indices, Ls)
        output[k] = _eval_nd_cell(buf, indices, hs, inv_hs, dLs, ops)
    end
    return output
end
