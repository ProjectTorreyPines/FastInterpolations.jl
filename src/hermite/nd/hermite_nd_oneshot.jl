# ========================================
# CubicHermiteInterpolantND — One-shot Evaluation (Pool-based)
# ========================================
#
# Zero-allocation one-shot API. The packed buffer and any Vector-grid
# extension are acquired from a thread-local pool via `@with_pool`, so loops
# over many queries (even with changing data / partials) see zero per-call
# alloc after warmup. The pool buffer must not escape the `@with_pool` block:
# it is consumed in-line through `_eval_nd_cell`, returning a scalar (or
# writing into the caller-owned `output` for batch).

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
        query::Tuple{Vararg{Number, N}};
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
    # `Tg` value-matched as in `_hermite_oneshot_prepare`, so `Tr` agrees with the eval.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), promote_type(Tv, Tv_part))
    Tq = _query_eltype(queries)
    Tr = _promote_eltype(_interp_op, Tg, promote_type(Tv, Tv_part), Tq)
    output = _alloc_query_output(Tr, queries)
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
        output::AbstractArray,
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
# Shared by scalar/batch so validation + promotion run once, identically.
# Pool acquisition happens later in `_hermite_interp_nd_oneshot[_batch!]`.
@inline function _hermite_oneshot_prepare(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{<:Any, N},
        partials::HermitePartials{N, Tv_part, K},
        bc, extrap, search, deriv,
    ) where {N, Tv_part, K}
    K == (1 << N) - 1 || _throw_partials_not_full_mixed(N, K)
    # Non-Real axes: same friendly refusal as the persistent ctor.
    _check_nd_hetero_grid(_promote_grid_eltype(grids))

    # Raw grids: the pack + cell-eval float the cell width, so no eager `Tg.(x)` copy.
    # `Tg` value-matched to data∪partials (Int grid + Float32 → Float32) keeps `Tv` narrow,
    # so `_coerce_*_eltype` pass through instead of copying data + K partials per call.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), promote_type(eltype(data), Tv_part))
    Tv_promoted = _promote_eltype(_coeff_op, Tg, eltype(data))
    Tv = promote_type(Tv_promoted, Tv_part)
    data_typed = _coerce_data_eltype(data, Tv, Val(N))
    partials_typed = _coerce_partials_eltype(partials, Tv, Val(N))

    _validate_nd_grids(grids, data_typed)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    _validate_hermite_nd_bcs(bcs)
    _validate_partial_sizes(data_typed, partials_typed)
    _validate_inclusive_seams(data_typed, partials_typed, bcs)

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))

    return grids, data_typed, partials_typed, bcs, extraps_val, searches, ops
end

# ========================================
# Pool-based oneshot — scalar
# ========================================
#
# validate domain → fill-OOB short circuit → pool-pack + extend → shared
# search + eval → scalar.
@with_pool pool function _hermite_interp_nd_oneshot(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        query::Tuple{Vararg{Number, N}},
        bcs::Tuple{Vararg{AbstractBC, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}},
    ) where {Tv, N, K}
    # Bare GridIdx(k).val is NaN → resolve to the grid coordinate for the value kernel (search still uses .idx).
    query = map(_resolve_grididx, query, grids)
    # Validate + per-axis NoExtrap→InBounds promotion, matching the other ND one-shot scalar paths.
    extraps_val = _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    # `bcs_post` (3rd return) is unused: no partial-solve step here, and
    # extrap is materialized from `extraps_val` + `grids_p`. `Tg` value-matches the
    # width/extension type (Int grid + Float32 data → Float32) — grids stay raw.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    grids_p, buf, _ = _pack_and_extend_nodal_derivs_pooled(pool, grids, data, partials, bcs, Tg)
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)

    # `mono` is all-true for a single point (no monotone-hint optimization).
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _scalar_mono(hint, Val(N))

    q_evals = _handle_all_extraps(query, grids_p, extraps_eff)
    indices, Ls, _ = _search_all_intervals(q_evals, grids_p, searches, hints, mono, extraps_eff)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids_p, indices, Ls, Tg)
    return _eval_nd_cell(buf, indices, hs, inv_hs, dLs, ops)
end

# ========================================
# Pool-based oneshot — batch in-place
# ========================================
#
# Build pass (pack + extend + per-axis extrap resolution + InBounds promotion)
# happens ONCE; then a tight per-query eval loop writes into `output`.
@with_pool pool function _hermite_interp_nd_oneshot_batch!(
        output::AbstractArray,
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        queries,
        bcs::Tuple{Vararg{AbstractBC, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        ops::NTuple{N, AbstractEvalOp},
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}},
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}},
    ) where {Tv, N, K}
    policies, hints = _resolve_oneshot_search_nd(search, queries, hint, Val(N))
    nq = _query_length(queries)
    _check_query_output_size(output, queries)
    _query_validate(queries)

    # `bcs_post` (3rd return) is unused here — see the scalar path note.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    grids_p, buf, _ = _pack_and_extend_nodal_derivs_pooled(pool, grids, data, partials, bcs, Tg)
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)
    extraps_eff = _validate_nd_domain(grids_p, queries, extraps_eff)

    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N), grids_p)
        oob_val = _try_fill_oob(query_k, grids_p, extraps_eff, ops, first(data))
        if oob_val !== nothing
            output[k] = oob_val
            continue
        end
        q_evals = _handle_all_extraps(query_k, grids_p, extraps_eff)
        indices, Ls, _ = _search_all_intervals(q_evals, grids_p, policies, hints, extraps_eff)
        hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids_p, indices, Ls, Tg)
        output[k] = _eval_nd_cell(buf, indices, hs, inv_hs, dLs, ops)
    end
    return output
end
