# ========================================
# ND Quadratic Interpolation — One-Shot Evaluation
# ========================================
#
# Zero-allocation one-shot API and pool-based backends for ND quadratic interpolation.
# Interpolant construction is in quadratic_nd_interpolant.jl.

# ========================================
# POOL-BASED ONE-SHOT IMPLEMENTATION
# ========================================
#
# Pool-based evaluation functions that bypass Interpolant construction.
# Nodal derivatives are computed in pool buffers (reused across calls).
# Zero heap allocation for scalar queries after warmup.

"""
    _quadratic_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops, hints=nothing)

Pool-based scalar one-shot ND quadratic evaluation.
Computes 2^N partial derivatives in a pool buffer and evaluates at a single point.
Zero-allocation after warmup (pool reuse).
"""
@with_pool pool function _quadratic_interp_nd_oneshot(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing
    ) where {Tv, N}
    # Bare GridIdx(k).val is NaN → resolve to the grid coordinate for the value kernel (search still uses .idx).
    query = map(_resolve_grididx, query, grids)
    # 0. Validate (NoExtrap throw must precede FillExtrap short-circuit) AND promote per axis:
    #    an in-domain NoExtrap axis becomes InBounds for the lean search; InBounds no-ops through
    #    `_try_fill_oob` / `_resolve_extrap` / `_handle_all_extraps`.
    extraps_val = _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    # 1. Value-matched grid float (Int/OneTo grid + Float32 data → Float32) shared by the pooled
    # per-axis cache and the partials type, so the pool buffer + output follow natural promote_type.
    # `map` dispatches per-element on the concrete axis type (Range vs Vector).
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    grids_c = _cache_axes_pooled(pool, grids, Tg)  # @generated static-Tg unroll (no Type-captured closure)

    # 2. Pool-allocate partials array (THE KEY: pool instead of heap). Tz widens Tv with Tg
    # (Dual grid → Dual derivs); `_coeff_op` floats Int.
    Tz = _promote_eltype(_coeff_op, Tg, Tv)
    n_partials = 1 << N
    partials = acquire!(pool, Tz, (n_partials, size(data)...))

    # 3. Compute all partial derivatives in-place
    _compute_nd_partials_quadratic!(partials, grids_c, data, bcs)

    # 4. Per-axis extrap passthrough against the (possibly extended) grid.
    extraps_eff = map(_resolve_extrap, extraps_val, grids_c)

    # 5. Eval pipeline (axis-only — grids carry `h`/`inv_h` directly)
    q_eval = _handle_all_extraps(query, grids_c, extraps_eff)
    indices, Ls, _ = _search_all_intervals(q_eval, grids_c, searches, hints, extraps_eff)
    _, inv_hs, dLs = _compute_all_local_params(q_eval, grids_c, indices, Ls)

    # 6. Tensor-product kernel evaluation (quadratic uses only inv_h/dL)
    return _eval_nd_quad_cell(partials, indices, inv_hs, dLs, ops)
end

"""
    _quadratic_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps_val, searches, ops, hints=nothing)

Pool-based in-place batch one-shot ND quadratic evaluation.
Computes partials ONCE, then evaluates at all query points into `output`.
Uses query protocol (`_query_length`, `_query_extract`) — works with any query format.
"""
@with_pool pool function _quadratic_interp_nd_oneshot_batch!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        ops::NTuple{N, AbstractEvalOp},
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}},
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}},
    ) where {Tg, Tv, N}
    # Resolve here so the fresh Ref tuple stays local to this frame (stack-elidable).
    policies, hints = _resolve_oneshot_search_nd(search, queries, hint, Val(N))
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)

    # Pool-backed per-axis cache — build phase + eval loop reuse h/inv_h.
    # `map` over the tuple dispatches per-element on concrete axis type;
    # `ntuple(d -> grids[d], ...)` would Union-box heterogeneous tuples.
    grids_c = map(g -> _cache_axis_pooled(pool, g), grids)

    # Build phase (done once)
    Tz = _promote_eltype(_coeff_op, Tg, Tv)
    n_partials = 1 << N
    partials = acquire!(pool, Tz, (n_partials, size(data)...))
    _compute_nd_partials_quadratic!(partials, grids_c, data, bcs)
    extraps_eff = map(_resolve_extrap, extraps_val, grids_c)
    # Validate + batch-level InBounds promotion (see cubic_nd_oneshot.jl): all-in-bounds axes get
    # `InBounds()`, eliding per-query wrap/clamp/fill branches.
    extraps_eff = _validate_nd_domain(grids_c, queries, extraps_eff)

    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids_c, extraps_eff, ops, first(data))
        if oob_val !== nothing
            output[k] = oob_val; continue
        end
        q_eval = _handle_all_extraps(query_k, grids_c, extraps_eff)
        indices, Ls, _ = _search_all_intervals(q_eval, grids_c, policies, hints, extraps_eff)
        _, inv_hs, dLs = _compute_all_local_params(q_eval, grids_c, indices, Ls)
        output[k] = _eval_nd_quad_cell(partials, indices, inv_hs, dLs, ops)
    end
    return output
end

# Function barrier — specializes on concrete `search` type.
function _quadratic_nd_batch_dispatch!(output, grids, data, queries, bcs, extraps, ops, search, hint)
    return _quadratic_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps, ops, search, hint)
end

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

"""
    quadratic_interp(grids, data, query; deriv=EvalValue(), coeffs=AutoCoeffs(), kwargs...)

One-shot ND quadratic interpolation at a single point.
Zero-allocation after warmup.

# Strategy selection (`coeffs`)
- `AutoCoeffs()` (default): shared `_resolve_coeffs_nd_oneshot` policy, same as cubic —
  scalar query → `OnTheFly()` (cell-local collapse, skips the `2^N` partial build),
  batch → `PreCompute()` (amortized).
- `PreCompute()`: build all nodal partials first — bit-identical to the persistent
  interpolant (`quadratic_interp(grids, data)(q)`); opt in when `===` parity matters.
- `OnTheFly()`: sequential 1D collapse; matches PreCompute to a few ULPs of
  FP-reordering noise (not bit-exact) for every user `bc`.
"""
function quadratic_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = Left(QuadraticFit()),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    # Scalar one-shot: raw grids — both PreCompute and OnTheFly accept them
    # (`_compute_all_local_params` floats the cell width). `Tg` value-matched (Int/OneTo grid +
    # Float32 data → Float32) so eval + witness `Tr` agree. Batch keeps eager-convert.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    _validate_nd_grids(grids, data)
    Tr = _promote_eltype(_interp_op, Tg, Tv, promote_type(typeof.(query)...))

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → BinarySearch/axis

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))

    # Central policy (same as cubic): scalar AutoCoeffs → OnTheFly. Bit-exact parity
    # with the persistent interpolant stays available via explicit coeffs=PreCompute().
    coeffs_resolved = _resolve_coeffs_nd_oneshot(coeffs, query, ntuple(_ -> QuadraticInterp(), Val(N)))
    if coeffs_resolved isa OnTheFly
        sample = @inbounds first(data)
        methods = map(bc_i -> QuadraticInterp(_to_quadratic_bc(bc_i, sample)), bcs)
        return _interp_nd_oneshot_onthefly(grids, data, query, methods, extraps_val, searches, ops, hint)::Tr
    end
    return _quadratic_interp_nd_oneshot(
        grids, data, query, bcs, extraps_val, searches, ops, hint
    )::Tr
end

"""
    quadratic_interp(grids, data, queries; deriv=EvalValue(), kwargs...)

One-shot ND quadratic interpolation at multiple points (batch).
Accepts any query format implementing the query protocol.
Only allocates the output vector.
"""
function quadratic_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = Left(QuadraticFit()),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _, Tg, _, _ = _nd_promote_grids(grids, data)
    Tq = _query_eltype(queries)
    Tr = _promote_eltype(_interp_op, Tg, Tv, Tq)
    output = Vector{Tr}(undef, _query_length(queries))
    quadratic_interp!(output, grids, data, queries; deriv, bc, extrap, search, coeffs, hint)
    return output
end

# ========================================
# IN-PLACE PUBLIC API (ND batch)
# ========================================

"""
    quadratic_interp!(output, grids, data, queries; deriv=EvalValue(), kwargs...)

In-place one-shot ND quadratic interpolation at multiple points (batch).
Accepts any query format implementing the query protocol.
Writes results into pre-allocated `output` vector.
"""
function quadratic_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = Left(QuadraticFit()),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _query_check_ndims(queries, Val(N))
    grids_typed, _, _, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _quadratic_nd_batch_dispatch!(output, grids_typed, data, queries, bcs, extraps_val, ops, search, hint)
end
