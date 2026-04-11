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
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing
    ) where {Tg, Tv, N}
    # 0. NoExtrap domain check must precede FillExtrap short-circuit
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    # 1. Pool-allocate partials array (THE KEY: pool instead of heap)
    n_partials = 1 << N
    partials = acquire!(pool, Tv, (n_partials, size(data)...))

    # 2. Compute all partial derivatives in-place
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)

    # 3. Create spacings (ScalarSpacing for Range grids = zero alloc)
    spacings = _create_spacings_pooled(pool, grids)

    # 4. Eval pipeline (all standalone functions, no Interpolant needed)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)

    # 5. Tensor-product kernel evaluation
    return _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
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
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing
    ) where {Tg, Tv, N}
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps_val)

    # Build phase (done once)
    n_partials = 1 << N
    partials = acquire!(pool, Tv, (n_partials, size(data)...))
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)
    spacings = _create_spacings_pooled(pool, grids)

    # Eval loop
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids, extraps_val, ops, first(data))
        if oob_val !== nothing
            output[k] = oob_val; continue
        end
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)
        output[k] = _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

# Function barrier: forces Julia to runtime-dispatch on the concrete
# searches tuple type before entering the @with_pool boundary.
function _quadratic_nd_batch_dispatch!(output, grids, data, queries, bcs, extraps, searches, ops, hints)
    return _quadratic_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps, searches, ops, hints)
end

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

"""
    quadratic_interp(grids, data, query; deriv=EvalValue(), coeffs=AutoCoeffs(), kwargs...)

One-shot ND quadratic interpolation at a single point.
Zero-allocation after warmup.

# Strategy selection (`coeffs`)
- `AutoCoeffs()` (default): resolves to `PreCompute()` for both scalar and
  batch quadratic queries. This matches the interpolant constructor and AD
  rules **bit-exactly**, at the cost of always building the `2^N` partial
  array. Cubic `AutoCoeffs()` differs (it routes scalar queries to `OnTheFly`)
  because cubic OnTheFly↔PreCompute equivalence holds at `≈ rtol=1e-10` and
  AD seed agreement does not require strict equality there.
- `PreCompute()`: explicitly build all nodal partial derivatives first, then
  evaluate. The user `bc` is applied uniformly to every partial (pure and
  mixed), so the stored mixed partials match OnTheFly's composition formula.
- `OnTheFly()`: sequential 1D interpolation per fiber via `_collapse_dims`.
  Applies the user-specified `bc` uniformly at every 1D step.

`PreCompute()` and `OnTheFly()` are equivalent up to a few ULPs of FP
reordering noise for every user BC, but **not** bit-identical — that is why
quadratic `AutoCoeffs` defaults to `PreCompute`. This was not the case prior
to the mixed-partial BC consistency fix; see
`claudedocs/TODO/DONE/mixed_partial_bc_fix.md` for the history.
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
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)
    Tr = _output_eltype(Tv, Tg, typeof.(query)...)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → BinarySearch/axis

    extraps_val = _resolve_extrap_nd(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))

    # Keep AutoCoeffs on the specialized PreCompute path so scalar one-shot
    # calls match the interpolant constructor and AD rules.
    coeffs_resolved = coeffs isa AutoCoeffs ? PreCompute() : coeffs
    if coeffs_resolved isa OnTheFly
        sample = @inbounds first(data)
        methods = map(bc_i -> QuadraticInterp(_to_quadratic_bc(bc_i, sample)), bcs)
        return _interp_nd_oneshot_onthefly(grids_typed, data, query, methods, extraps_val, searches, ops, hint)::Tr
    end
    return _quadratic_interp_nd_oneshot(
        grids_typed, data, query, bcs, extraps_val, searches, ops, hint
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
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    Tq = _query_eltype(queries)
    Tr = _output_eltype(Tv, Tg, Tq)
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
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd_uniform(search, Val(N), queries, hint)

    extraps_val = _resolve_extrap_nd(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _quadratic_nd_batch_dispatch!(output, grids_typed, data, queries, bcs, extraps_val, searches, ops, hint)
end
