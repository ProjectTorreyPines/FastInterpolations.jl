# ========================================
# LinearInterpolantND — One-Shot Evaluation
# ========================================
#
# Zero-allocation one-shot API and backends for ND linear interpolation.
# Interpolant construction is in linear_nd_interpolant.jl.

# ========================================
# ZERO-ALLOC ONE-SHOT IMPLEMENTATION
# ========================================
#
# Standalone evaluation functions that bypass Interpolant construction.
# All pipeline functions (extrap, search, params, kernel) are standalone.
# Zero heap allocation for scalar queries after warmup.

"""
    _linear_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops, hints=nothing)

Zero-allocation scalar one-shot ND multilinear evaluation.
Evaluates directly from grids + data — no pool, no data extension even for
exclusive periodic axes. Per-axis bc is projected into typed `WrapExtrap` via
`_resolve_extrap` (validation + materialization); BC-aware `Searcher` returns `(idx_L=n, idx_R=1)`
at periodic seam cells so the kernel reads wrapped corners directly from the
original `data`. Expect ~1× parity with persistent ND interpolant for periodic
exclusive (no per-query N-dim data copy).
"""
function _linear_interp_nd_oneshot(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing
    ) where {Tv, N}
    # Per-axis BC-aware axis resolution: `:exclusive` axes (Vector or Range) →
    # `_ExclusivePeriodicAxis` carrying the precomputed virtual endpoint and
    # period; non-periodic → passthrough or cached float form. After this,
    # `(first(g), last(g))` is the canonical wrap domain on every axis and the
    # wrapper's specialized `search_interval` returns post-fold `idx_R` (= 1 at
    # seam) so kernels read raw `data[..., idx_R, ...]` directly. With BC info
    # encoded in the axis type, the searcher carries `NoBC()` — the wrapper's
    # dispatch handles seam regardless.
    # Value-matched grid float (Int grid + Float32 data → Float32) — eval matches the `Tr` witness.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    grids_eff = _resolve_axes(grids, bcs, Tg)  # @generated static-Tg unroll (no Type-captured closure)
    # Bare GridIdx(k).val is NaN → resolve to the grid coordinate for the value kernel (search still uses .idx).
    query = map(_resolve_grididx, query, grids_eff)
    # Validate (NoExtrap throw must precede the FillExtrap short-circuit) AND promote per axis:
    # an in-domain NoExtrap axis becomes InBounds for the search (lean), mirroring the persistent
    # path. InBounds passes through `_try_fill_oob` / `_resolve_extrap` / `_handle_all_extraps`
    # (all no-op on it) and reaches the extrap-aware `_search_all_intervals_stencil` below.
    extraps_val = _validate_nd_domain(grids_eff, query, extraps_val)
    oob_result = _try_fill_oob(query, grids_eff, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    extraps_eff = _resolve_extrap(extraps_val, bcs, grids_eff, data, Val(N))
    q_eval = _handle_all_extraps(query, grids_eff, extraps_eff)
    stencils, Ls, Rs = _search_all_intervals_stencil(q_eval, grids_eff, searches, hints, extraps_eff)
    # Width-first `_get_inv_h(Tg, g, idx, xL, xR)` — `_CachedVector`/`_CachedRange`
    # use cached fields; a raw `Vector` spans `xR - xL` in its own eltype and divides
    # at the value-matched `Tg` (span-first: an Int axis would otherwise mint Float64
    # via `inv(Int)`). α derives from the typed inv_h — query-blood promotion preserved
    # (Dual query ⇒ Dual α) and the seam cell shares the deriv path's denominator.
    idxLs = map(first, stencils)
    inv_hs = _typed_inv_hs(grids_eff, idxLs, Ls, Rs, Tg)
    αs = map(_alpha_of, q_eval, Ls, inv_hs)
    return _multilinear_sum(data, stencils, inv_hs, αs, ops, Val(N))
end

"""
    _linear_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps_val, ops, search, hint)

In-place batch one-shot ND multilinear evaluation.
"""
function _linear_interp_nd_oneshot_batch!(
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
    grids_eff = map(_resolve_axis, grids, bcs)
    extraps_eff = _resolve_extrap(extraps_val, bcs, grids_eff, data, Val(N))
    # Validate + batch-level InBounds promotion: throws on an OOB NoExtrap axis and returns
    # per-axis `InBounds()` when all queries on an axis are in-bounds, eliding the per-query
    # wrap/clamp/fill via `_handle_axis_extrap(::InBounds)`.
    extraps_eff = _validate_nd_domain(grids_eff, queries, extraps_eff)
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids_eff, extraps_eff, ops, first(data))
        if oob_val !== nothing
            output[k] = oob_val; continue
        end
        q_eval = _handle_all_extraps(query_k, grids_eff, extraps_eff)
        stencils, Ls, Rs = _search_all_intervals_stencil(q_eval, grids_eff, policies, hints, extraps_eff)
        αs = map(_alpha_of, q_eval, Ls, Rs, grids_eff)
        idxLs = map(first, stencils)
        inv_hs = map(_get_inv_h, grids_eff, idxLs, Ls, Rs)
        output[k] = _multilinear_sum(data, stencils, inv_hs, αs, ops, Val(N))
    end
    return output
end

# Function barrier — specializes on concrete `search` type.
function _linear_nd_batch_dispatch!(output, grids, data, queries, bcs, extraps_val, ops, search, hint)
    return _linear_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps_val, ops, search, hint)
end

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

"""
    linear_interp(grids, data, query; kwargs...)

One-shot N-dimensional linear interpolation (scalar query).
Zero-allocation after warmup.
"""
function linear_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    # Scalar one-shot: raw grids shaped per axis by `_resolve_axis(…, Tg)` at the value-matched
    # float (Int grid + Float32 data → Float32) — no eager copy, witness `Tr` matches the eval.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    _validate_nd_grids(grids, data)
    Tr = _promote_eltype(_interp_op, Tg, Tv, promote_type(typeof.(query)...))

    searches = _resolve_search_nd(search, Val(N), query)  # scalar: type-based (no monotonicity check)

    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _linear_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops, hint)::Tr
end

"""
    linear_interp(grids, data, queries; kwargs...)

One-shot N-dimensional linear interpolation (batch query).
Accepts any query format implementing the query protocol.
Only allocates the output vector.
"""
function linear_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _, Tg, _, _ = _nd_promote_grids(grids, data)
    Tq = _query_eltype(queries)
    Tr = _promote_eltype(_interp_op, Tg, Tv, Tq)
    output = Vector{Tr}(undef, _query_length(queries))
    linear_interp!(output, grids, data, queries; bc, extrap, search, deriv, hint)
    return output
end

# ========================================
# IN-PLACE PUBLIC API (ND batch)
# ========================================

"""
    linear_interp!(output, grids, data, queries; kwargs...)

In-place one-shot N-dimensional linear interpolation (batch query).
Accepts any query format implementing the query protocol.
Writes results into pre-allocated `output` vector.
"""
function linear_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _query_check_ndims(queries, Val(N))
    grids_typed, _, _, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _linear_nd_batch_dispatch!(output, grids_typed, data, queries, bcs, extraps_val, ops, search, hint)
end
