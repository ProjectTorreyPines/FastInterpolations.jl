# ========================================
# ConstantInterpolantND — One-Shot Evaluation
# ========================================
#
# Zero-allocation one-shot API and backends for ND constant interpolation.
# Interpolant construction is in constant_nd_interpolant.jl.

# ========================================
# ZERO-ALLOC ONE-SHOT IMPLEMENTATION
# ========================================
#
# Standalone evaluation functions that bypass Interpolant construction.
# Zero heap allocation for scalar queries after warmup.

"""
    _constant_interp_nd_oneshot(grids, data, query, bcs, extraps_val, side_vals, searches, hints=nothing)

Zero-allocation scalar one-shot ND constant evaluation.
Evaluates directly from grids + data without constructing a ConstantInterpolantND.
"""
function _constant_interp_nd_oneshot(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        side_vals::Tuple{Vararg{AbstractSide, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        hints = nothing
    ) where {Tg, Tv, N}
    # Surface-API axis resolution (mirrors `_linear_interp_nd_oneshot`):
    # `:exclusive` Vector/Range → `_ExclusivePeriodicAxis`; non-periodic
    # passthrough or cached float form. Wrapper search returns post-fold
    # `idx_R = 1` at seam so `data[..., idx_R, ...]` indexes raw data.
    grids_eff = map(_resolve_axis, grids, bcs)
    # NoExtrap domain check must precede FillExtrap short-circuit
    _validate_nd_domain(grids_eff, query, extraps_val)
    oob_result = _try_fill_oob(query, grids_eff, extraps_val, EvalValue(), @inbounds first(data))
    oob_result !== nothing && return oob_result

    extraps_eff = _resolve_extrap(extraps_val, bcs, grids_eff, data, Val(N))
    q_eval = _handle_all_extraps(query, grids_eff, extraps_eff)
    stencils, Ls, Rs = _search_all_intervals_stencil(q_eval, grids_eff, searches, hints)
    # 4-arg `_get_h(g, idx, xL, xR)` — cached path for `_CachedVector` (idx)
    # / `_CachedRange` (scalar field); raw `Vector` falls back to `xR - xL`.
    idxLs = map(first, stencils)
    hs = map(_get_h, grids_eff, idxLs, Ls, Rs)
    return _constant_nd_kernel(data, stencils, hs, side_vals, q_eval, Ls)
end

"""
    _constant_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps_val, side_vals, search, hint)

In-place batch one-shot ND constant evaluation.
"""
function _constant_interp_nd_oneshot_batch!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        side_vals::Tuple{Vararg{AbstractSide, N}},
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
    # Batch-level InBounds promotion: see cubic_nd_oneshot.jl / linear_nd_oneshot.jl
    # for the same pattern. Replaces `_validate_nd_domain` (throw via 1D
    # `_check_domain`'s `@boundscheck`) and shrinks `fill_dims` at compile time.
    extraps_eff = _check_domain_nd(grids_eff, queries, extraps_eff)
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids_eff, extraps_eff, EvalValue(), first(data))
        if oob_val !== nothing
            output[k] = oob_val; continue
        end
        q_eval = _handle_all_extraps(query_k, grids_eff, extraps_eff)
        stencils, Ls, Rs = _search_all_intervals_stencil(q_eval, grids_eff, policies, hints)
        idxLs = map(first, stencils)
        hs = map(_get_h, grids_eff, idxLs, Ls, Rs)
        output[k] = _constant_nd_kernel(data, stencils, hs, side_vals, q_eval, Ls)
    end
    return output
end

"""
    _constant_interp_nd_oneshot_batch(grids, data, queries, bcs, extraps_val, side_vals, search, hint)

Allocating wrapper: creates output vector, delegates to in-place batch.
"""
function _constant_interp_nd_oneshot_batch(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        side_vals::Tuple{Vararg{AbstractSide, N}},
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}},
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}},
    ) where {Tg, Tv, N}
    output = Vector{Tv}(undef, _query_length(queries))
    return _constant_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps_val, side_vals, search, hint)
end

# ========================================
# Derivative Check Helper
# ========================================

@inline _is_any_deriv(op::DerivOp) = !(op isa DerivOp{0})
@inline _is_any_deriv(ops::Tuple{Vararg{DerivOp}}) = any(op -> !(op isa DerivOp{0}), ops)

# Function barrier — specializes on concrete `search` type.
function _constant_nd_batch_dispatch!(output, grids, data, queries, bcs, extraps, sides, search, hint)
    return _constant_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps, sides, search, hint)
end
function _constant_nd_batch_dispatch(grids, data, queries, bcs, extraps, sides, search, hint)
    return _constant_interp_nd_oneshot_batch(grids, data, queries, bcs, extraps, sides, search, hint)
end

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

"""
    constant_interp(grids, data, query; kwargs...)

One-shot N-dimensional constant interpolation (scalar query).
Zero-allocation after warmup.
"""
function constant_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    # Any derivative of constant interpolation is zero
    if _is_any_deriv(deriv)
        return 0 * first(data)
    end

    grids_typed, _, _ = _nd_promote_grids_raw(grids, data)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → BinarySearch/axis

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    return _constant_interp_nd_oneshot(
        grids_typed, data, query, bcs, extraps_val, sides, searches, hint
    )::Tv
end

"""
    constant_interp(grids, data, queries; kwargs...)

One-shot N-dimensional constant interpolation (batch query).
Accepts any query format implementing the query protocol.
Only allocates the output vector.
"""
function constant_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    if _is_any_deriv(deriv)
        return zeros(Tv, _query_length(queries))
    end

    grids_typed, _, _ = _nd_promote_grids_raw(grids, data)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    sides = _resolve_side_nd(side, Val(N))

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    return _constant_nd_batch_dispatch(
        grids_typed, data, queries, bcs, extraps_val, sides, search, hint
    )::Vector{Tv}
end

# ========================================
# IN-PLACE PUBLIC API (ND batch)
# ========================================

"""
    constant_interp!(output, grids, data, queries; kwargs...)

In-place one-shot N-dimensional constant interpolation (batch query).
Accepts any query format implementing the query protocol.
Writes results into pre-allocated `output` vector.
"""
function constant_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _query_check_ndims(queries, Val(N))
    if _is_any_deriv(deriv)
        fill!(output, 0 * first(data))
        return output
    end

    grids_typed, _, _ = _nd_promote_grids_raw(grids, data)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    sides = _resolve_side_nd(side, Val(N))

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    return _constant_nd_batch_dispatch!(
        output, grids_typed, data, queries, bcs, extraps_val, sides, search, hint
    )
end
