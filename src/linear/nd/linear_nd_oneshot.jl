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
    _linear_interp_nd_oneshot(grids, data, query, extraps_val, searches, ops, hints=nothing)

Zero-allocation scalar one-shot ND multilinear evaluation.
Evaluates directly from grids + data without constructing a LinearInterpolantND.
"""
@with_pool pool function _linear_interp_nd_oneshot(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing
    ) where {Tg, Tv, N}
    # NoExtrap domain check must precede FillExtrap short-circuit
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    spacings = _create_spacings_pooled(pool, grids)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
    hs, αs = _compute_linear_params(q_eval, spacings, indices, Ls, Val(N))
    return _multilinear_sum(data, indices, hs, αs, ops, Val(N))
end

"""
    _linear_interp_nd_oneshot_batch!(output, grids, data, queries, extraps_val, searches, ops, hints=nothing)

In-place batch one-shot ND multilinear evaluation.
Uses query protocol (`_query_length`, `_query_extract`) — works with any query format.
Writes results into `output`. No heap allocation beyond spacings.
"""
@with_pool pool function _linear_interp_nd_oneshot_batch!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        queries,
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing
    ) where {Tg, Tv, N}
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps_val)
    spacings = _create_spacings_pooled(pool, grids)
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids, extraps_val, ops, first(data))
        if oob_val !== nothing
            output[k] = oob_val; continue
        end
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
        hs, αs = _compute_linear_params(q_eval, spacings, indices, Ls, Val(N))
        output[k] = _multilinear_sum(data, indices, hs, αs, ops, Val(N))
    end
    return output
end

# Function barrier: forces Julia to runtime-dispatch on the concrete
# searches tuple type before entering the @with_pool boundary.
function _linear_nd_batch_dispatch!(output, grids, data, queries, extraps, searches, ops, hints)
    return _linear_interp_nd_oneshot_batch!(output, grids, data, queries, extraps, searches, ops, hints)
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
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    grids_typed, Tg, _, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)
    Tr = _output_eltype(Tv, Tg, typeof.(query)...)

    searches = _resolve_search_nd(search, Val(N), query)  # scalar: type-based (no monotonicity check)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _linear_interp_nd_oneshot(grids_typed, data, query, extraps_val, searches, ops, hint)::Tr
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
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _, Tg, _, _ = _nd_promote_grids(grids, data)
    Tq = _query_eltype(queries)
    Tr = _output_eltype(Tv, Tg, Tq)
    output = Vector{Tr}(undef, _query_length(queries))
    linear_interp!(output, grids, data, queries; extrap, search, deriv, hint)
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
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _query_check_ndims(queries, Val(N))
    grids_typed, _, _, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)

    searches = _resolve_search_nd_uniform(search, Val(N), queries, hint)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _linear_nd_batch_dispatch!(output, grids_typed, data, queries, extraps_val, searches, ops, hint)
end
