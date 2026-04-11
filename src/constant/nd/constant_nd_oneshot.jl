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
    _constant_interp_nd_oneshot(grids, data, query, extraps_val, side_vals, searches, hints=nothing)

Zero-allocation scalar one-shot ND constant evaluation.
Evaluates directly from grids + data without constructing a ConstantInterpolantND.
"""
@with_pool pool function _constant_interp_nd_oneshot(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        side_vals::Tuple{Vararg{AbstractSide, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        hints = nothing
    ) where {Tg, Tv, N}
    # NoExtrap domain check must precede FillExtrap short-circuit
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, EvalValue(), @inbounds first(data))
    oob_result !== nothing && return oob_result

    spacings = _create_spacings_pooled(pool, grids)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
    return _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
end

"""
    _constant_interp_nd_oneshot_batch!(output, grids, data, queries, extraps_val, side_vals, searches, hints=nothing)

In-place batch one-shot ND constant evaluation.
Uses query protocol (`_query_length`, `_query_extract`) — works with any query format.
Writes results into `output`. No heap allocation beyond spacings.
"""
@with_pool pool function _constant_interp_nd_oneshot_batch!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        queries,
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        side_vals::Tuple{Vararg{AbstractSide, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        hints = nothing
    ) where {Tg, Tv, N}
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps_val)
    spacings = _create_spacings_pooled(pool, grids)
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids, extraps_val, EvalValue(), first(data))
        if oob_val !== nothing
            output[k] = oob_val; continue
        end
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
        output[k] = _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
    end
    return output
end

"""
    _constant_interp_nd_oneshot_batch(grids, data, queries, extraps_val, side_vals, searches, hints=nothing)

Allocating wrapper: creates output vector, delegates to in-place batch.
"""
function _constant_interp_nd_oneshot_batch(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        queries,
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        side_vals::Tuple{Vararg{AbstractSide, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        hints = nothing
    ) where {Tg, Tv, N}
    output = Vector{Tv}(undef, _query_length(queries))
    return _constant_interp_nd_oneshot_batch!(output, grids, data, queries, extraps_val, side_vals, searches, hints)
end

# ========================================
# Derivative Check Helper
# ========================================

@inline _is_any_deriv(op::DerivOp) = !(op isa DerivOp{0})
@inline _is_any_deriv(ops::Tuple{Vararg{DerivOp}}) = any(op -> !(op isa DerivOp{0}), ops)

# Function barrier: forces Julia to runtime-dispatch on the concrete
# searches tuple type before entering the @with_pool boundary.
function _constant_nd_batch_dispatch!(output, grids, data, queries, extraps, sides, searches, hints)
    return _constant_interp_nd_oneshot_batch!(output, grids, data, queries, extraps, sides, searches, hints)
end
function _constant_nd_batch_dispatch(grids, data, queries, extraps, sides, searches, hints)
    return _constant_interp_nd_oneshot_batch(grids, data, queries, extraps, sides, searches, hints)
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

    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → BinarySearch/axis

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    return _constant_interp_nd_oneshot(
        grids_typed, data, query, extraps_val, sides, searches, hint
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
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    if _is_any_deriv(deriv)
        return zeros(Tv, _query_length(queries))
    end

    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd_uniform(search, Val(N), queries, hint)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    return _constant_nd_batch_dispatch(
        grids_typed, data, queries, extraps_val, sides, searches, hint
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

    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd_uniform(search, Val(N), queries, hint)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    return _constant_nd_batch_dispatch!(
        output, grids_typed, data, queries, extraps_val, sides, searches, hint
    )
end
