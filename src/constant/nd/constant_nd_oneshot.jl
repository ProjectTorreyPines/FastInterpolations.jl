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
    _constant_interp_nd_oneshot(grids, data, query, extraps_val, side_vals, searches)

Zero-allocation scalar one-shot ND constant evaluation.
Evaluates directly from grids + data without constructing a ConstantInterpolantND.
"""
@with_pool pool function _constant_interp_nd_oneshot(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}},
    extraps_val::Tuple{Vararg{AbstractExtrap, N}},
    side_vals::Tuple{Vararg{AbstractSide, N}},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    spacings = _create_spacings_pooled(pool, grids)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
    return _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
end

"""
    _constant_interp_nd_oneshot_soa!(output, grids, data, queries, extraps_val, side_vals, searches)

In-place SoA batch one-shot ND constant evaluation.
Writes results into `output`. No heap allocation beyond spacings.
"""
@with_pool pool function _constant_interp_nd_oneshot_soa!(
    output::AbstractVector{Tv},
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    extraps_val::Tuple{Vararg{AbstractExtrap, N}},
    side_vals::Tuple{Vararg{AbstractSide, N}},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end
    length(output) == n_queries || throw(DimensionMismatch(
        "output length ($(length(output))) must match query length ($n_queries)"))
    spacings = _create_spacings_pooled(pool, grids)
    @inbounds for k in 1:n_queries
        query_k = ntuple(d -> queries[d][k], Val(N))
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        output[k] = _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
    end
    return output
end

"""
    _constant_interp_nd_oneshot_soa(grids, data, queries, extraps_val, side_vals, searches)

Allocating wrapper: creates output vector, delegates to in-place `_soa!`.
"""
function _constant_interp_nd_oneshot_soa(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    extraps_val::Tuple{Vararg{AbstractExtrap, N}},
    side_vals::Tuple{Vararg{AbstractSide, N}},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    output = Vector{Tv}(undef, length(queries[1]))
    return _constant_interp_nd_oneshot_soa!(output, grids, data, queries, extraps_val, side_vals, searches)
end

"""
    _constant_interp_nd_oneshot_aos!(output, grids, data, queries, extraps_val, side_vals, searches)

In-place AoS batch one-shot ND constant evaluation.
Writes results into `output`. No heap allocation beyond spacings.
"""
@with_pool pool function _constant_interp_nd_oneshot_aos!(
    output::AbstractVector{Tv},
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    extraps_val::Tuple{Vararg{AbstractExtrap, N}},
    side_vals::Tuple{Vararg{AbstractSide, N}},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries)
    length(output) == n_queries || throw(DimensionMismatch(
        "output length ($(length(output))) must match query length ($n_queries)"))
    spacings = _create_spacings_pooled(pool, grids)
    @inbounds for k in 1:n_queries
        query_k = queries[k]
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        output[k] = _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
    end
    return output
end

"""
    _constant_interp_nd_oneshot_aos(grids, data, queries, extraps_val, side_vals, searches)

Allocating wrapper: creates output vector, delegates to in-place `_aos!`.
"""
function _constant_interp_nd_oneshot_aos(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    extraps_val::Tuple{Vararg{AbstractExtrap, N}},
    side_vals::Tuple{Vararg{AbstractSide, N}},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    output = Vector{Tv}(undef, length(queries))
    return _constant_interp_nd_oneshot_aos!(output, grids, data, queries, extraps_val, side_vals, searches)
end

# ========================================
# Derivative Check Helper
# ========================================

@inline _is_any_deriv(op::DerivOp) = !(op isa DerivOp{0})
@inline _is_any_deriv(ops::Tuple{Vararg{DerivOp}}) = any(op -> !(op isa DerivOp{0}), ops)

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
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
) where {Tv, N}
    # Any derivative of constant interpolation is zero
    if _is_any_deriv(deriv)
        return zero(Tv)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N), first(query))

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N))
    return _constant_interp_nd_oneshot(
        grids_typed, data, query, extraps_val, sides, searches)::Tv
end

"""
    constant_interp(grids, data, queries::NTuple{N,AbstractVector}; kwargs...)

One-shot N-dimensional constant interpolation (batch SoA query).
Only allocates the output vector.
"""
function constant_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::NTuple{N, AbstractVector{<:Real}};
    side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
    extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
) where {Tv, N}
    if _is_any_deriv(deriv)
        n_queries = length(queries[1])
        return zeros(Tv, n_queries)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N), first(queries))

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N))
    return _constant_interp_nd_oneshot_soa(
        grids_typed, data, queries, extraps_val, sides, searches)::Vector{Tv}
end

"""
    constant_interp(grids, data, queries::AbstractVector{<:NTuple}; kwargs...)

One-shot N-dimensional constant interpolation (batch AoS query).
Only allocates the output vector.
"""
function constant_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
    extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
) where {Tv, N}
    if _is_any_deriv(deriv)
        n_queries = length(queries)
        return zeros(Tv, n_queries)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N), queries)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N))
    return _constant_interp_nd_oneshot_aos(
        grids_typed, data, queries, extraps_val, sides, searches)::Vector{Tv}
end

# ========================================
# IN-PLACE PUBLIC API (ND batch)
# ========================================

"""
    constant_interp!(output, grids, data, queries::NTuple{N,AbstractVector}; kwargs...)

In-place one-shot N-dimensional constant interpolation (batch SoA query).
Writes results into pre-allocated `output` vector.
"""
function constant_interp!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::NTuple{N, AbstractVector{<:Real}};
    side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
    extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
) where {Tv, N}
    if _is_any_deriv(deriv)
        fill!(output, zero(Tv))
        return output
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N), first(queries))

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N))
    return _constant_interp_nd_oneshot_soa!(
        output, grids_typed, data, queries, extraps_val, sides, searches)
end

"""
    constant_interp!(output, grids, data, queries::AbstractVector{<:NTuple}; kwargs...)

In-place one-shot N-dimensional constant interpolation (batch AoS query).
Writes results into pre-allocated `output` vector.
"""
function constant_interp!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
    extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
) where {Tv, N}
    if _is_any_deriv(deriv)
        fill!(output, zero(Tv))
        return output
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N), queries)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N))
    return _constant_interp_nd_oneshot_aos!(
        output, grids_typed, data, queries, extraps_val, sides, searches)
end
