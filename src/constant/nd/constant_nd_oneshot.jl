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
@inline function _constant_interp_nd_oneshot(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}},
    extraps_val::NTuple{N, Val},
    side_vals::NTuple{N, SideVal},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    spacings = _create_spacings_typed(grids)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
    return _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
end

"""
    _constant_interp_nd_oneshot_soa(grids, data, queries, extraps_val, side_vals, searches)

SoA batch one-shot ND constant evaluation.
Only allocates the output vector (return value).
"""
function _constant_interp_nd_oneshot_soa(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    extraps_val::NTuple{N, Val},
    side_vals::NTuple{N, SideVal},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end
    spacings = _create_spacings_typed(grids)
    output = Vector{Tv}(undef, n_queries)
    @inbounds for k in 1:n_queries
        query_k = ntuple(d -> queries[d][k], Val(N))
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        output[k] = _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
    end
    return output
end

"""
    _constant_interp_nd_oneshot_aos(grids, data, queries, extraps_val, side_vals, searches)

AoS batch one-shot ND constant evaluation.
Only allocates the output vector (return value).
"""
function _constant_interp_nd_oneshot_aos(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    extraps_val::NTuple{N, Val},
    side_vals::NTuple{N, SideVal},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries)
    spacings = _create_spacings_typed(grids)
    output = Vector{Tv}(undef, n_queries)
    @inbounds for k in 1:n_queries
        query_k = queries[k]
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        output[k] = _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
    end
    return output
end

# ========================================
# Derivative Check Helper
# ========================================

@inline _is_any_deriv(d::Int) = d != 0
@inline _is_any_deriv(::Val{T}) where {T} = any(!=(0), T)
@inline _is_any_deriv(d::NTuple{N, Int}) where {N} = any(!=(0), d)

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
    side::Union{Symbol, NTuple{N, Symbol}} = :nearest,
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    # Any derivative of constant interpolation is zero
    if _is_any_deriv(deriv)
        return zero(Tv)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))
    side_vals = _to_side_vals(sides)

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        return _constant_interp_nd_oneshot(
            grids_typed, data, query, extraps_val, side_vals, searches)::Tv
    end
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
    side::Union{Symbol, NTuple{N, Symbol}} = :nearest,
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    if _is_any_deriv(deriv)
        n_queries = length(queries[1])
        return zeros(Tv, n_queries)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))
    side_vals = _to_side_vals(sides)

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        return _constant_interp_nd_oneshot_soa(
            grids_typed, data, queries, extraps_val, side_vals, searches)::Vector{Tv}
    end
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
    side::Union{Symbol, NTuple{N, Symbol}} = :nearest,
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    if _is_any_deriv(deriv)
        n_queries = length(queries)
        return zeros(Tv, n_queries)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))
    side_vals = _to_side_vals(sides)

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        return _constant_interp_nd_oneshot_aos(
            grids_typed, data, queries, extraps_val, side_vals, searches)::Vector{Tv}
    end
end
