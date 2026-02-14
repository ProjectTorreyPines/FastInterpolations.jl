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
    _linear_interp_nd_oneshot(grids, data, query, extraps_val, searches, ops)

Zero-allocation scalar one-shot ND multilinear evaluation.
Evaluates directly from grids + data without constructing a LinearInterpolantND.
"""
@inline function _linear_interp_nd_oneshot(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    spacings = _create_spacings_typed(grids)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
    hs, αs = _compute_linear_params(q_eval, spacings, indices, Ls, Val(N))
    return _multilinear_sum(data, indices, hs, αs, ops, Val(N))
end

"""
    _linear_interp_nd_oneshot_soa!(output, grids, data, queries, extraps_val, searches, ops)

In-place SoA batch one-shot ND multilinear evaluation.
Writes results into `output`. No heap allocation beyond spacings.
"""
function _linear_interp_nd_oneshot_soa!(
    output::AbstractVector{Tv},
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end
    length(output) == n_queries || throw(DimensionMismatch(
        "output length ($(length(output))) must match query length ($n_queries)"))
    spacings = _create_spacings_typed(grids)
    @inbounds for k in 1:n_queries
        query_k = ntuple(d -> queries[d][k], Val(N))
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        hs, αs = _compute_linear_params(q_eval, spacings, indices, Ls, Val(N))
        output[k] = _multilinear_sum(data, indices, hs, αs, ops, Val(N))
    end
    return output
end

"""
    _linear_interp_nd_oneshot_soa(grids, data, queries, extraps_val, searches, ops)

Allocating wrapper: creates output vector, delegates to in-place `_soa!`.
"""
function _linear_interp_nd_oneshot_soa(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    output = Vector{Tv}(undef, length(queries[1]))
    return _linear_interp_nd_oneshot_soa!(output, grids, data, queries, extraps_val, searches, ops)
end

"""
    _linear_interp_nd_oneshot_aos!(output, grids, data, queries, extraps_val, searches, ops)

In-place AoS batch one-shot ND multilinear evaluation.
Writes results into `output`. No heap allocation beyond spacings.
"""
function _linear_interp_nd_oneshot_aos!(
    output::AbstractVector{Tv},
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries)
    length(output) == n_queries || throw(DimensionMismatch(
        "output length ($(length(output))) must match query length ($n_queries)"))
    spacings = _create_spacings_typed(grids)
    @inbounds for k in 1:n_queries
        query_k = queries[k]
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        hs, αs = _compute_linear_params(q_eval, spacings, indices, Ls, Val(N))
        output[k] = _multilinear_sum(data, indices, hs, αs, ops, Val(N))
    end
    return output
end

"""
    _linear_interp_nd_oneshot_aos(grids, data, queries, extraps_val, searches, ops)

Allocating wrapper: creates output vector, delegates to in-place `_aos!`.
"""
function _linear_interp_nd_oneshot_aos(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    output = Vector{Tv}(undef, length(queries))
    return _linear_interp_nd_oneshot_aos!(output, grids, data, queries, extraps_val, searches, ops)
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
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _linear_interp_nd_oneshot(grids_typed, data, query, extraps_val, searches, ops)::Tv
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _linear_interp_nd_oneshot(grids_typed, data, query, extraps_val, searches, ops)::Tv
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _linear_interp_nd_oneshot(grids_typed, data, query, extraps_val, searches, ops)::Tv
        end
    end
end

"""
    linear_interp(grids, data, queries::NTuple{N,AbstractVector}; kwargs...)

One-shot N-dimensional linear interpolation (batch SoA query).
Only allocates the output vector.
"""
function linear_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::NTuple{N, AbstractVector{<:Real}};
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _linear_interp_nd_oneshot_soa(grids_typed, data, queries, extraps_val, searches, ops)::Vector{Tv}
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _linear_interp_nd_oneshot_soa(grids_typed, data, queries, extraps_val, searches, ops)::Vector{Tv}
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _linear_interp_nd_oneshot_soa(grids_typed, data, queries, extraps_val, searches, ops)::Vector{Tv}
        end
    end
end

"""
    linear_interp(grids, data, queries::AbstractVector{<:NTuple}; kwargs...)

One-shot N-dimensional linear interpolation (batch AoS query).
Only allocates the output vector.
"""
function linear_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _linear_interp_nd_oneshot_aos(grids_typed, data, queries, extraps_val, searches, ops)::Vector{Tv}
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _linear_interp_nd_oneshot_aos(grids_typed, data, queries, extraps_val, searches, ops)::Vector{Tv}
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _linear_interp_nd_oneshot_aos(grids_typed, data, queries, extraps_val, searches, ops)::Vector{Tv}
        end
    end
end

# ========================================
# IN-PLACE PUBLIC API (ND batch)
# ========================================

"""
    linear_interp!(output, grids, data, queries::NTuple{N,AbstractVector}; kwargs...)

In-place one-shot N-dimensional linear interpolation (batch SoA query).
Writes results into pre-allocated `output` vector.
"""
function linear_interp!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::NTuple{N, AbstractVector{<:Real}};
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _linear_interp_nd_oneshot_soa!(output, grids_typed, data, queries, extraps_val, searches, ops)
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _linear_interp_nd_oneshot_soa!(output, grids_typed, data, queries, extraps_val, searches, ops)
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _linear_interp_nd_oneshot_soa!(output, grids_typed, data, queries, extraps_val, searches, ops)
        end
    end
end

"""
    linear_interp!(output, grids, data, queries::AbstractVector{<:NTuple}; kwargs...)

In-place one-shot N-dimensional linear interpolation (batch AoS query).
Writes results into pre-allocated `output` vector.
"""
function linear_interp!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _linear_interp_nd_oneshot_aos!(output, grids_typed, data, queries, extraps_val, searches, ops)
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _linear_interp_nd_oneshot_aos!(output, grids_typed, data, queries, extraps_val, searches, ops)
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _linear_interp_nd_oneshot_aos!(output, grids_typed, data, queries, extraps_val, searches, ops)
        end
    end
end
