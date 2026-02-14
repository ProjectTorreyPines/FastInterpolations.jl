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
    _quadratic_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops)

Pool-based scalar one-shot ND quadratic evaluation.
Computes 2^N partial derivatives in a pool buffer and evaluates at a single point.
Zero-allocation after warmup (pool reuse).
"""
@with_pool pool function _quadratic_interp_nd_oneshot(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}},
    bcs::NTuple{N, QuadraticBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    # 1. Pool-allocate partials array (THE KEY: pool instead of heap)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data)...))

    # 2. Compute all partial derivatives in-place
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)

    # 3. Create spacings (ScalarSpacing for Range grids = zero alloc)
    spacings = _create_spacings_pooled(pool, grids)

    # 4. Eval pipeline (all standalone functions, no Interpolant needed)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)

    # 5. Tensor-product kernel evaluation
    return _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
end

"""
    _quadratic_interp_nd_oneshot_soa!(output, grids, data, queries, bcs, extraps_val, searches, ops)

Pool-based in-place SoA batch one-shot ND quadratic evaluation.
Computes partials ONCE, then evaluates at all query points into `output`.
"""
@with_pool pool function _quadratic_interp_nd_oneshot_soa!(
    output::AbstractVector{Tv},
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    bcs::NTuple{N, QuadraticBC},
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

    # Build phase (done once)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data)...))
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)
    spacings = _create_spacings_pooled(pool, grids)

    # Eval loop
    @inbounds for k in 1:n_queries
        query_k = ntuple(d -> queries[d][k], Val(N))
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)
        output[k] = _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

"""
    _quadratic_interp_nd_oneshot_soa(grids, data, queries, bcs, extraps_val, searches, ops)

Allocating wrapper: creates output vector, delegates to in-place `_soa!`.
"""
function _quadratic_interp_nd_oneshot_soa(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    bcs::NTuple{N, QuadraticBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    output = Vector{Tv}(undef, length(queries[1]))
    return _quadratic_interp_nd_oneshot_soa!(output, grids, data, queries, bcs, extraps_val, searches, ops)
end

"""
    _quadratic_interp_nd_oneshot_aos!(output, grids, data, queries, bcs, extraps_val, searches, ops)

Pool-based in-place AoS batch one-shot ND quadratic evaluation.
Computes partials ONCE, then evaluates at all query points into `output`.
"""
@with_pool pool function _quadratic_interp_nd_oneshot_aos!(
    output::AbstractVector{Tv},
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    bcs::NTuple{N, QuadraticBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries)
    length(output) == n_queries || throw(DimensionMismatch(
        "output length ($(length(output))) must match query length ($n_queries)"))

    # Build phase (done once)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data)...))
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)
    spacings = _create_spacings_pooled(pool, grids)

    # Eval loop
    @inbounds for k in 1:n_queries
        query_k = queries[k]
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)
        output[k] = _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

"""
    _quadratic_interp_nd_oneshot_aos(grids, data, queries, bcs, extraps_val, searches, ops)

Allocating wrapper: creates output vector, delegates to in-place `_aos!`.
"""
function _quadratic_interp_nd_oneshot_aos(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    bcs::NTuple{N, QuadraticBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    output = Vector{Tv}(undef, length(queries))
    return _quadratic_interp_nd_oneshot_aos!(output, grids, data, queries, bcs, extraps_val, searches, ops)
end

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

"""
    quadratic_interp(grids, data, query; deriv=0, kwargs...)

One-shot ND quadratic interpolation at a single point.
Zero-allocation after warmup.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _quadratic_interp_nd_oneshot(
                    grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _quadratic_interp_nd_oneshot(
                grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _quadratic_interp_nd_oneshot(
                grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
        end
    end
end

"""
    quadratic_interp(grids, data, queries::NTuple{N,AbstractVector}; deriv=0, kwargs...)

One-shot ND quadratic interpolation at multiple points (batch SoA).
Only allocates the output vector.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _quadratic_interp_nd_oneshot_soa(
                    grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _quadratic_interp_nd_oneshot_soa(
                grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _quadratic_interp_nd_oneshot_soa(
                grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        end
    end
end

"""
    quadratic_interp(grids, data, queries::AbstractVector{<:NTuple}; deriv=0, kwargs...)

One-shot ND quadratic interpolation at multiple points (batch AoS).
Only allocates the output vector.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _quadratic_interp_nd_oneshot_aos(
                    grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _quadratic_interp_nd_oneshot_aos(
                grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _quadratic_interp_nd_oneshot_aos(
                grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        end
    end
end

# ========================================
# IN-PLACE PUBLIC API (ND batch)
# ========================================

"""
    quadratic_interp!(output, grids, data, queries::NTuple{N,AbstractVector}; deriv=0, kwargs...)

In-place one-shot ND quadratic interpolation at multiple points (SoA batch).
Writes results into pre-allocated `output` vector.
"""
function quadratic_interp!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _quadratic_interp_nd_oneshot_soa!(
                    output, grids_typed, data, queries, bcs, extraps_val, searches, ops)
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _quadratic_interp_nd_oneshot_soa!(
                output, grids_typed, data, queries, bcs, extraps_val, searches, ops)
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _quadratic_interp_nd_oneshot_soa!(
                output, grids_typed, data, queries, bcs, extraps_val, searches, ops)
        end
    end
end

"""
    quadratic_interp!(output, grids, data, queries::AbstractVector{<:NTuple}; deriv=0, kwargs...)

In-place one-shot ND quadratic interpolation at multiple points (AoS batch).
Writes results into pre-allocated `output` vector.
"""
function quadratic_interp!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _quadratic_interp_nd_oneshot_aos!(
                    output, grids_typed, data, queries, bcs, extraps_val, searches, ops)
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _quadratic_interp_nd_oneshot_aos!(
                output, grids_typed, data, queries, bcs, extraps_val, searches, ops)
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _quadratic_interp_nd_oneshot_aos!(
                output, grids_typed, data, queries, bcs, extraps_val, searches, ops)
        end
    end
end
