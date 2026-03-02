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
    hints=nothing
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
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)

    # 5. Tensor-product kernel evaluation
    return _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
end

"""
    _quadratic_interp_nd_oneshot_soa!(output, grids, data, queries, bcs, extraps_val, searches, ops, hints=nothing)

Pool-based in-place SoA batch one-shot ND quadratic evaluation.
Computes partials ONCE, then evaluates at all query points into `output`.
"""
@with_pool pool function _quadratic_interp_nd_oneshot_soa!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    bcs::NTuple{N, AbstractBC},
    extraps_val::Tuple{Vararg{AbstractExtrap, N}},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp},
    hints=nothing
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
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)
        output[k] = _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

"""
    _quadratic_interp_nd_oneshot_aos!(output, grids, data, queries, bcs, extraps_val, searches, ops, hints=nothing)

Pool-based in-place AoS batch one-shot ND quadratic evaluation.
Computes partials ONCE, then evaluates at all query points into `output`.
"""
@with_pool pool function _quadratic_interp_nd_oneshot_aos!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    bcs::NTuple{N, AbstractBC},
    extraps_val::Tuple{Vararg{AbstractExtrap, N}},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp},
    hints=nothing
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
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)
        output[k] = _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

# Function barrier for SoA paths: forces Julia to runtime-dispatch on the concrete
# searches tuple type, resolving per-element Union{BinarySearch,LinearBinarySearch} before
# entering the @with_pool boundary. NOT @inline — specialization requires real call.
function _quadratic_nd_soa_dispatch!(output, grids, data, queries, bcs, extraps, searches, ops, hints)
    _quadratic_interp_nd_oneshot_soa!(output, grids, data, queries, bcs, extraps, searches, ops, hints)
end

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

"""
    quadratic_interp(grids, data, query; deriv=EvalValue(), kwargs...)

One-shot ND quadratic interpolation at a single point.
Zero-allocation after warmup.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{AbstractExtrap, NTuple{N,AbstractExtrap}}=NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=AutoSearch(),
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)
    Tr = promote_type(Tv, Tg, typeof.(query)...)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → BinarySearch/axis

    extraps_val = _resolve_extrap_nd(extrap, bcs, Val(N))
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _quadratic_interp_nd_oneshot(
        grids_typed, data, query, bcs, extraps_val, searches, ops, hint)::Tr
end

"""
    quadratic_interp(grids, data, queries::NTuple{N,AbstractVector}; deriv=EvalValue(), kwargs...)

One-shot ND quadratic interpolation at multiple points (batch SoA).
Only allocates the output vector.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{AbstractExtrap, NTuple{N,AbstractExtrap}}=NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=AutoSearch(),
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    Tr = promote_type(Tv, Tg, _promote_grid_eltype(queries))
    output = Vector{Tr}(undef, length(queries[1]))
    quadratic_interp!(output, grids, data, queries; deriv, bc, extrap, search, hint)
    return output
end

"""
    quadratic_interp(grids, data, queries::AbstractVector{<:NTuple}; deriv=EvalValue(), kwargs...)

One-shot ND quadratic interpolation at multiple points (batch AoS).
Only allocates the output vector.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{AbstractExtrap, NTuple{N,AbstractExtrap}}=NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=AutoSearch(),
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    Tr = promote_type(Tv, Tg)
    output = Vector{Tr}(undef, length(queries))
    quadratic_interp!(output, grids, data, queries; deriv, bc, extrap, search, hint)
    return output
end

# ========================================
# IN-PLACE PUBLIC API (ND batch)
# ========================================

"""
    quadratic_interp!(output, grids, data, queries::NTuple{N,AbstractVector}; deriv=EvalValue(), kwargs...)

In-place one-shot ND quadratic interpolation at multiple points (SoA batch).
Writes results into pre-allocated `output` vector.
"""
function quadratic_interp!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{AbstractExtrap, NTuple{N,AbstractExtrap}}=NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=AutoSearch(),
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd_uniform(search, Val(N), queries, hint)  # all-or-nothing adaptive for zero-alloc

    extraps_val = _resolve_extrap_nd(extrap, bcs, Val(N))
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _quadratic_nd_soa_dispatch!(output, grids_typed, data, queries, bcs, extraps_val, searches, ops, hint)
end

"""
    quadratic_interp!(output, grids, data, queries::AbstractVector{<:NTuple}; deriv=EvalValue(), kwargs...)

In-place one-shot ND quadratic interpolation at multiple points (AoS batch).
Writes results into pre-allocated `output` vector.
"""
function quadratic_interp!(
    output::AbstractVector,
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{AbstractExtrap, NTuple{N,AbstractExtrap}}=NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=AutoSearch(),
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N), queries)  # AoS: type-based (no per-axis SoA check)

    extraps_val = _resolve_extrap_nd(extrap, bcs, Val(N))
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _quadratic_interp_nd_oneshot_aos!(
        output, grids_typed, data, queries, bcs, extraps_val, searches, ops, hint)
end
