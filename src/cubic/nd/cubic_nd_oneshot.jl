# ========================================
# ND Cubic Interpolation — One-Shot Evaluation
# ========================================
#
# Zero-allocation one-shot API and pool-based backends for ND cubic interpolation.
# Interpolant construction is in cubic_nd_interpolant.jl.

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

"""
    cubic_interp(grids, data, query; deriv=0, kwargs...)

One-shot ND cubic interpolation at a single point.
Zero-allocation after warmup: uses pool-based partials instead of constructing an Interpolant.

# Keywords
- `deriv`: `Int` (0-3) or `Val((d1,d2,...))` for mixed partials
- `bc`, `extrap`, `search`, `coeffs`: Same as the Interpolant constructor form
"""
function cubic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=NaturalBC(),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary(),
    coeffs::AbstractCoeffStrategy=PreCompute()
) where {Tv, N}
    # Type promotion + validation (same as constructor path)
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Validate periodic+extrap compatibility (once, before dispatch)
    _check_periodic_extrap(bcs, extraps, Val(N))

    # Dispatch extrap → concrete Val tuple, then deriv → concrete ops
    # Both dispatches happen BEFORE entering @with_pool for type stability
    # Type assertion (::Tv) prevents boxing from multi-branch return type inference failure
    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _cubic_interp_nd_oneshot(grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _cubic_interp_nd_oneshot(grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _cubic_interp_nd_oneshot(grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
        end
    end
end

"""
    cubic_interp(grids, data, queries::NTuple{N,AbstractVector}; deriv=0, kwargs...)

One-shot ND cubic interpolation at multiple points (SoA batch).
Zero-allocation for workspace after warmup; output vector is heap-allocated.
"""
function cubic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=NaturalBC(),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary(),
    coeffs::AbstractCoeffStrategy=PreCompute()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    _check_periodic_extrap(bcs, extraps, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _cubic_interp_nd_oneshot_soa(grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _cubic_interp_nd_oneshot_soa(grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _cubic_interp_nd_oneshot_soa(grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        end
    end
end

# ========================================
# POOL-BASED ND ONE-SHOT IMPLEMENTATION
# ========================================
#
# Zero-allocation ND one-shot evaluation using pool-based partials.
# Bypasses Interpolant construction entirely — computes partials in a pool buffer,
# evaluates at the query point(s), then releases all buffers on scope exit.
#
# All eval pipeline functions are standalone:
#   _handle_all_extraps, _search_all_intervals, _compute_all_local_params, _eval_nd_cell

"""
    _cubic_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops)

Pool-based scalar one-shot ND cubic Hermite evaluation.
Computes 2^N partial derivatives in a pool buffer and evaluates at a single point.
Zero-allocation after warmup (pool reuse).

`extraps_val` must be a pre-resolved tuple of `Val` types (e.g., `(Val(:none), Val(:none))`),
computed via `@_dispatch_extrap_nd` in the API layer for type stability.
"""
@with_pool pool function _cubic_interp_nd_oneshot(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}},
    bcs::NTuple{N, AbstractBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    # 1. Extend exclusive periodic axes (pool-based, zero heap alloc)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs)

    # 2. Pool-allocate partials array (THE KEY: pool instead of heap)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data_p)...))

    # 3. Compute all partial derivatives in-place
    #    (internally uses autocached 1D caches + nested @with_pool for temp buffers)
    _compute_nd_partials!(partials, grids_p, data_p, bcs_p)

    # 4. Create spacings (ScalarSpacing for Range grids = zero alloc)
    spacings = _create_spacings_typed(grids_p)

    # 5. Eval pipeline (all standalone functions, no Interpolant needed)
    q_evals = _handle_all_extraps(query, grids_p, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_evals, grids_p, spacings, searches)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, spacings, indices, Ls)

    # 6. Tensor-product kernel evaluation
    return _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
end

"""
    _cubic_interp_nd_oneshot_soa(grids, data, queries_soa, bcs, extraps_val, searches, ops)

Pool-based SoA batch one-shot ND cubic Hermite evaluation.
Computes partials ONCE, then evaluates at all query points.
Output vector is heap-allocated (return value).

`extraps_val` must be a pre-resolved tuple of `Val` types.
"""
@with_pool pool function _cubic_interp_nd_oneshot_soa(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    bcs::NTuple{N, AbstractBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    # Validate query lengths
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end

    # Build phase (same as scalar, done once)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data_p)...))
    _compute_nd_partials!(partials, grids_p, data_p, bcs_p)
    spacings = _create_spacings_typed(grids_p)

    # Allocate output (heap — this IS the return value)
    output = Vector{Tv}(undef, n_queries)

    # Eval loop: search + kernel per query point
    @inbounds for k in 1:n_queries
        query_k = ntuple(d -> queries[d][k], Val(N))
        q_evals = _handle_all_extraps(query_k, grids_p, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_evals, grids_p, spacings, searches)
        hs, inv_hs, dLs = _compute_all_local_params(q_evals, spacings, indices, Ls)
        output[k] = _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end
