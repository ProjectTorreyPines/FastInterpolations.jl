# ========================================
# ConstantInterpolantND Evaluation
# ========================================
#
# Evaluation logic for N-dimensional constant interpolation.
# Supports scalar, vector, and batch (SoA/AoS) queries.

# ========================================
# Callable Interface
# ========================================

# Scalar tuple query
@inline function (itp::ConstantInterpolantND{Tg,Tv,N})(
    query::NTuple{N, <:Real};
    deriv::Union{Int, Val} = 0,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = itp.searches
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_constant_nd(itp, query, ops, search_tuple)
end

# Vector query (for ForwardDiff/Optim compatibility)
@inline function (itp::ConstantInterpolantND{Tg,Tv,N})(
    query::AbstractVector{<:Real};
    deriv::Union{Int, Val} = 0,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = itp.searches
) where {Tg, Tv, N}
    length(query) == N || throw(ArgumentError("Query vector must have $N elements, got $(length(query))"))
    query_tuple = ntuple(i -> query[i], Val(N))
    return itp(query_tuple; deriv=deriv, search=search)
end

# Batch SoA query: tuple of vectors
@inline function (itp::ConstantInterpolantND{Tg,Tv,N})(
    queries::NTuple{N, AbstractVector{<:Real}};
    deriv::Union{Int, Val} = 0,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = itp.searches
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_constant_nd_batch_soa(itp, queries, ops, search_tuple)
end

# Batch AoS query: vector of tuples
@inline function (itp::ConstantInterpolantND{Tg,Tv,N})(
    queries::AbstractVector{<:NTuple{N, <:Real}};
    deriv::Union{Int, Val} = 0,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = itp.searches
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_constant_nd_batch_aos(itp, queries, ops, search_tuple)
end

# ========================================
# Core Evaluation Logic
# ========================================

"""
    _eval_constant_nd(itp, query, ops, search_tuple)

Evaluate ConstantInterpolantND at a single point.

For constant interpolation:
- If any derivative order > 0, return zero(Tv)
- Otherwise, find interval and select corner based on side mode
"""
@inline function _eval_constant_nd(
    itp::ConstantInterpolantND{Tg,Tv,N},
    query::NTuple{N, <:Real},
    ops::NTuple{N, AbstractEvalOp},
    search_tuple::NTuple{N, AbstractSearchPolicy}
) where {Tg, Tv, N}
    # Early return: any derivative order > 0 returns zero
    if _has_any_derivative(ops, Val(N))
        return zero(Tv)
    end

    # Handle extrapolation per axis (shared utility from core/nd_utils.jl)
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)

    # Search intervals (shared utility from core/nd_utils.jl)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, search_tuple)
    hs, dLs = _compute_local_params_constant(q_eval, itp.spacings, indices, Ls, Val(N))

    # Determine corner offsets based on side modes
    offsets = _compute_side_offsets(itp.sides, hs, dLs, Val(N))

    # Fetch value from data array
    return @inbounds itp.data[ntuple(d -> indices[d] + offsets[d], Val(N))...]
end

# ========================================
# Derivative Check
# ========================================

@inline function _has_any_derivative(ops::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N}
    for d in 1:N
        @inbounds if ops[d] isa EvalDeriv1 || ops[d] isa EvalDeriv2 || ops[d] isa EvalDeriv3
            return true
        end
    end
    return false
end

# ========================================
# Local Parameter Computation
# ========================================

@inline function _compute_local_params_constant(
    q_eval::NTuple{N, <:Real},
    spacings::NTuple{N, AbstractGridSpacing},
    indices::NTuple{N, Int},
    Ls::NTuple{N},
    ::Val{N}
) where {N}
    hs = ntuple(Val(N)) do d
        @inbounds _get_h(spacings[d], indices[d])
    end
    dLs = ntuple(Val(N)) do d
        @inbounds q_eval[d] - Ls[d]
    end
    return (hs, dLs)
end

# ========================================
# Side Offset Computation
# ========================================

@inline function _compute_side_offsets(
    sides::NTuple{N, SideVal},
    hs::NTuple{N},
    dLs::NTuple{N},
    ::Val{N}
) where {N}
    return ntuple(Val(N)) do d
        @inbounds _compute_single_offset(sides[d], hs[d], dLs[d])
    end
end

@inline function _compute_single_offset(::Val{:left}, h, dL)
    return 0
end

@inline function _compute_single_offset(::Val{:right}, h, dL)
    dL_primal = _extract_primal(dL)
    return iszero(dL_primal) ? 0 : 1
end

@inline function _compute_single_offset(::Val{:nearest}, h, dL)
    dL_primal = _extract_primal(dL)
    return dL_primal <= h / 2 ? 0 : 1
end

# ========================================
# Batch Evaluation - SoA
# ========================================

function _eval_constant_nd_batch_soa(
    itp::ConstantInterpolantND{Tg,Tv,N},
    queries::NTuple{N, AbstractVector{<:Real}},
    ops::NTuple{N, AbstractEvalOp},
    search_tuple::NTuple{N, AbstractSearchPolicy}
) where {Tg, Tv, N}
    n = length(queries[1])
    for d in 2:N
        length(queries[d]) == n || throw(ArgumentError(
            "All query vectors must have same length, got $(length(queries[d])) at dimension $d vs $n at dimension 1"
        ))
    end

    # Early return: any derivative order > 0 returns zeros
    if _has_any_derivative(ops, Val(N))
        return zeros(Tv, n)
    end

    results = Vector{Tv}(undef, n)
    @inbounds for i in 1:n
        query = ntuple(d -> queries[d][i], Val(N))
        results[i] = _eval_constant_nd(itp, query, ops, search_tuple)
    end
    return results
end

# ========================================
# Batch Evaluation - AoS
# ========================================

function _eval_constant_nd_batch_aos(
    itp::ConstantInterpolantND{Tg,Tv,N},
    queries::AbstractVector{<:NTuple{N, <:Real}},
    ops::NTuple{N, AbstractEvalOp},
    search_tuple::NTuple{N, AbstractSearchPolicy}
) where {Tg, Tv, N}
    n = length(queries)

    # Early return: any derivative order > 0 returns zeros
    if _has_any_derivative(ops, Val(N))
        return zeros(Tv, n)
    end

    results = Vector{Tv}(undef, n)
    @inbounds for i in 1:n
        results[i] = _eval_constant_nd(itp, queries[i], ops, search_tuple)
    end
    return results
end
