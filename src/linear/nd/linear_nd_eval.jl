# ========================================
# LinearInterpolantND Evaluation
# ========================================
#
# Evaluation logic for N-dimensional multilinear interpolation.
# Supports scalar, vector, and batch (SoA/AoS) queries.
#
# Key Algorithm: Tensor-product linear interpolation
# - Sum over 2^N corners with weights determined by normalized coordinates
# - Weights: ∏ᵢ (1-αᵢ if corner_bit=0, αᵢ if corner_bit=1)

# ========================================
# Callable Interface
# ========================================

# Scalar tuple query
@inline function (itp::LinearInterpolantND{Tg,Tv,N})(
    query::NTuple{N, <:Real};
    deriv::Union{Int, Val} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy,N}}} = itp.searches
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_linear_nd(itp, query, ops, search_tuple)
end

# Vector query (for ForwardDiff/Optim compatibility)
@inline function (itp::LinearInterpolantND{Tg,Tv,N})(
    query::AbstractVector{<:Real};
    deriv::Union{Int, Val} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy,N}}} = itp.searches
) where {Tg, Tv, N}
    length(query) == N || throw(ArgumentError("Query vector must have $N elements, got $(length(query))"))
    query_tuple = ntuple(i -> query[i], Val(N))
    return itp(query_tuple; deriv=deriv, search=search)
end

# Batch SoA query: tuple of vectors
@inline function (itp::LinearInterpolantND{Tg,Tv,N})(
    queries::NTuple{N, AbstractVector{<:Real}};
    deriv::Union{Int, Val} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy,N}}} = itp.searches
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_linear_nd_batch_soa(itp, queries, ops, search_tuple)
end

# Batch AoS query: vector of tuples
@inline function (itp::LinearInterpolantND{Tg,Tv,N})(
    queries::AbstractVector{<:NTuple{N, <:Real}};
    deriv::Union{Int, Val} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy,N}}} = itp.searches
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_linear_nd_batch_aos(itp, queries, ops, search_tuple)
end

# ========================================
# Core Evaluation Logic
# ========================================

"""
    _eval_linear_nd(itp, query, ops, search_tuple)

Evaluate LinearInterpolantND at a single point using multilinear interpolation.

Algorithm:
1. Handle extrapolation per axis (shared utility)
2. Search intervals to find containing cell (shared utility)
3. Compute normalized coordinates α = (q - L) / h
4. Sum over 2^N corners with tensor-product weights
"""
@inline function _eval_linear_nd(
    itp::LinearInterpolantND{Tg,Tv,N},
    query::NTuple{N, <:Real},
    ops::NTuple{N, AbstractEvalOp},
    search_tuple::NTuple{N, AbstractSearchPolicy}
) where {Tg, Tv, N}
    # Check for second+ derivative (always zero for linear)
    if _has_second_or_higher_derivative(ops, Val(N))
        return zero(promote_type(Tv, Tg))
    end

    # Handle extrapolation per axis (shared utility from core/nd_utils.jl)
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)

    # Search intervals (shared utility from core/nd_utils.jl)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, search_tuple)

    # Compute local parameters
    hs, αs = _compute_linear_params(q_eval, itp.spacings, indices, Ls, Val(N))

    # Multilinear interpolation sum over 2^N corners
    return _multilinear_sum(itp.data, indices, hs, αs, ops, Val(N))
end

# ========================================
# Derivative Check
# ========================================

@inline function _has_second_or_higher_derivative(ops::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N}
    for d in 1:N
        @inbounds if ops[d] isa EvalDeriv2 || ops[d] isa EvalDeriv3
            return true
        end
    end
    return false
end

# ========================================
# Local Parameter Computation
# ========================================

"""
    _compute_linear_params(q_eval, spacings, indices, Ls, Val(N)) -> (hs, αs)

Compute cell widths and normalized coordinates for multilinear interpolation.
- hs: cell widths for each dimension
- αs: normalized coordinates α = (q - L) / h
"""
@inline function _compute_linear_params(
    q_eval::Tuple{Vararg{Any,N}},
    spacings::Tuple{Vararg{Any,N}},
    indices::NTuple{N, Int},
    Ls::Tuple{Vararg{Any,N}},
    ::Val{N}
) where {N}
    hs = ntuple(Val(N)) do d
        @inbounds _get_h(spacings[d], indices[d])
    end
    αs = ntuple(Val(N)) do d
        @inbounds (q_eval[d] - Ls[d]) / hs[d]
    end
    return (hs, αs)
end

# ========================================
# Multilinear Interpolation Kernel
# ========================================

"""
    _multilinear_sum(data, indices, hs, αs, ops, Val(N))

Compute the multilinear interpolation sum over 2^N corners.

For each corner indexed by bit pattern b ∈ {0,1}^N:
- Corner index: (indices[1]+b₁, indices[2]+b₂, ..., indices[N]+bₙ)
- Weight: ∏ᵢ _linear_weight(ops[i], αs[i], hs[i], bᵢ)

The weight function depends on the evaluation operation:
- EvalValue: (1-α) if b=0, α if b=1
- EvalDeriv1: -1/h if b=0, 1/h if b=1
"""
@generated function _multilinear_sum(
    data::Array{Tv, N},
    indices::NTuple{N, Int},
    hs::NTuple{N},
    αs::NTuple{N},
    ops::NTuple{N, AbstractEvalOp},
    ::Val{N}
) where {Tv, N}
    num_corners = 1 << N  # 2^N

    # Generate the unrolled sum over all corners
    corner_exprs = []
    for corner in 0:(num_corners - 1)
        # Extract bit pattern for this corner
        bits = ntuple(d -> (corner >> (d-1)) & 1, N)

        # Generate index expression: indices[d] + bit
        idx_expr = Expr(:tuple, [:(indices[$d] + $(bits[d])) for d in 1:N]...)

        # Generate weight expression: product of _linear_weight for each dimension
        weight_exprs = [:(
            _linear_weight(ops[$d], αs[$d], hs[$d], Val($(bits[d])))
        ) for d in 1:N]
        weight_expr = foldl((a, b) -> :($a * $b), weight_exprs)

        # data[idx...] * weight
        push!(corner_exprs, :(
            @inbounds data[$idx_expr...] * $weight_expr
        ))
    end

    # Sum all corners
    sum_expr = foldl((a, b) -> :($a + $b), corner_exprs)

    return quote
        @inbounds $sum_expr
    end
end

# ========================================
# Linear Weight Functions
# ========================================

# For value evaluation: (1-α) for bit=0, α for bit=1
@inline _linear_weight(::EvalValue, α, h, ::Val{0}) = one(α) - α
@inline _linear_weight(::EvalValue, α, h, ::Val{1}) = α

# For first derivative: -1/h for bit=0, 1/h for bit=1
@inline _linear_weight(::EvalDeriv1, α, h, ::Val{0}) = -inv(h)
@inline _linear_weight(::EvalDeriv1, α, h, ::Val{1}) = inv(h)

# Second and higher derivatives are zero (handled by early return in _eval_linear_nd)
# But define them for completeness if somehow called
@inline _linear_weight(::EvalDeriv2, α, h, ::Val{B}) where {B} = zero(α)
@inline _linear_weight(::EvalDeriv3, α, h, ::Val{B}) where {B} = zero(α)

# ========================================
# Batch Evaluation - SoA
# ========================================

function _eval_linear_nd_batch_soa(
    itp::LinearInterpolantND{Tg,Tv,N},
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

    # Determine output type
    Tout = promote_type(Tv, Tg)

    results = Vector{Tout}(undef, n)
    @inbounds for i in 1:n
        query = ntuple(d -> queries[d][i], Val(N))
        results[i] = _eval_linear_nd(itp, query, ops, search_tuple)
    end
    return results
end

# ========================================
# Batch Evaluation - AoS
# ========================================

function _eval_linear_nd_batch_aos(
    itp::LinearInterpolantND{Tg,Tv,N},
    queries::AbstractVector{<:NTuple{N, <:Real}},
    ops::NTuple{N, AbstractEvalOp},
    search_tuple::NTuple{N, AbstractSearchPolicy}
) where {Tg, Tv, N}
    n = length(queries)

    # Determine output type
    Tout = promote_type(Tv, Tg)

    results = Vector{Tout}(undef, n)
    @inbounds for i in 1:n
        results[i] = _eval_linear_nd(itp, queries[i], ops, search_tuple)
    end
    return results
end
