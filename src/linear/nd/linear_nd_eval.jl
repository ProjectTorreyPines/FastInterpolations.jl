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
    query::Tuple{Vararg{Real, N}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy,N}}} = itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → BinarySearch/axis
    return _eval_linear_nd(itp, query, ops, search_tuple, hint)
end

# ========================================
# IN-PLACE BATCH EVALUATION
# ========================================

"""
    (itp::LinearInterpolantND)(output, queries::NTuple{N,AbstractVector}; ...)

In-place SoA batch evaluation. Writes results into pre-allocated `output`.
Returns `output` for chaining.
"""
function (itp::LinearInterpolantND{Tg,Tv,N})(
    output::AbstractVector,
    queries::NTuple{N, AbstractVector{<:Real}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy,N}}} = itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tg, Tv, N}
    n_queries = length(queries[1])
    length(output) == n_queries || throw(DimensionMismatch(
        "output length $(length(output)) must match query length $n_queries"
    ))
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N), queries, hint)  # adaptive: check monotonicity for AutoSearch+no hint
    if _has_second_or_higher_derivative(ops, Val(N))
        fill!(output, zero(eltype(output)))
        return output
    end
    _batch_nd_soa!(output, itp, queries, ops, search_tuple, hint)
    return output
end

"""
    (itp::LinearInterpolantND)(output, queries::AbstractVector{<:NTuple}; ...)

In-place AoS batch evaluation. Writes results into pre-allocated `output`.
Returns `output` for chaining.
"""
function (itp::LinearInterpolantND{Tg,Tv,N})(
    output::AbstractVector,
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy,N}}} = itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tg, Tv, N}
    n_queries = length(queries)
    length(output) == n_queries || throw(DimensionMismatch(
        "output length $(length(output)) must match query length $n_queries"
    ))
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N), queries, hint)  # AoS: falls through to _resolve_search_nd
    if _has_second_or_higher_derivative(ops, Val(N))
        fill!(output, zero(eltype(output)))
        return output
    end
    _batch_nd_aos!(output, itp, queries, ops, search_tuple, hint)
    return output
end

# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================

# Generic N-dimensional
@inline function _locate_cell(
    itp::LinearInterpolantND{Tg,Tv,N},
    query::Tuple{Vararg{Real, N}},
    search_tuple::NTuple{N, AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv, N}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, search_tuple, hints)
    hs, αs = _compute_linear_params(q_eval, itp.spacings, indices, Ls, Val(N))
    return (itp.data, indices, hs, αs)
end

# N=2 specialization: direct destructuring eliminates ntuple closure overhead
@inline function _locate_cell(
    itp::LinearInterpolantND{Tg,Tv,2},
    query::Tuple{Vararg{Real, 2}},
    search_tuple::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv}
    x_eval, y_eval, ix, iy, xL, yL = _locate_cell_2d_preamble(
        query, itp.grids, itp.spacings, itp.extraps, search_tuple, hints)

    hx = _get_h(itp.spacings[1], ix)
    hy = _get_h(itp.spacings[2], iy)
    αx = (x_eval - xL) / hx
    αy = (y_eval - yL) / hy

    return (itp.data, (ix, iy), (hx, hy), (αx, αy))
end

# Evaluate kernel at a pre-located cell with given derivative ops
@inline function _eval_at_cell(
    itp::LinearInterpolantND{Tg,Tv,N},
    cell::Tuple,
    ops::NTuple{N, AbstractEvalOp}
) where {Tg, Tv, N}
    if _has_second_or_higher_derivative(ops, Val(N))
        return 0 * first(itp.data)
    end
    data, indices, hs, αs = cell
    return _multilinear_sum(data, indices, hs, αs, ops, Val(N))
end

# ========================================
# Core Evaluation Logic
# ========================================

# Generic N-dimensional version (uses _locate_cell + _eval_at_cell)
@inline function _eval_linear_nd(
    itp::LinearInterpolantND{Tg,Tv,N},
    query::Tuple{Vararg{Real, N}},
    ops::NTuple{N, AbstractEvalOp},
    search_tuple::NTuple{N, AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv, N}
    if _has_second_or_higher_derivative(ops, Val(N))
        return 0 * first(itp.data)
    end
    cell = _locate_cell(itp, query, search_tuple, hints)
    return _eval_at_cell(itp, cell, ops)
end

# N=2 specialization: dispatches to N=2 _locate_cell via type
@inline function _eval_linear_nd(
    itp::LinearInterpolantND{Tg,Tv,2},
    query::Tuple{Vararg{Real, 2}},
    ops::NTuple{2, AbstractEvalOp},
    search_tuple::NTuple{2, AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv}
    op_x, op_y = ops
    if op_x isa EvalDeriv2 || op_x isa EvalDeriv3 || op_y isa EvalDeriv2 || op_y isa EvalDeriv3
        return 0 * first(itp.data)
    end
    cell = _locate_cell(itp, query, search_tuple, hints)
    return _eval_at_cell(itp, cell, ops)
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
    q_eval::Tuple{Vararg{Real,N}},
    spacings::Tuple{Vararg{AbstractGridSpacing,N}},
    indices::NTuple{N, Int},
    Ls::Tuple{Vararg{Real,N}},
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
    data::AbstractArray{Tv, N},
    indices::NTuple{N, Int},
    hs::NTuple{N},
    αs::Tuple{Vararg{Real, N}},
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
        Base.@_inline_meta
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

