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
    query::Tuple{Vararg{Real, N}};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_constant_nd(itp, query, ops, search_tuple, hint)
end

# ========================================
# IN-PLACE BATCH EVALUATION
# ========================================

"""
    (itp::ConstantInterpolantND)(output, queries::NTuple{N,AbstractVector}; ...)

In-place SoA batch evaluation. Writes results into pre-allocated `output`.
Returns `output` for chaining.
"""
function (itp::ConstantInterpolantND{Tg,Tv,N})(
    output::AbstractVector,
    queries::NTuple{N, AbstractVector{<:Real}};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
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
    search_tuple = _resolve_search_nd(search, Val(N))

    if deriv isa Int
        @_dispatch_deriv deriv => op begin
            ops = ntuple(_ -> op, Val(N))
            if _has_any_derivative(ops, Val(N))
                fill!(output, zero(eltype(output)))
                return output
            end
            _batch_nd_soa!(output, itp, queries, ops, search_tuple, hint)
        end
    elseif deriv isa Val
        ops = _resolve_deriv_nd(deriv, Val(N))
        if _has_any_derivative(ops, Val(N))
            fill!(output, zero(eltype(output)))
            return output
        end
        _batch_nd_soa!(output, itp, queries, ops, search_tuple, hint)
    else
        ops = _resolve_deriv_nd(Val(deriv), Val(N))
        if _has_any_derivative(ops, Val(N))
            fill!(output, zero(eltype(output)))
            return output
        end
        _batch_nd_soa!(output, itp, queries, ops, search_tuple, hint)
    end
    return output
end

"""
    (itp::ConstantInterpolantND)(output, queries::AbstractVector{<:NTuple}; ...)

In-place AoS batch evaluation. Writes results into pre-allocated `output`.
Returns `output` for chaining.
"""
function (itp::ConstantInterpolantND{Tg,Tv,N})(
    output::AbstractVector,
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tg, Tv, N}
    n_queries = length(queries)
    length(output) == n_queries || throw(DimensionMismatch(
        "output length $(length(output)) must match query length $n_queries"
    ))
    search_tuple = _resolve_search_nd(search, Val(N))

    if deriv isa Int
        @_dispatch_deriv deriv => op begin
            ops = ntuple(_ -> op, Val(N))
            if _has_any_derivative(ops, Val(N))
                fill!(output, zero(eltype(output)))
                return output
            end
            _batch_nd_aos!(output, itp, queries, ops, search_tuple, hint)
        end
    elseif deriv isa Val
        ops = _resolve_deriv_nd(deriv, Val(N))
        if _has_any_derivative(ops, Val(N))
            fill!(output, zero(eltype(output)))
            return output
        end
        _batch_nd_aos!(output, itp, queries, ops, search_tuple, hint)
    else
        ops = _resolve_deriv_nd(Val(deriv), Val(N))
        if _has_any_derivative(ops, Val(N))
            fill!(output, zero(eltype(output)))
            return output
        end
        _batch_nd_aos!(output, itp, queries, ops, search_tuple, hint)
    end
    return output
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
# Generic N-dimensional version (uses _locate_cell + _eval_at_cell)
@inline function _eval_constant_nd(
    itp::ConstantInterpolantND{Tg,Tv,N},
    query::Tuple{Vararg{Real, N}},
    ops::NTuple{N, AbstractEvalOp},
    search_tuple::NTuple{N, AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv, N}
    if _has_any_derivative(ops, Val(N))
        return zero(Tv)
    end
    cell = _locate_cell(itp, query, search_tuple, hints)
    return _eval_at_cell(itp, cell, ops)
end

# N=2 specialization: dispatches to N=2 _locate_cell via type
@inline function _eval_constant_nd(
    itp::ConstantInterpolantND{Tg,Tv,2},
    query::Tuple{Vararg{Real, 2}},
    ops::NTuple{2, AbstractEvalOp},
    search_tuple::NTuple{2, AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv}
    op_x, op_y = ops
    if op_x isa EvalDeriv1 || op_x isa EvalDeriv2 || op_x isa EvalDeriv3 ||
       op_y isa EvalDeriv1 || op_y isa EvalDeriv2 || op_y isa EvalDeriv3
        return zero(Tv)
    end
    cell = _locate_cell(itp, query, search_tuple, hints)
    return _eval_at_cell(itp, cell, ops)
end

# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================

# Generic N-dimensional
@inline function _locate_cell(
    itp::ConstantInterpolantND{Tg,Tv,N},
    query::Tuple{Vararg{Real, N}},
    search_tuple::NTuple{N, AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv, N}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, search_tuple, hints)
    return (itp.data, itp.spacings, itp.sides, indices, q_eval, Ls)
end

# N=2 specialization: direct destructuring eliminates ntuple closure overhead
@inline function _locate_cell(
    itp::ConstantInterpolantND{Tg,Tv,2},
    query::Tuple{Vararg{Real, 2}},
    search_tuple::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv}
    x_eval, y_eval, ix, iy, xL, yL = _locate_cell_2d_preamble(
        query, itp.grids, itp.spacings, itp.extraps, search_tuple, hints)

    return (itp.data, itp.spacings, itp.sides, (ix, iy), (x_eval, y_eval), (xL, yL))
end

# Evaluate kernel at a pre-located cell with given derivative ops
@inline function _eval_at_cell(
    itp::ConstantInterpolantND{Tg,Tv,N},
    cell::Tuple,
    ops::NTuple{N, AbstractEvalOp}
) where {Tg, Tv, N}
    if _has_any_derivative(ops, Val(N))
        return zero(Tv)
    end
    data, spacings, sides, indices, q_eval, Ls = cell
    return _constant_nd_kernel(data, spacings, sides, indices, q_eval, Ls)
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
# Generated Kernel for Constant ND Evaluation
# ========================================

"""
    _constant_nd_kernel(data, spacings, sides, indices, q_eval, Ls)

@generated kernel that unrolls the constant interpolation lookup for N dimensions.
Computes cell widths, distances from left edge, side-based offsets, and returns data value.
"""
@generated function _constant_nd_kernel(
    data::AbstractArray{Tv, N},
    spacings::NTuple{N, AbstractGridSpacing},
    sides::Tuple{Vararg{AbstractSide, N}},
    indices::NTuple{N, Int},
    q_eval::Tuple{Vararg{Real, N}},
    Ls::Tuple{Vararg{Real, N}}
) where {Tv, N}
    # Build list of expressions at compile-time (loops here are fine)
    exprs = Expr[]

    # Generate: h_d = _get_h(spacings[d], indices[d]) for each dimension
    for d in 1:N
        h_sym = Symbol("h_", d)
        push!(exprs, :($h_sym = @inbounds _get_h(spacings[$d], indices[$d])))
    end

    # Generate: dL_d = q_eval[d] - Ls[d] for each dimension
    for d in 1:N
        dL_sym = Symbol("dL_", d)
        push!(exprs, :($dL_sym = @inbounds q_eval[$d] - Ls[$d]))
    end

    # Generate: offset_d = _compute_single_offset(sides[d], h_d, dL_d) for each dimension
    for d in 1:N
        h_sym = Symbol("h_", d)
        dL_sym = Symbol("dL_", d)
        offset_sym = Symbol("offset_", d)
        push!(exprs, :($offset_sym = _compute_single_offset(sides[$d], $h_sym, $dL_sym)))
    end

    # Build final index expression: (indices[1]+offset_1, indices[2]+offset_2, ...)
    idx_parts = Expr[]
    for d in 1:N
        offset_sym = Symbol("offset_", d)
        push!(idx_parts, :(indices[$d] + $offset_sym))
    end
    idx_expr = Expr(:tuple, idx_parts...)

    # Build the final data access
    push!(exprs, :(@inbounds data[$idx_expr...]))

    # Build final block expression (no comprehensions/generators in returned AST)
    return Expr(:block, :(Base.@_inline_meta), exprs...)
end

# ========================================
# Side Offset Computation (dispatch helpers for @generated kernel)
# ========================================

@inline function _compute_single_offset(::LeftSide, h, dL)
    return 0
end

@inline function _compute_single_offset(::RightSide, h, dL)
    dL_primal = _extract_primal(dL)
    return iszero(dL_primal) ? 0 : 1
end

@inline function _compute_single_offset(::NearestSide, h, dL)
    dL_primal = _extract_primal(dL)
    return dL_primal <= h / 2 ? 0 : 1
end

