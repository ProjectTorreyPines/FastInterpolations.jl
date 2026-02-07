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
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = itp.searches
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_constant_nd(itp, query, ops, search_tuple)
end

# Vector query (for ForwardDiff/Optim compatibility)
@inline function (itp::ConstantInterpolantND{Tg,Tv,N})(
    query::AbstractVector{<:Real};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = itp.searches
) where {Tg, Tv, N}
    length(query) == N || throw(ArgumentError("Query vector must have $N elements, got $(length(query))"))
    query_tuple = ntuple(i -> query[i], Val(N))
    return itp(query_tuple; deriv=deriv, search=search)
end

# Batch SoA query: tuple of vectors
@inline function (itp::ConstantInterpolantND{Tg,Tv,N})(
    queries::NTuple{N, AbstractVector{<:Real}};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = itp.searches
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    return _eval_constant_nd_batch_soa(itp, queries, ops, search_tuple)
end

# Batch AoS query: vector of tuples
@inline function (itp::ConstantInterpolantND{Tg,Tv,N})(
    queries::AbstractVector{<:NTuple{N, <:Real}};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
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
# Generic N-dimensional version
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

    # Use @generated kernel for unrolled evaluation
    return _constant_nd_kernel(itp.data, itp.spacings, itp.sides, indices, q_eval, Ls)
end

# N=2 specialization: direct destructuring eliminates ntuple closure overhead
@inline function _eval_constant_nd(
    itp::ConstantInterpolantND{Tg,Tv,2},
    query::NTuple{2, <:Real},
    ops::NTuple{2, AbstractEvalOp},
    search_tuple::NTuple{2, AbstractSearchPolicy}
) where {Tg, Tv}
    op_x, op_y = ops

    # Early return: any derivative order > 0 returns zero
    if op_x isa EvalDeriv1 || op_x isa EvalDeriv2 || op_x isa EvalDeriv3 ||
       op_y isa EvalDeriv1 || op_y isa EvalDeriv2 || op_y isa EvalDeriv3
        return zero(Tv)
    end

    # Direct destructuring - no ntuple closures
    xq, yq = query
    grid_x, grid_y = itp.grids
    spacing_x, spacing_y = itp.spacings
    extrap_x, extrap_y = itp.extraps
    side_x, side_y = itp.sides
    search_x, search_y = search_tuple

    # Handle extrapolation per axis (direct calls)
    x_eval = _handle_axis_extrap(xq, grid_x, extrap_x)
    y_eval = _handle_axis_extrap(yq, grid_y, extrap_y)

    # Search intervals (direct calls)
    searcher_x = _to_searcher(search_x)
    searcher_y = _to_searcher(search_y)
    ix, xL, _ = search_interval(searcher_x, grid_x, spacing_x, x_eval)
    iy, yL, _ = search_interval(searcher_y, grid_y, spacing_y, y_eval)

    # Compute local parameters and offsets
    hx = _get_h(spacing_x, ix)
    hy = _get_h(spacing_y, iy)
    dLx = x_eval - xL
    dLy = y_eval - yL

    offset_x = _compute_single_offset(side_x, hx, dLx)
    offset_y = _compute_single_offset(side_y, hy, dLy)

    # Direct lookup
    @inbounds return itp.data[ix + offset_x, iy + offset_y]
end

# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================

# Generic N-dimensional
@inline function _locate_cell(
    itp::ConstantInterpolantND{Tg,Tv,N},
    query::NTuple{N, <:Real},
    search_tuple::NTuple{N, AbstractSearchPolicy}
) where {Tg, Tv, N}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, search_tuple)
    return (itp.data, itp.spacings, itp.sides, indices, q_eval, Ls)
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
    data::Array{Tv, N},
    spacings::NTuple{N, AbstractGridSpacing},
    sides::NTuple{N, SideVal},
    indices::NTuple{N, Int},
    q_eval::NTuple{N},
    Ls::NTuple{N}
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

@inline function _eval_constant_nd_batch_soa(
    itp::ConstantInterpolantND{Tg,Tv,N},
    queries::NTuple{N, <:AbstractVector{Tq}},
    ops::OPS,
    search_tuple::SEARCH
) where {Tg, Tv, Tq<:Real, N, OPS<:NTuple{N,AbstractEvalOp}, SEARCH<:NTuple{N,AbstractSearchPolicy}}
    n = length(queries[1])
    for d in 2:N
        length(queries[d]) == n || throw(ArgumentError(
            "All query vectors must have same length, got $(length(queries[d])) at dimension $d vs $n at dimension 1"
        ))
    end

    # Determine output type (include Tq for AD support)
    Tout = promote_type(Tv, Tg, Tq)

    # Early return: any derivative order > 0 returns zeros
    if _has_any_derivative(ops, Val(N))
        return zeros(Tout, n)
    end

    results = Vector{Tout}(undef, n)
    @inbounds for i in 1:n
        query = ntuple(d -> queries[d][i], Val(N))
        results[i] = _eval_constant_nd(itp, query, ops, search_tuple)
    end
    return results
end

# ========================================
# Batch Evaluation - AoS
# ========================================

@inline function _eval_constant_nd_batch_aos(
    itp::ConstantInterpolantND{Tg,Tv,N},
    queries::AbstractVector{<:NTuple{N, Tq}},
    ops::OPS,
    search_tuple::SEARCH
) where {Tg, Tv, Tq<:Real, N, OPS<:NTuple{N,AbstractEvalOp}, SEARCH<:NTuple{N,AbstractSearchPolicy}}
    n = length(queries)

    # Determine output type (include Tq for AD support)
    Tout = promote_type(Tv, Tg, Tq)

    # Early return: any derivative order > 0 returns zeros
    if _has_any_derivative(ops, Val(N))
        return zeros(Tout, n)
    end

    results = Vector{Tout}(undef, n)
    @inbounds for i in 1:n
        results[i] = _eval_constant_nd(itp, queries[i], ops, search_tuple)
    end
    return results
end
