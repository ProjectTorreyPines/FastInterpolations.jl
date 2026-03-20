# ========================================
# TensorProductInterpolantND — Evaluation
# ========================================
# On-the-fly tensor product via sequential 1D interpolation.
#
# Algorithm: For query (q₁, q₂, ..., qₙ), collapse dimensions sequentially:
#   1. Along dim 1: for each fiber data[:, i₂, ...], build 1D interp, eval at q₁
#   2. Along dim 2: for each fiber of the intermediate result, eval at q₂
#   3. Continue until scalar result.
#
# Type stability: achieved via recursive Base.tail dispatch (not a loop).

# ========================================
# 1D Fiber Build-and-Evaluate Helpers
# ========================================
# Each method type dispatches to the corresponding 1D public API.
# BinarySearch is used since we evaluate a single point per fiber.

@inline function _build_and_eval_1d(m::CubicInterp, grid, fiber, extrap, q, op)
    itp1d = cubic_interp(grid, fiber; bc = m.bc, extrap = extrap, search = BinarySearch())
    return itp1d(q; deriv = op)
end

@inline function _build_and_eval_1d(::LinearInterp, grid, fiber, extrap, q, op)
    itp1d = linear_interp(grid, fiber; extrap = extrap, search = BinarySearch())
    return itp1d(q; deriv = op)
end

@inline function _build_and_eval_1d(m::QuadraticInterp, grid, fiber, extrap, q, op)
    itp1d = quadratic_interp(grid, fiber; bc = m.bc, extrap = extrap, search = BinarySearch())
    return itp1d(q; deriv = op)
end

@inline function _build_and_eval_1d(m::ConstantInterp, grid, fiber, extrap, q, op)
    itp1d = constant_interp(grid, fiber; side = m.side, extrap = extrap, search = BinarySearch())
    return itp1d(q; deriv = op)
end

# ========================================
# Sequential Dimension Collapse
# ========================================
# Recursive type-stable dispatch: each step removes dim 1 and recurses
# with Base.tail of all tuples. Julia infers concrete types at each level.

# Base case: 1D data → build and eval final interpolant
@inline function _collapse_dims(
        data::AbstractVector,
        grids::Tuple{AbstractVector},
        methods::Tuple{AbstractInterpMethod},
        extraps::Tuple{AbstractExtrap},
        q_eval::Tuple{Real},
        ops::Tuple{AbstractEvalOp},
    )
    return _build_and_eval_1d(methods[1], grids[1], data, extraps[1], q_eval[1], ops[1])
end

# Recursive case: collapse dim 1 → (M-1)D array, then recurse
@inline function _collapse_dims(
        data::AbstractArray{Tv, M},
        grids::Tuple{AbstractVector, Vararg{AbstractVector}},
        methods::Tuple{AbstractInterpMethod, Vararg{AbstractInterpMethod}},
        extraps::Tuple{AbstractExtrap, Vararg{AbstractExtrap}},
        q_eval::Tuple{Real, Vararg{Real}},
        ops::Tuple{AbstractEvalOp, Vararg{AbstractEvalOp}},
    ) where {Tv, M}
    # Allocate intermediate array for collapsed result
    remaining_size = Base.tail(size(data))
    result = Array{Tv}(undef, remaining_size...)

    # Collapse first dimension: for each fiber along dim 1, build 1D interp + eval
    for idx in CartesianIndices(remaining_size)
        fiber = view(data, :, idx)  # column-major contiguous
        result[idx] = _build_and_eval_1d(
            first(methods), first(grids), fiber,
            first(extraps), first(q_eval), first(ops)
        )
    end

    # Recurse with remaining dimensions
    return _collapse_dims(
        result, Base.tail(grids), Base.tail(methods),
        Base.tail(extraps), Base.tail(q_eval), Base.tail(ops)
    )
end

# ========================================
# Core Eval Entry Point
# ========================================

# OnTheFly path: sequential 1D interpolation (builds 1D interps per query)
@inline function _eval_tensor_product_nd(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        searches::NTuple{N, AbstractSearchPolicy},
        hints,
    ) where {Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    return _collapse_dims(itp.data, itp.grids, itp.methods, itp.extraps, q_eval, ops)
end

# PreCompute path: precomputed partials + local kernel eval (O(1) per query)
@inline function _eval_tensor_product_nd(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:NodalDerivativesND},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        searches::NTuple{N, AbstractSearchPolicy},
        hints,
    ) where {Tg, Tv, N, G, S, M, E, P}
    return _eval_tensor_product_precomputed(
        itp.data, itp.grids, itp.spacings, itp.methods, itp.extraps,
        query, ops, searches, hints
    )
end

# ========================================
# Callable Interface
# ========================================

# Tuple query form
@inline function (itp::TensorProductInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Real, N}};
        deriv = EvalValue(),
        search = itp.searches,
        hint = nothing,
    ) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N), query)
    return _eval_tensor_product_nd(itp, query, ops, search_tuple, hint)
end

# Vararg form: itp(0.5, 0.3) → itp((0.5, 0.3))
@inline function (itp::TensorProductInterpolantND{Tg, Tv, N})(
        q::Vararg{Real, N};
        kw...,
    ) where {Tg, Tv, N}
    return itp(q; kw...)
end

# ========================================
# _locate_cell / _eval_at_cell Protocol
# ========================================
# Enables vector_calculus.jl functions (gradient, hessian, laplacian).

# OnTheFly: cell stores everything needed for re-collapse
@inline function _locate_cell(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        search_tuple::NTuple{N, AbstractSearchPolicy},
        hints = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    return (itp.data, itp.grids, itp.methods, itp.extraps, q_eval)
end

@inline function _eval_at_cell(
        ::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, G, S, M, E, P}
    data, grids, methods, extraps, q_eval = cell
    return _collapse_dims(data, grids, methods, extraps, q_eval, ops)
end

# PreCompute: cell stores precomputed cell location (locate-once optimization)
@inline function _locate_cell(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:NodalDerivativesND},
        query::Tuple{Vararg{Real, N}},
        search_tuple::NTuple{N, AbstractSearchPolicy},
        hints = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, search_tuple, hints)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, itp.spacings, indices, Ls)
    return (itp.data.partials, indices, hs, inv_hs, dLs)
end

@inline function _eval_at_cell(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:NodalDerivativesND},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, G, S, M, E, P}
    partials, indices, hs, inv_hs, dLs = cell
    return _eval_hetero_nd_cell(partials, indices, hs, inv_hs, dLs, ops, itp.methods)
end

# ========================================
# Required Traits
# ========================================

@inline _zero_ref(itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array}) where {Tg, Tv, N, G, S, M, E, P} =
    @inbounds first(itp.data)
@inline _zero_ref(itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:NodalDerivativesND}) where {Tg, Tv, N, G, S, M, E, P} =
    @inbounds itp.data.partials[1]

@inline _deriv_zero_fill(::TensorProductInterpolantND, ::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N} = false
