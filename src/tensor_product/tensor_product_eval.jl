# ========================================
# TensorProductInterpolantND — Evaluation
# ========================================
# On-the-fly tensor product via sequential 1D one-shot interpolation.
#
# Algorithm: For query (q₁, q₂, ..., qₙ), collapse dimensions sequentially:
#   1. Along dim 1: for each fiber data[:, i₂, ...], one-shot eval at q₁
#   2. Along dim 2: for each fiber of the intermediate result, eval at q₂
#   3. Continue until scalar result.
#
# Type stability: achieved via recursive Base.tail dispatch (not a loop).

# ========================================
# Hint Peeling Helpers
# ========================================
# hints can be `nothing` (no hints) or a tuple of per-axis hints.
# These helpers allow _collapse_dims to peel hints with Base.tail uniformly.

@inline _first_hint(::Nothing) = nothing
@inline _first_hint(hints::Tuple) = first(hints)
@inline _tail_hints(::Nothing) = nothing
@inline _tail_hints(hints::Tuple) = Base.tail(hints)

# ========================================
# 1D Fiber One-Shot Helpers
# ========================================
# Each method type dispatches to the corresponding 1D one-shot API.
# No intermediate interpolant object is created — direct grid + data + query eval.
# search and hint are forwarded per-axis from the user-specified values.

@inline function _oneshot_eval_1d(m::CubicInterp, grid, fiber, extrap, q, op, search, hint)
    return cubic_interp(grid, fiber, q; bc = m.bc, extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(::LinearInterp, grid, fiber, extrap, q, op, search, hint)
    return linear_interp(grid, fiber, q; extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(m::QuadraticInterp, grid, fiber, extrap, q, op, search, hint)
    return quadratic_interp(grid, fiber, q; bc = m.bc, extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(m::ConstantInterp, grid, fiber, extrap, q, op, search, hint)
    return constant_interp(grid, fiber, q; side = m.side, extrap = extrap, deriv = op, search = search, hint = hint)
end

# ========================================
# Sequential Dimension Collapse
# ========================================
# Recursive type-stable dispatch: each step removes dim 1 and recurses
# with Base.tail of all tuples. Julia infers concrete types at each level.

# Base case: 1D data → one-shot eval final dimension
@inline function _collapse_dims(
        data::AbstractVector,
        grids::Tuple{AbstractVector},
        methods::Tuple{AbstractInterpMethod},
        extraps::Tuple{AbstractExtrap},
        q_eval::Tuple{Real},
        ops::Tuple{AbstractEvalOp},
        searches::Tuple{AbstractSearchPolicy},
        hints,
    )
    return _oneshot_eval_1d(
        methods[1], grids[1], data, extraps[1], q_eval[1], ops[1], searches[1], _first_hint(hints)
    )
end

# Recursive case: collapse dim 1 → (M-1)D array, then recurse
@inline function _collapse_dims(
        data::AbstractArray{Tv, M},
        grids::Tuple{AbstractVector, Vararg{AbstractVector}},
        methods::Tuple{AbstractInterpMethod, Vararg{AbstractInterpMethod}},
        extraps::Tuple{AbstractExtrap, Vararg{AbstractExtrap}},
        q_eval::Tuple{Real, Vararg{Real}},
        ops::Tuple{AbstractEvalOp, Vararg{AbstractEvalOp}},
        searches::Tuple{AbstractSearchPolicy, Vararg{AbstractSearchPolicy}},
        hints,
    ) where {Tv, M}
    # Allocate intermediate array for collapsed result
    remaining_size = Base.tail(size(data))
    result = Array{Tv}(undef, remaining_size...)
    hint_1 = _first_hint(hints)

    # Collapse first dimension: for each fiber along dim 1, one-shot eval
    for idx in CartesianIndices(remaining_size)
        fiber = view(data, :, idx)  # column-major contiguous
        result[idx] = _oneshot_eval_1d(
            first(methods), first(grids), fiber,
            first(extraps), first(q_eval), first(ops), first(searches), hint_1
        )
    end

    # Recurse with remaining dimensions
    return _collapse_dims(
        result, Base.tail(grids), Base.tail(methods),
        Base.tail(extraps), Base.tail(q_eval), Base.tail(ops),
        Base.tail(searches), _tail_hints(hints)
    )
end

# ========================================
# Core Eval Entry Point
# ========================================

# OnTheFly path: sequential 1D one-shot interpolation per query
@inline function _eval_tensor_product_nd(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        searches::NTuple{N, AbstractSearchPolicy},
        hints,
    ) where {Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    return _collapse_dims(itp.data, itp.grids, itp.methods, itp.extraps, q_eval, ops, searches, hints)
end

# PreCompute path: precomputed partials + local kernel eval (O(1) per query)
@inline function _eval_tensor_product_nd(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:HeteroPartials},
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
    _validate_nd_domain(itp.grids, query, itp.extraps)
    oob_result = _try_fill_oob(query, itp.grids, itp.extraps, ops, _zero_ref(itp))
    oob_result !== nothing && return oob_result
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

# GridIdx tuple query form: itp((0.5, GridIdx(3))) — dispatches to NoInterp eval
# Only matches when at least one GridIdx (all-Real → more specific Tuple{Vararg{Real,N}} above)
@inline function (itp::TensorProductInterpolantND{Tg, Tv, N})(
        query::Q;
        deriv = EvalValue(),
        search = itp.searches,
        hint = nothing,
    ) where {Tg, Tv, N, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    ops = _resolve_deriv_nd(deriv, Val(N))
    _validate_nointerp_grididx(itp.methods, query)
    return _eval_nointerp(itp, query, ops, search, hint)
end

# ========================================
# _locate_cell / _eval_at_cell Protocol
# ========================================
# Enables vector_calculus.jl functions (gradient, hessian, laplacian).

# OnTheFly: cell stores everything needed for re-collapse (including searches + hints)
@inline function _locate_cell(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        search_tuple::NTuple{N, AbstractSearchPolicy},
        hints = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    return (itp.data, itp.grids, itp.methods, itp.extraps, q_eval, search_tuple, hints)
end

@inline function _eval_at_cell(
        ::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, G, S, M, E, P}
    data, grids, methods, extraps, q_eval, searches, hints = cell
    return _collapse_dims(data, grids, methods, extraps, q_eval, ops, searches, hints)
end

# PreCompute: cell stores precomputed cell location (locate-once optimization)
@inline function _locate_cell(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:HeteroPartials},
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
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:HeteroPartials},
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
@inline _zero_ref(itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, <:HeteroPartials}) where {Tg, Tv, N, G, S, M, E, P} =
    @inbounds itp.data.partials[1]

@inline _deriv_zero_fill(::TensorProductInterpolantND, ::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N} = false
