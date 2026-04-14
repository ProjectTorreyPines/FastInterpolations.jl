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
@inline function (itp::ConstantInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    resolved = map(_resolve_grididx, query, itp.grids)
    ops = _resolve_deriv_nd(deriv, Val(N))
    policies = _resolve_search_nd(search, Val(N))
    hints = _ensure_hint_nd(hint, Val(N))
    mono = ntuple(_ -> true, Val(N))
    return _eval_constant_nd(itp, resolved, ops, policies, hints, mono)
end

# In-place batch evaluation (SoA + AoS) is handled by the unified
# AbstractInterpolantND callable in nd_interpolant_protocol.jl.
# Zero-fill for any derivative is handled by _deriv_zero_fill trait below.

# Derivative zero-fill trait: constant has zero derivative at all orders
@inline _deriv_zero_fill(::ConstantInterpolantND, ops::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N} =
    _has_any_derivative(ops, Val(N))

# ========================================
# Core Evaluation Logic
# ========================================

"""
    _eval_constant_nd(itp, query, ops, search_tuple)

Evaluate ConstantInterpolantND at a single point.

For constant interpolation:
- If any derivative order > 0, return zero via `0 * y` (duck-typing compatible)
- Otherwise, find interval and select corner based on side mode
"""
# Zero-ref for fill-value derivative computation (duck-typed zero via 0 * data_element)
@inline _zero_ref(itp::ConstantInterpolantND) = @inbounds first(itp.data)

# Generic N-dimensional version (uses _locate_cell + _eval_at_cell)
@inline function _eval_constant_nd(
        itp::ConstantInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    _validate_nd_domain(itp.grids, query, itp.extraps)
    oob_result = _try_fill_oob(query, itp.grids, itp.extraps, ops, _zero_ref(itp))
    oob_result !== nothing && return oob_result
    if _has_any_derivative(ops, Val(N))
        return 0 * first(itp.data)
    end
    cell = _locate_cell(itp, query, policies, hints, mono)
    return _eval_at_cell(itp, cell, ops)
end

# N=2 specialization: dispatches to N=2 _locate_cell via type
@inline function _eval_constant_nd(
        itp::ConstantInterpolantND{Tg, Tv, 2},
        query::Tuple{Vararg{Real, 2}},
        ops::NTuple{2, AbstractEvalOp},
        policies::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
        hints::Tuple{Base.RefValue{Int}, Base.RefValue{Int}},
        mono::Tuple{Bool, Bool},
    ) where {Tg, Tv}
    _validate_nd_domain(itp.grids, query, itp.extraps)
    oob_result = _try_fill_oob(query, itp.grids, itp.extraps, ops, _zero_ref(itp))
    oob_result !== nothing && return oob_result
    op_x, op_y = ops
    if !(op_x isa EvalValue) || !(op_y isa EvalValue)
        return 0 * first(itp.data)
    end
    cell = _locate_cell(itp, query, policies, hints, mono)
    return _eval_at_cell(itp, cell, ops)
end

# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================

# Generic N-dimensional
@inline function _locate_cell(
        itp::ConstantInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, policies, hints, mono)
    return (itp.data, itp.spacings, itp.sides, indices, q_eval, Ls)
end

# N=2 specialization: direct destructuring eliminates ntuple closure overhead
@inline function _locate_cell(
        itp::ConstantInterpolantND{Tg, Tv, 2},
        query::Tuple{Vararg{Real, 2}},
        policies::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
        hints::Tuple{Base.RefValue{Int}, Base.RefValue{Int}},
        mono::Tuple{Bool, Bool},
    ) where {Tg, Tv}
    x_eval, y_eval, ix, iy, xL, yL = _locate_cell_2d_preamble(
        query, itp.grids, itp.spacings, itp.extraps, policies, hints, mono
    )

    return (itp.data, itp.spacings, itp.sides, (ix, iy), (x_eval, y_eval), (xL, yL))
end

# Evaluate kernel at a pre-located cell with given derivative ops
@inline function _eval_at_cell(
        itp::ConstantInterpolantND{Tg, Tv, N},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tg, Tv, N}
    if _has_any_derivative(ops, Val(N))
        return 0 * first(itp.data)
    end
    data, spacings, sides, indices, q_eval, Ls = cell
    return _constant_nd_kernel(data, spacings, sides, indices, q_eval, Ls)
end

# ========================================
# Derivative Check
# ========================================

@inline function _has_any_derivative(ops::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N}
    for d in 1:N
        @inbounds if !(ops[d] isa EvalValue)
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

# Side offset helpers (_compute_single_offset) are defined in
# src/constant/constant_kernels.jl and shared by 1D adjoint and ND eval.
