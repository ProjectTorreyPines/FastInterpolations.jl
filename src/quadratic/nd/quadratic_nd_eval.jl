# ========================================
# Generic ND Quadratic Evaluation
# ========================================
#
# N-dimensional quadratic interpolation with:
# - Tg/Tv type separation (grid vs value types)
# - @generated tensor product for zero-allocation O(1) evaluation
# - N=2 specialization for optimal performance
# - AD support (query type preserved through evaluation)
#
# Key difference from cubic: uses 3 nodal values (fL, fR, dfL) per dimension
# instead of 4 (fL, fR, dfL, dfR). The quadratic coefficient `a` is computed
# on-the-fly from these 3 values.

# ========================================
# 1D Quadratic Kernel for ND Tensor Product
# ========================================

"""
    _quadratic_kernel_nd(op, fL, fR, dfL, inv_h, dL)

Evaluate 1D quadratic kernel for ND tensor product evaluation.
Computes quadratic coefficient `a` on-the-fly from (fL, fR, dfL, inv_h),
then delegates to `_quadratic_kernel(op, a, dfL, fL, dL)`.

This is the quadratic analog of `_hermite_kernel_1d(op, yL, yR, dyL, dyR, h, inv_h, dL)`.
Note: Unlike Hermite, quadratic works in physical coordinates so `h` is not needed.
"""
@inline function _quadratic_kernel_nd(
    op::AbstractEvalOp,
    fL, fR, dfL,
    inv_h, dL
)
    s = (fR - fL) * inv_h    # secant slope
    a = (s - dfL) * inv_h     # quadratic coefficient
    return _quadratic_kernel(op, a, dfL, fL, dL)
end

# ========================================
# CALLABLE INTERFACE
# ========================================

"""
    (itp::QuadraticInterpolantND)(query; deriv=EvalValue(), search=itp.searches)

Evaluate N-dimensional quadratic interpolant.

# Keywords
- `deriv`: Derivative specification
  - `DerivOp`: same order for all axes, e.g. `EvalValue()`, `DerivOp(1)`
  - `NTuple{N,DerivOp}`: per-axis orders, e.g. `(DerivOp(1),EvalValue())` for ∂f/∂x
- `search`: Override search policy (single or per-axis tuple)

# Examples
```julia
itp((1.0, 0.5))                                      # value
itp((1.0, 0.5); deriv=DerivOp(1))                     # all first derivatives
itp((1.0, 0.5); deriv=(DerivOp(1), EvalValue()))      # ∂f/∂x only
```
"""
# Single-point evaluation
@inline function (itp::QuadraticInterpolantND{Tg, Tv, N})(
    query::Tuple{Vararg{Real, N}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}}=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → Binary/axis
    return _eval_nd_quadratic(itp, query, ops, search_tuple, hint)
end

# ========================================
# IN-PLACE BATCH EVALUATION
# ========================================

"""
    (itp::QuadraticInterpolantND)(output, queries::NTuple{N,AbstractVector}; ...)

In-place SoA batch evaluation. Writes results into pre-allocated `output`.
Returns `output` for chaining.
"""
function (itp::QuadraticInterpolantND{Tg, Tv, N})(
    output::AbstractVector,
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}}=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
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
    search_tuple = _resolve_search_nd_adaptive(search, Val(N), queries, hint)  # adaptive: check monotonicity for AutoSearch+no hint
    _batch_nd_soa!(output, itp, queries, ops, search_tuple, hint)
    return output
end

"""
    (itp::QuadraticInterpolantND)(output, queries::AbstractVector{<:Tuple}; ...)

In-place AoS batch evaluation. Writes results into pre-allocated `output`.
Returns `output` for chaining.
"""
function (itp::QuadraticInterpolantND{Tg, Tv, N})(
    output::AbstractVector,
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}}=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg, Tv, N}
    n_queries = length(queries)
    length(output) == n_queries || throw(DimensionMismatch(
        "output length $(length(output)) must match query length $n_queries"
    ))
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd_adaptive(search, Val(N), queries, hint)  # AoS: falls through to _resolve_search_nd
    _batch_nd_aos!(output, itp, queries, ops, search_tuple, hint)
    return output
end


# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================

# Generic N-dimensional
@inline function _locate_cell(
    itp::QuadraticInterpolantND{Tg, Tv, N},
    query::Tuple{Vararg{Real, N}},
    search::SEARCH,
    hints=nothing
) where {Tg, Tv, N, SEARCH<:NTuple{N,AbstractSearchPolicy}}
    q_evals = _handle_all_extraps(query, itp.grids, itp.extraps)
    indices, Ls, _ = _search_all_intervals(q_evals, itp.grids, itp.spacings, search, hints)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, itp.spacings, indices, Ls)

    return (itp.nodal_derivs.partials, indices, hs, inv_hs, dLs)
end

# N=2 specialization: direct destructuring eliminates ntuple closure overhead
@inline function _locate_cell(
    itp::QuadraticInterpolantND{Tg, Tv, 2},
    query::Tuple{Vararg{Real, 2}},
    search::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv}
    x_eval, y_eval, ix, iy, xL, yL = _locate_cell_2d_preamble(
        query, itp.grids, itp.spacings, itp.extraps, search, hints)

    hx = _get_h(itp.spacings[1], ix);  hy = _get_h(itp.spacings[2], iy)
    inv_hx = _get_inv_h(itp.spacings[1], ix); inv_hy = _get_inv_h(itp.spacings[2], iy)
    dLx = x_eval - xL;  dLy = y_eval - yL

    return (itp.nodal_derivs.partials, (ix, iy), (hx, hy), (inv_hx, inv_hy), (dLx, dLy))
end

# Evaluate kernel at a pre-located cell with given derivative ops
@inline function _eval_at_cell(
    ::QuadraticInterpolantND,
    cell::Tuple,
    ops::NTuple{N, AbstractEvalOp}
) where {N}
    partials, indices, hs, inv_hs, dLs = cell
    return _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
end

# ========================================
# CORE QUADRATIC EVALUATION
# ========================================

# Generic N-dimensional (uses _locate_cell + _eval_at_cell)
@inline function _eval_nd_quadratic(
    itp::QuadraticInterpolantND{Tg, Tv, N},
    query::Tuple{Vararg{Real, N}},
    ops::OPS,
    search::SEARCH,
    hints=nothing
) where {Tg, Tv, N, OPS<:NTuple{N,AbstractEvalOp}, SEARCH<:NTuple{N,AbstractSearchPolicy}}
    cell = _locate_cell(itp, query, search, hints)
    return _eval_at_cell(itp, cell, ops)
end

# N=2 specialization: dispatches to N=2 _locate_cell via type
@inline function _eval_nd_quadratic(
    itp::QuadraticInterpolantND{Tg, Tv, 2},
    query::Tuple{Vararg{Real, 2}},
    ops::Tuple{<:AbstractEvalOp, <:AbstractEvalOp},
    search::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
    hints=nothing
) where {Tg, Tv}
    cell = _locate_cell(itp, query, search, hints)
    return _eval_at_cell(itp, cell, ops)
end

# ========================================
# @GENERATED TENSOR PRODUCT KERNEL (Quadratic)
# ========================================
# Same dimension-collapsing strategy as cubic, but uses 3 nodal values
# per dimension (fL, fR, dfL) instead of 4 (fL, fR, dfL, dfR).
# The quadratic coefficient `a` is computed on-the-fly in the kernel.

@inline @generated function _eval_nd_quad_cell(
    partials::AbstractArray{Tv, NP1},
    indices::NTuple{N, Int},
    hs::NTuple{N, Tg},
    inv_hs::NTuple{N, Tg},
    dLs::Tuple{Vararg{Real, N}},
    ops::NTuple{N, AbstractEvalOp}
) where {Tv, Tg, N, NP1}
    NP1 == N + 1 || error("NP1 must equal N+1")

    stmts = Expr[]

    # Unpack tuples using destructuring
    for (prefix, source) in [("idx_", :indices), ("h_", :hs), ("inv_h_", :inv_hs),
                              ("dL_", :dLs), ("op_", :ops)]
        syms = ntuple(d -> Symbol(prefix, d), N)
        lhs = Expr(:tuple, syms...)
        push!(stmts, :($lhs = $source))
    end

    # Collapse each dimension
    for stage in 1:N
        num_corners = 1 << (N - stage)
        num_derivs = 1 << (N - stage)

        for corner in 0:(num_corners - 1)
            for deriv in 0:(num_derivs - 1)
                out_var = _varname(stage, corner, deriv)

                if stage == 1
                    # Read from partials array
                    function make_partial_access_q(c_dim1::Int, d_dim1::Int)
                        corner_full = c_dim1 | (corner << 1)
                        deriv_full = d_dim1 | (deriv << 1)
                        p_idx = _partial_index(deriv_full)
                        offsets = _corner_offset_expr(corner_full, N)
                        idx_exprs = [:($(Symbol("idx_", d)) + $(offsets[d])) for d in 1:N]
                        return :(partials[$p_idx, $(idx_exprs...)])
                    end

                    fL = make_partial_access_q(0, 0)   # left corner, no deriv
                    fR = make_partial_access_q(1, 0)   # right corner, no deriv
                    dfL = make_partial_access_q(0, 1)  # left corner, has deriv
                    # NO dfR — quadratic only needs 3 values
                else
                    # Read from previous stage variables
                    prev_stage = stage - 1
                    fL = _varname(prev_stage, 0 | (corner << 1), 0 | (deriv << 1))
                    fR = _varname(prev_stage, 1 | (corner << 1), 0 | (deriv << 1))
                    dfL = _varname(prev_stage, 0 | (corner << 1), 1 | (deriv << 1))
                    # NO dfR — quadratic only needs 3 values
                end

                inv_h = Symbol("inv_h_", stage)
                dL = Symbol("dL_", stage)
                op = Symbol("op_", stage)

                kernel_call = :(_quadratic_kernel_nd($op, $fL, $fR, $dfL, $inv_h, $dL))
                push!(stmts, :($out_var = $kernel_call))
            end
        end
    end

    # Final result
    final_var = _varname(N, 0, 0)
    push!(stmts, :(return $final_var))

    result = quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
    end

    return result
end
