# ========================================
# Generic ND Quadratic Evaluation
# ========================================
#
# N-dimensional quadratic interpolation with:
# - Tg/Tv type separation (grid vs value types)
# - @generated tensor product for zero-allocation O(1) evaluation
# - Generic-N tensor-product locate (no N=2 specialization — verified equal-or-slower)
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
        fL::Tv, fR, dfL,
        inv_h::Tinv, dL
    ) where {Tinv, Tv}
    # No `h` here (physical coords) — `inv_h` is the only spacing arg and is its own
    # type `Tinv` (≠ grid `Tg`: `inv(Int)::Float`). Witness the coeff field through it.
    s = _fielddiff(_promote_eltype(_coeff_op, Tinv, Tv), fR, fL) * inv_h    # secant slope
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
        query::Tuple{Vararg{Number, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _eval_nd_scalar_query(itp, query, deriv, extrap, search, hint)
end

# In-place batch evaluation (SoA + AoS) is handled by the unified
# AbstractInterpolantND callable in interpolant_protocol.jl. Scalar
# evaluation routes through the generic `_eval_nd_at_point` there —
# QuadraticInterpolantND has no zero-fill trait (default false).

# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================

# Generic N-dimensional. `extraps` carries batch-level InBounds promotion
# from `_validate_nd_domain` when applicable; scalar callers route via the
# 5-arg forwarder (interpolant_protocol.jl) injecting `itp.extraps`.
@inline function _locate_cell(
        itp::QuadraticInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    q_evals = _handle_all_extraps(query, itp.grids, extraps)
    # 6-arg search: per-axis `extraps` let InBounds range axes take the lean direct
    # search (one-sided clamp; hint still written back) — bit-identical, per-axis, all N.
    # `_compute_all_local_params` uses the spacings-free overload (cached h/inv_h).
    indices, Ls, _ = _search_all_intervals(q_evals, itp.grids, policies, hints, mono, extraps)
    # Non-Real axes: the scaled store is [Y]-homogeneous, so the kernel consumes
    # dimensionless local params (type-folded — Real is the exact old call).
    hs, inv_hs, dLs = Tg <: Real ?
        _compute_all_local_params(q_evals, itp.grids, indices, Ls) :
        _compute_all_local_params_reparam(q_evals, itp.grids, indices, Ls)

    return (itp.nodal_derivs.partials, indices, hs, inv_hs, dLs)
end

# No N=2 specialization: the generic-N locate above inlines to the same code at
# N=2, so a hand-destructured 2D variant is equal-or-slower (verified via
# same-process method-swap A/B).

# Evaluate kernel at a pre-located cell with given derivative ops.
# Non-Real axes: the kernel runs dimensionless over the [Y]-scaled store, so
# derivative results restore their per-axis grid⁻ᵏ units at this single seam
# (canonical `_nd_deriv_scale` fold; `true` on value ops and Real grids — folds).
@inline function _eval_at_cell(
        itp::QuadraticInterpolantND{Tg},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tg, N}
    partials, indices, _, inv_hs, dLs = cell   # `hs` in the cell tuple is unused by quadratic
    r = _eval_nd_quad_cell(partials, indices, inv_hs, dLs, ops)
    return Tg <: Real ? r : r * _nd_fill_deriv_scale(itp.grids, ops)
end

# Per-method sample of `Tv` for fill-value paths (e.g. `_try_fill_oob`).
@inline _sample_data(itp::QuadraticInterpolantND) = @inbounds first(itp.nodal_derivs.partials)

# ========================================
# @GENERATED TENSOR PRODUCT KERNEL (Quadratic)
# ========================================
# Same dimension-collapsing strategy as cubic, but uses 3 nodal values
# per dimension (fL, fR, dfL) instead of 4 (fL, fR, dfL, dfR).
# The quadratic coefficient `a` is computed on-the-fly in the kernel.

@inline @generated function _eval_nd_quad_cell(
        partials::AbstractArray{Tv, NP1},
        indices::NTuple{N, Int},
        inv_hs::NTuple{N, Tg},
        dLs::Tuple{Vararg{Number, N}},
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N, NP1}
    NP1 == N + 1 || error("NP1 must equal N+1")

    stmts = Expr[]

    # Unpack tuples using destructuring. Quadratic reads only `inv_h`/`dL` — the
    # cell width `h` never reaches `_quadratic_kernel_nd` (physical-coord form).
    for (prefix, source) in [
            ("idx_", :indices), ("inv_h_", :inv_hs),
            ("dL_", :dLs), ("op_", :ops),
        ]
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
