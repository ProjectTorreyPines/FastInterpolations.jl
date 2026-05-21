# ========================================
# Generic ND Hermite Evaluation
# ========================================
#
# N-dimensional cubic Hermite interpolation with:
# - Tg/Tv type separation (grid vs value types)
# - @generated tensor product for zero-allocation O(1) evaluation
# - N=2 specialization for optimal batch performance
# - AD support (query type preserved through evaluation)

const _DEBUG_GENERATED_CELL = Ref(false)  # Debug: inspect @generated code

# ========================================
# CALLABLE INTERFACE
# ========================================

"""
    (itp::CubicInterpolantND)(query; deriv=EvalValue(), search=itp.searches)

Evaluate N-dimensional cubic Hermite interpolant.

# Keywords
- `deriv`: Derivative specification
  - `DerivOp`: same order for all axes (fastest), e.g. `DerivOp(1)`
  - `NTuple{N,DerivOp}`: per-axis orders, e.g. `(DerivOp(1), EvalValue())` for ∂f/∂x
- `search`: Override search policy (single or per-axis tuple)

# Examples
```julia
itp((1.0, 0.5))                                  # value
itp((1.0, 0.5); deriv=DerivOp(1))                # all first derivatives
itp((1.0, 0.5); deriv=(DerivOp(1), EvalValue()))  # ∂f/∂x only
```
"""
# Single-point evaluation
@inline function (itp::CubicInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Real, N}};  # Allow Real, Dual (AD), and GridIdx
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    # Resolve bare GridIdx → GridIdx{Tg}(idx, val). No-op for Real/Dual.
    resolved = map(_resolve_grididx, query, itp.grids)
    ops = _resolve_deriv_nd(deriv, Val(N))
    policies = _resolve_search_nd(search, Val(N))
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _scalar_mono(hint, Val(N))
    return _eval_nd_at_point(itp, resolved, ops, policies, hints, mono)
end

# In-place batch evaluation (SoA + AoS) is handled by the unified
# AbstractInterpolantND callable in nd_interpolant_protocol.jl.
# Scalar evaluation routes through the generic `_eval_nd_at_point` defined
# there — Cubic exposes only the `_locate_cell` + `_eval_at_cell` traits.

# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================
#
# _locate_cell: extrapolation + interval search + local params → cell tuple
# _eval_at_cell: kernel-only evaluation with pre-located cell
#
# Factoring the eval pipeline into locate/eval phases enables vector calculus
# functions (gradient, hessian, laplacian) to search intervals ONCE and
# evaluate the kernel multiple times with different derivative ops.

# Generic N-dimensional. `extraps` is the per-axis effective extrap tuple —
# batch callers pass an InBounds-promoted version from `_check_domain_nd`;
# scalar callers go through the 5-arg forwarder which injects `itp.extraps`.
@inline function _locate_cell(
        itp::CubicInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    q_evals = _handle_all_extraps(query, itp.grids, extraps)
    indices, Ls, _ = _search_all_intervals(q_evals, itp.grids, policies, hints, mono)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, itp.grids, indices, Ls)

    return (itp.nodal_derivs.partials, indices, hs, inv_hs, dLs)
end

# N=2 specialization: direct destructuring eliminates ntuple closure overhead
@inline function _locate_cell(
        itp::CubicInterpolantND{Tg, Tv, 2},
        query::Tuple{Vararg{Real, 2}},
        extraps::Tuple{AbstractExtrap, AbstractExtrap},
        policies::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
        hints::Tuple{Base.RefValue{Int}, Base.RefValue{Int}},
        mono::Tuple{Bool, Bool},
    ) where {Tg, Tv}
    x_eval, y_eval, ix, iy, xL, yL = _locate_cell_2d_preamble(
        query, itp.grids, extraps, policies, hints, mono
    )

    hx = _get_h(itp.grids[1], ix);  hy = _get_h(itp.grids[2], iy)
    inv_hx = _get_inv_h(itp.grids[1], ix); inv_hy = _get_inv_h(itp.grids[2], iy)
    dLx = x_eval - xL;  dLy = y_eval - yL

    return (itp.nodal_derivs.partials, (ix, iy), (hx, hy), (inv_hx, inv_hy), (dLx, dLy))
end

# Evaluate kernel at a pre-located cell with given derivative ops
@inline function _eval_at_cell(
        ::CubicInterpolantND,
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp}
    ) where {N}
    partials, indices, hs, inv_hs, dLs = cell
    return _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
end

# Zero-ref for fill-value derivative computation (duck-typed zero via 0 * data_element)
@inline _value_sample(itp::CubicInterpolantND) = @inbounds first(itp.nodal_derivs.partials)

# ========================================
# @GENERATED TENSOR PRODUCT KERNEL
# ========================================
# Collapses N dimensions via sequential 1D Hermite interpolations.
# Each stage reduces 2^(N-d+1) → 2^(N-d) values.

# _varname, _partial_index, _corner_offset_expr → core/nd_utils.jl (shared with quadratic)
@inline @generated function _eval_nd_cell(
        partials::AbstractArray{Tv, NP1},
        indices::NTuple{N, Int},
        hs::NTuple{N, Tg},
        inv_hs::NTuple{N, Tg},
        dLs::Tuple{Vararg{Real, N}},  # Allow heterogeneous Real types (AD support)
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N, NP1}
    # Validate dimensions
    NP1 == N + 1 || error("NP1 must equal N+1")

    # Generate all statements
    stmts = Expr[]

    # Unpack tuples using destructuring (efficient AST)
    for (prefix, source) in [
            ("idx_", :indices), ("h_", :hs), ("inv_h_", :inv_hs),
            ("dL_", :dLs), ("op_", :ops),
        ]
        syms = ntuple(d -> Symbol(prefix, d), N)
        lhs = Expr(:tuple, syms...)
        push!(stmts, :($lhs = $source))
    end

    # Collapse each dimension
    for stage in 1:N
        # After collapsing dim 'stage', we have 2^(N-stage) corners and derivs
        num_corners = 1 << (N - stage)
        num_derivs = 1 << (N - stage)

        for corner in 0:(num_corners - 1)
            for deriv in 0:(num_derivs - 1)
                out_var = _varname(stage, corner, deriv)

                if stage == 1
                    # Read from partials array
                    function make_partial_access(c_dim1::Int, d_dim1::Int)
                        corner_full = c_dim1 | (corner << 1)
                        deriv_full = d_dim1 | (deriv << 1)
                        p_idx = _partial_index(deriv_full)

                        # Build index expression: partials[p_idx, idx_1 + off_1, ...]
                        offsets = _corner_offset_expr(corner_full, N)
                        idx_exprs = [:($(Symbol("idx_", d)) + $(offsets[d])) for d in 1:N]
                        return :(partials[$p_idx, $(idx_exprs...)])
                    end

                    fL = make_partial_access(0, 0)
                    fR = make_partial_access(1, 0)
                    dfL = make_partial_access(0, 1)
                    dfR = make_partial_access(1, 1)
                else
                    # Read from previous stage variables
                    prev_stage = stage - 1
                    fL = _varname(prev_stage, 0 | (corner << 1), 0 | (deriv << 1))
                    fR = _varname(prev_stage, 1 | (corner << 1), 0 | (deriv << 1))
                    dfL = _varname(prev_stage, 0 | (corner << 1), 1 | (deriv << 1))
                    dfR = _varname(prev_stage, 1 | (corner << 1), 1 | (deriv << 1))
                end

                h = Symbol("h_", stage)
                inv_h = Symbol("inv_h_", stage)
                dL = Symbol("dL_", stage)
                op = Symbol("op_", stage)

                kernel_call = :(_hermite_kernel_1d($op, $fL, $fR, $dfL, $dfR, $h, $inv_h, $dL))
                push!(stmts, :($out_var = $kernel_call))
            end
        end
    end

    # Final result is g_{N}_{0}_{0}
    final_var = _varname(N, 0, 0)
    push!(stmts, :(return $final_var))

    # Wrap in quote block with @inbounds and @inline_meta
    result = quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
    end

    # Debug output (controlled by _DEBUG_GENERATED_CELL flag)
    if _DEBUG_GENERATED_CELL[]
        function count_ast_nodes(ex)
            if ex isa Expr
                return 1 + sum(count_ast_nodes(arg) for arg in ex.args; init = 0)
            else
                return 1
            end
        end

        println("="^60)
        println("Generated _eval_nd_cell for N=$N, Tv=$Tv, Tg=$Tg")
        println("AST nodes: ", count_ast_nodes(result))
        println("="^60)
        println(Base.remove_linenums!(deepcopy(result)))
        println("="^60)
    end

    return result
end
