# ========================================
# Heterogeneous ND Eval Kernel (@generated)
# ========================================
# Same staging topology as _eval_nd_cell (cubic) but with per-axis kernel dispatch.
# Each stage uses the 1D kernel matching the axis method:
#   CubicInterp    → _hermite_kernel_1d(op, fL, fR, dfL, dfR, h, inv_h, dL)
#   QuadraticInterp → _quadratic_kernel_nd(op, fL, fR, dfL, inv_h, dL)
#   LinearInterp   → _linear_kernel(op, fL, fR, inv_h, dL)
#   ConstantInterp → nearest-neighbor select
#
# For non-derivative axes (Linear/Constant), the derivative slots in partials
# contain identity copies (from build step). The staging topology is unchanged —
# redundant intermediate computations occur but produce correct results.

"""
    _eval_hetero_nd_cell(partials, indices, hs, inv_hs, dLs, ops, methods)

Evaluate the heterogeneous tensor-product kernel at a single cell.

Uses @generated to produce straight-line code (no branches, no allocations).
Dispatches per-axis 1D kernel at compile time based on the methods tuple type.
"""
@inline @generated function _eval_hetero_nd_cell(
        partials::Array{Tv, NP1},
        indices::NTuple{N, Int},
        hs::NTuple{N, Tg},
        inv_hs::NTuple{N, Tg},
        dLs::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        ::M,
    ) where {Tv, Tg, N, NP1, M <: Tuple{Vararg{AbstractInterpMethod, N}}}
    NP1 == N + 1 || error("NP1 must equal N+1")

    stmts = Expr[]

    # Unpack tuples using destructuring
    for (prefix, source) in [
            ("idx_", :indices), ("h_", :hs), ("inv_h_", :inv_hs),
            ("dL_", :dLs), ("op_", :ops),
        ]
        syms = ntuple(d -> Symbol(prefix, d), N)
        lhs = Expr(:tuple, syms...)
        push!(stmts, :($lhs = $source))
    end

    # Collapse each dimension (same staging topology as _eval_nd_cell)
    for stage in 1:N
        num_corners = 1 << (N - stage)
        num_derivs = 1 << (N - stage)
        method_type = M.parameters[stage]

        for corner in 0:(num_corners - 1)
            for deriv in 0:(num_derivs - 1)
                out_var = _varname(stage, corner, deriv)

                if stage == 1
                    # Read from partials array
                    function make_partial_access(c_dim1::Int, d_dim1::Int)
                        corner_full = c_dim1 | (corner << 1)
                        deriv_full = d_dim1 | (deriv << 1)
                        p_idx = _partial_index(deriv_full)
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
                    prev = stage - 1
                    fL = _varname(prev, 0 | (corner << 1), 0 | (deriv << 1))
                    fR = _varname(prev, 1 | (corner << 1), 0 | (deriv << 1))
                    dfL = _varname(prev, 0 | (corner << 1), 1 | (deriv << 1))
                    dfR = _varname(prev, 1 | (corner << 1), 1 | (deriv << 1))
                end

                h = Symbol("h_", stage)
                inv_h = Symbol("inv_h_", stage)
                dL = Symbol("dL_", stage)
                op = Symbol("op_", stage)

                # Per-axis kernel dispatch at compile time
                kernel_call = if method_type <: CubicInterp
                    :(_hermite_kernel_1d($op, $fL, $fR, $dfL, $dfR, $h, $inv_h, $dL))
                elseif method_type <: QuadraticInterp
                    :(_quadratic_kernel_nd($op, $fL, $fR, $dfL, $inv_h, $dL))
                elseif method_type <: LinearInterp
                    :(_linear_kernel($op, $fL, $fR, $inv_h, $dL))
                elseif method_type <: ConstantInterp
                    # Nearest-neighbor: pick left or right based on position
                    :(ifelse($dL < $h * $(Tg(0.5)), $fL, $fR))
                else
                    error("Unsupported method type: $method_type")
                end

                push!(stmts, :($out_var = $kernel_call))
            end
        end
    end

    # Final result
    final_var = _varname(N, 0, 0)
    push!(stmts, :(return $final_var))

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
    end
end

# ========================================
# PreCompute Eval Entry Point
# ========================================

@inline function _eval_tensor_product_precomputed(
        itp_data::NodalDerivativesND{Tv, N},
        grids::NTuple{N, AbstractVector{Tg}},
        spacings,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extraps,
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        searches::NTuple{N, AbstractSearchPolicy},
        hints,
    ) where {Tv, Tg, N}
    # Handle extrapolation
    q_eval = _handle_all_extraps(query, grids, extraps)

    # Cell location
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches, hints)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)

    # Evaluate kernel
    return _eval_hetero_nd_cell(itp_data.partials, indices, hs, inv_hs, dLs, ops, methods)
end
