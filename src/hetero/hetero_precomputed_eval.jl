# ========================================
# Heterogeneous ND Eval Kernel (@generated) — Compact Storage
# ========================================
# Uses mixed-radix indexing for compact partial derivative storage.
# sizes[d] = 2 for derivative axes (Cubic/Quadratic), 1 for others.
# Total partials = prod(sizes) ≤ 2^N.
#
# Staging topology:
# - num_corners at stage s = 2^(N-s) (spatial, always halves)
# - num_derivs at stage s = prod(sizes[s+1:N]) (compact, only derivative axes contribute)
#
# Partial index (column-major mixed-radix):
#   p = 1 + d₁ + sizes[1]*(d₂ + sizes[2]*(d₃ + ...))
# where dₖ ∈ {0, ..., sizes[k]-1}.

"""
    _eval_hetero_nd_cell(partials, indices, hs, inv_hs, dLs, ops, methods)

Evaluate the heterogeneous tensor-product kernel at a single cell.

Uses @generated with compact mixed-radix partial indexing.
Dispatches per-axis 1D kernel at compile time based on the methods tuple type.
Non-derivative axes (Linear/Constant) produce fewer intermediates (no derivative entries).
"""
@inline @generated function _eval_hetero_nd_cell(
        partials::AbstractArray{Tv, NP1},
        indices::NTuple{N, Int},
        hs::NTuple{N, Tg},
        inv_hs::NTuple{N, Tg},
        dLs::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        ::M,
    ) where {Tv, Tg, N, NP1, M <: Tuple{Vararg{AbstractInterpMethod, N}}}
    NP1 == N + 1 || error("NP1 must equal N+1")

    sizes = _deriv_sizes(M)
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

    # Collapse each dimension with compact derivative tracking
    for stage in 1:N
        num_corners = 1 << (N - stage)
        # Compact derivative count for remaining dims AFTER this stage
        compact_derivs_remaining = prod(sizes[(stage + 1):N]; init = 1)
        method_type = M.parameters[stage]

        for corner in 0:(num_corners - 1)
            for deriv in 0:(compact_derivs_remaining - 1)
                out_var = _varname(stage, corner, deriv)

                if stage == 1
                    # Read from partials: compact mixed-radix indexing
                    # p = 1 + d₁ + sizes[1] * compact_remaining_index
                    function make_compact_partial_access(c_dim1::Int, d_dim1::Int)
                        corner_full = c_dim1 | (corner << 1)
                        offsets = _corner_offset_expr(corner_full, N)
                        idx_exprs = [:($(Symbol("idx_", d)) + $(offsets[d])) for d in 1:N]
                        p_idx = 1 + d_dim1 + sizes[1] * deriv
                        return :(partials[$p_idx, $(idx_exprs...)])
                    end

                    fL = make_compact_partial_access(0, 0)
                    fR = make_compact_partial_access(1, 0)
                    if sizes[1] == 2
                        dfL = make_compact_partial_access(0, 1)
                        dfR = make_compact_partial_access(1, 1)
                    end
                else
                    # Read from previous stage variables
                    # prev has compact_derivs = sizes[stage] * compact_derivs_remaining
                    prev = stage - 1
                    s_d = sizes[stage]

                    # column-major: deriv_prev = d_stage + sizes[stage] * deriv
                    fL = _varname(prev, 0 | (corner << 1), 0 + s_d * deriv)
                    fR = _varname(prev, 1 | (corner << 1), 0 + s_d * deriv)
                    if s_d == 2
                        dfL = _varname(prev, 0 | (corner << 1), 1 + s_d * deriv)
                        dfR = _varname(prev, 1 | (corner << 1), 1 + s_d * deriv)
                    end
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
                    # New `_linear_kernel` signature takes α (= dL * inv_h)
                    # instead of dL. Hetero path is fully cached, so the
                    # extra mul is one register op — no DCE needed here.
                    :(_linear_kernel($op, $fL, $fR, $inv_h, $dL * $inv_h))
                elseif method_type <: ConstantInterp
                    side_inst = fieldtype(method_type, :side)()
                    :(_constant_kernel($op, $fL, $fR, $h, $dL, $side_inst))
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

@inline function _eval_hetero_precomputed(
        itp_data::_HeteroPartials{Tv, N},
        grids::NTuple{N, AbstractVector{Tg}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extraps,
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tv, Tg, N}
    # Handle extrapolation
    q_eval = _handle_all_extraps(query, grids, extraps)

    # Cell location
    indices, Ls, _ = _search_all_intervals(q_eval, grids, policies, hints, mono)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, grids, indices, Ls)

    # Evaluate kernel with compact partials
    return _eval_hetero_nd_cell(itp_data.partials, indices, hs, inv_hs, dLs, ops, methods)
end
