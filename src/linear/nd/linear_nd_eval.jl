# ========================================
# LinearInterpolantND Evaluation
# ========================================
#
# Evaluation logic for N-dimensional multilinear interpolation.
# Supports scalar, vector, and batch (SoA/AoS) queries.
#
# Key Algorithm: Tensor-product linear interpolation
# - Sum over 2^N corners with weights determined by normalized coordinates
# - Weights: ∏ᵢ (1-αᵢ if corner_bit=0, αᵢ if corner_bit=1)

# ========================================
# Callable Interface
# ========================================

# Scalar tuple query
@inline function (itp::LinearInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    resolved = map(_resolve_grididx, query, itp.grids)
    ops = _resolve_deriv_nd(deriv, Val(N))
    policies = _resolve_search_nd(search, Val(N))
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _scalar_mono(hint, Val(N))
    return _eval_linear_nd(itp, resolved, ops, policies, hints, mono)
end

# In-place batch evaluation (SoA + AoS) is handled by the unified
# AbstractInterpolantND callable in nd_interpolant_protocol.jl.
# Zero-fill for 2nd+ derivatives is handled by _deriv_zero_fill trait below.

# Derivative zero-fill trait: linear has zero 2nd+ derivative
@inline _deriv_zero_fill(::LinearInterpolantND, ops::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N} =
    _has_second_or_higher_derivative(ops, Val(N))

# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================

# Generic N-dimensional
@inline function _locate_cell(
        itp::LinearInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, policies, hints, mono)
    hs, αs = _compute_linear_params(q_eval, itp.spacings, indices, Ls, Val(N))
    return (itp.data, indices, hs, αs)
end

# N=2 specialization: direct destructuring eliminates ntuple closure overhead
@inline function _locate_cell(
        itp::LinearInterpolantND{Tg, Tv, 2},
        query::Tuple{Vararg{Real, 2}},
        policies::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
        hints::Tuple{Base.RefValue{Int}, Base.RefValue{Int}},
        mono::Tuple{Bool, Bool},
    ) where {Tg, Tv}
    x_eval, y_eval, ix, iy, xL, yL = _locate_cell_2d_preamble(
        query, itp.grids, itp.spacings, itp.extraps, policies, hints, mono
    )

    hx = _get_h(itp.spacings[1], ix)
    hy = _get_h(itp.spacings[2], iy)
    αx = (x_eval - xL) / hx
    αy = (y_eval - yL) / hy

    return (itp.data, (ix, iy), (hx, hy), (αx, αy))
end

# Evaluate kernel at a pre-located cell with given derivative ops
@inline function _eval_at_cell(
        itp::LinearInterpolantND{Tg, Tv, N},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tg, Tv, N}
    if _has_second_or_higher_derivative(ops, Val(N))
        return 0 * first(itp.data)
    end
    data, indices, hs, αs = cell
    return _multilinear_sum(data, indices, hs, αs, ops, Val(N))
end

# ========================================
# Core Evaluation Logic
# ========================================

# Zero-ref for fill-value derivative computation (duck-typed zero via 0 * data_element)
@inline _zero_ref(itp::LinearInterpolantND) = @inbounds first(itp.data)

# Generic N-dimensional version (uses _locate_cell + _eval_at_cell)
@inline function _eval_linear_nd(
        itp::LinearInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    _validate_nd_domain(itp.grids, query, itp.extraps)
    oob_result = _try_fill_oob(query, itp.grids, itp.extraps, ops, _zero_ref(itp))
    oob_result !== nothing && return oob_result
    if _has_second_or_higher_derivative(ops, Val(N))
        return 0 * first(itp.data)
    end
    cell = _locate_cell(itp, query, policies, hints, mono)
    return _eval_at_cell(itp, cell, ops)
end

# N=2 specialization: dispatches to N=2 _locate_cell via type
@inline function _eval_linear_nd(
        itp::LinearInterpolantND{Tg, Tv, 2},
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
    if op_x isa EvalDeriv2 || op_x isa EvalDeriv3 || op_y isa EvalDeriv2 || op_y isa EvalDeriv3
        return 0 * first(itp.data)
    end
    cell = _locate_cell(itp, query, policies, hints, mono)
    return _eval_at_cell(itp, cell, ops)
end

# ========================================
# Derivative Check
# ========================================

@inline function _has_second_or_higher_derivative(ops::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N}
    for d in 1:N
        @inbounds if !(ops[d] isa EvalValue) && !(ops[d] isa EvalDeriv1)
            return true
        end
    end
    return false
end

# ========================================
# Local Parameter Computation
# ========================================

# Shared `αs` formula for both variants. `map` on NTuples dispatches
# per-element with concrete types — safe for heterogeneous `hs` / `Ls` /
# `q_eval` tuples (e.g., Real + Float64 mixes), avoiding the Union-box risk
# of `ntuple(d -> q_eval[d], Val(N))` where runtime-indexed lookup collapses
# to an abstract return type. See MEMORY.md "ND Constructor Inferrability
# Pattern" for the canonical precedent.
@inline _alpha_of(q::Real, L::Real, h::Real) = (q - L) / h
@inline _alphas_from_hs(q_eval, Ls, hs) = map(_alpha_of, q_eval, Ls, hs)

"""
    _compute_linear_params(q_eval, spacings, indices, Ls, Val(N)) -> (hs, αs)

Cell widths and normalized coordinates for multilinear interpolation via
spacing lookup. Used by persistent ND paths where `spacings` is precomputed.

Shares the `αs` formula with `_compute_linear_params_lr` via `_alphas_from_hs`;
only the `hs` derivation (spacing lookup) is variant-specific. Uses `map` over
the `(spacings, indices)` tuple pair to avoid Union boxing when the axes carry
heterogeneous spacing concrete types (ScalarSpacing + VectorSpacing mix).
"""
@inline function _compute_linear_params(
        q_eval::Tuple{Vararg{Real, N}},
        spacings::Tuple{Vararg{AbstractGridSpacing, N}},
        indices::NTuple{N, Int},
        Ls::Tuple{Vararg{Real, N}},
        ::Val{N}
    ) where {N}
    hs = map(_get_h, spacings, indices)
    return (hs, _alphas_from_hs(q_eval, Ls, hs))
end

"""
    _compute_linear_params_lr(q_eval, Ls, Rs, Val(N)) -> (hs, αs)

Pair-variant (Phase 6) — computes per-axis cell width directly from
`Rs[d] - Ls[d]` to avoid the idx-based spacing lookup (which is out-of-bounds
for the seam cell at `idx_L == n` on periodic-exclusive axes — there's no
`spacing.widths[n]`). For uniform Range grids via ScalarSpacing the cost is
unchanged (1 subtraction vs 1 field load). For Vector axes this matches what
VectorSpacing would return for interior cells and Just Works for the seam cell.

Shares the `αs` formula with `_compute_linear_params` via `_alphas_from_hs`.
`map(-, Rs, Ls)` dispatches per-element with concrete types (no Union-box risk
from heterogeneous `Rs` / `Ls` tuples; matches MEMORY.md's inferrability rules).
"""
@inline function _compute_linear_params_lr(
        q_eval::Tuple{Vararg{Real, N}},
        Ls::Tuple{Vararg{Real, N}},
        Rs::Tuple{Vararg{Real, N}},
        ::Val{N}
    ) where {N}
    hs = map(-, Rs, Ls)
    return (hs, _alphas_from_hs(q_eval, Ls, hs))
end

# ========================================
# Multilinear Interpolation Kernel
# ========================================

# Shared Expr builder for the two multilinear generators below. Both variants
# produce an identical flat 2^N straight-line unroll; only the per-axis corner
# addressing differs. `make_idx_expr(d, bit)` returns the address expression
# for axis `d` at corner bit `bit` ∈ {0, 1}:
#   non-pair : (d, bit) -> :(indices[$d] + $bit)
#   pair     : (d, bit) -> :(indices_pairs[$d][$(bit + 1)])
# Runs at compile time only (inside @generated); closure cost is free.
function _multilinear_sum_body(N::Int, make_idx_expr::Function)
    num_corners = 1 << N  # 2^N
    corner_exprs = []
    for corner in 0:(num_corners - 1)
        bits = ntuple(d -> (corner >> (d - 1)) & 1, N)
        idx_expr = Expr(:tuple, [make_idx_expr(d, bits[d]) for d in 1:N]...)
        weight_exprs = [
            :(_linear_weight(ops[$d], αs[$d], hs[$d], Val($(bits[d]))))
            for d in 1:N
        ]
        weight_expr = foldl((a, b) -> :($a * $b), weight_exprs)
        push!(corner_exprs, :(@inbounds data[$idx_expr...] * $weight_expr))
    end
    sum_expr = foldl((a, b) -> :($a + $b), corner_exprs)
    return quote
        Base.@_inline_meta
        @inbounds $sum_expr
    end
end

"""
    _multilinear_sum(data, indices, hs, αs, ops, Val(N))

Compute the multilinear interpolation sum over 2^N corners.

For each corner indexed by bit pattern b ∈ {0,1}^N:
- Corner index: (indices[1]+b₁, indices[2]+b₂, ..., indices[N]+bₙ)
- Weight: ∏ᵢ _linear_weight(ops[i], αs[i], hs[i], bᵢ)

The weight function depends on the evaluation operation:
- EvalValue: (1-α) if b=0, α if b=1
- EvalDeriv1: -1/h if b=0, 1/h if b=1

Shares body with `_multilinear_sum_lr` via `_multilinear_sum_body`.
"""
@generated function _multilinear_sum(
        data::AbstractArray{Tv, N},
        indices::NTuple{N, Int},
        hs::NTuple{N},
        αs::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        ::Val{N}
    ) where {Tv, N}
    return _multilinear_sum_body(N, (d, bit) -> :(indices[$d] + $bit))
end

"""
    _multilinear_sum_lr(data, indices_pairs, hs, αs, ops, Val(N))

Pair-valued indices variant for zero-copy periodic ND evaluation (Phase 6).

`indices_pairs[d] = (idx_L_d, idx_R_d)` — for each corner bit pattern
`b ∈ {0,1}^N`, corner address on axis `d` is `indices_pairs[d][b_d + 1]`
(bit 0 → left `idx_L_d`, bit 1 → right `idx_R_d`).

Distinct name (not an overload) because at `N=0` `NTuple{0, Int}` and
`NTuple{0, NTuple{2, Int}}` both collapse to `Tuple{}`, making
overload-style dispatch ambiguous (caught by Aqua static analysis). Used
only by periodic ND oneshot — non-periodic/persistent callers stay on
`_multilinear_sum`.

Shares body with `_multilinear_sum` via `_multilinear_sum_body`; only the
corner-address expression differs — `indices[d] + bit` → `indices_pairs[d][bit + 1]`.
"""
@generated function _multilinear_sum_lr(
        data::AbstractArray{Tv, N},
        indices_pairs::NTuple{N, NTuple{2, Int}},
        hs::NTuple{N},
        αs::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        ::Val{N}
    ) where {Tv, N}
    return _multilinear_sum_body(N, (d, bit) -> :(indices_pairs[$d][$(bit + 1)]))
end

# ========================================
# Linear Weight Functions
# ========================================

# For value evaluation: (1-α) for bit=0, α for bit=1
@inline _linear_weight(::EvalValue, α, h, ::Val{0}) = one(α) - α
@inline _linear_weight(::EvalValue, α, h, ::Val{1}) = α

# For first derivative: -1/h for bit=0, 1/h for bit=1
@inline _linear_weight(::EvalDeriv1, α, h, ::Val{0}) = -inv(h)
@inline _linear_weight(::EvalDeriv1, α, h, ::Val{1}) = inv(h)

# Second and higher derivatives are zero (handled by early return in _eval_linear_nd)
# But define them for completeness if somehow called
@inline _linear_weight(::EvalDeriv2, α, h, ::Val{B}) where {B} = zero(α)
@inline _linear_weight(::EvalDeriv3, α, h, ::Val{B}) where {B} = zero(α)

# Generic fallback: N-th derivative weight is zero for N ≥ 2
@inline _linear_weight(::DerivOp{N}, α, h, ::Val{B}) where {N, B} = zero(α)
