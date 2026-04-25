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
    # Persistent fast lane: 2-arg `_get_h` reads precomputed `spacings[d].h`
    # (or `.h[i]` for Vector). `map` dispatches per-axis with concrete types.
    hs = map(_get_h, itp.spacings, indices)
    αs = map(_alpha_of, q_eval, Ls, hs)
    # Wrap raw indices into the unified stencil shape so `_multilinear_sum`
    # has a single signature across persistent / BC oneshot callers.
    # Non-periodic cells are always `(idx, idx+1)` — no seam wrap.
    stencils = map(i -> _IdxPair(i, i + 1), indices)
    return (itp.data, stencils, hs, αs)
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

    # N=2 specialization wraps directly into the unified stencil shape — same
    # `_multilinear_sum` signature as the generic-N persistent / BC oneshot paths.
    return (itp.data, (_IdxPair(ix, ix + 1), _IdxPair(iy, iy + 1)), (hx, hy), (αx, αy))
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
    data, stencils, hs, αs = cell
    return _multilinear_sum(data, stencils, hs, αs, ops, Val(N))
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
#
# `_alpha_of` is the only surviving named helper from this region. The earlier
# `_alphas_from_hs(q, Ls, hs)`, `_compute_linear_params(q, spacings, indices, Ls, …)`,
# and `_compute_linear_params_stencil(q, Ls, Rs, …)` wrappers were single-line
# `map`-forwarders that hid the actual arithmetic behind two extra layers; call
# sites now do the `map(_get_h, …)` + `map(_alpha_of, q, Ls, hs)` directly.
#   - persistent path: `map(_get_h, spacings, indices)` (2-arg cached fast lane)
#   - BC oneshot path: `map(_get_h, grids, Ls, Rs)` (3-arg dispatch, seam-aware)
@inline _alpha_of(q::Real, L::Real, h::Real) = (q - L) / h

# ========================================
# Multilinear Interpolation Kernel
# ========================================

"""
    _multilinear_sum(data, stencils, hs, αs, ops, Val(N))

Compute the multilinear interpolation sum over 2^N corners.

`stencils[d]::_IdxStencil{2}` carries `(idx_L_d, idx_R_d)` — for each corner
bit pattern `b ∈ {0,1}^N`, the corner address on axis `d` is
`stencils[d][b_d + 1]` (bit 0 → left `idx_L_d`, bit 1 → right `idx_R_d`).
For non-periodic cells `idx_R == idx_L + 1`; for periodic-exclusive seam
cells `idx_R == 1` (wrap), so the kernel reads the wrapped neighbor without
any data extension.

Per-corner weight: `∏ᵢ _linear_weight(ops[i], αs[i], hs[i], bᵢ)`. The weight
function depends on the evaluation op — EvalValue: `(1-α)` for `b=0`, `α`
for `b=1`; EvalDeriv1: `-1/h` for `b=0`, `+1/h` for `b=1`.

Single-overload, stencil-only kernel. Persistent callers wrap their
single-index `indices` via `map(i -> _IdxPair(i, i+1), indices)` at the
call site before invoking; BC oneshot callers receive seam-aware stencils
directly from `_search_all_intervals_stencil`.
"""
@generated function _multilinear_sum(
        data::AbstractArray{Tv, N},
        stencils::NTuple{N, _IdxStencil{2}},
        hs::NTuple{N},
        αs::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        ::Val{N}
    ) where {Tv, N}
    return _multilinear_sum_body(N, (d, bit) -> :(stencils[$d][$(bit + 1)]))
end

# Shared Expr builder for the multilinear @generated kernel above. Produces
# the flat 2^N straight-line unroll. `make_idx_expr(d, bit)` returns the
# address expression for axis `d` at corner bit `bit` ∈ {0, 1}; the kernel
# above passes the stencil-style `(d, bit) -> :(stencils[$d][$(bit + 1)])`.
# Body kept as a callback-driven helper so future K > 2 kernels (ND Hermite)
# can reuse it without duplicating the corner-unroll skeleton.
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
@generated function _multilinear_sum_stencil(
        data::AbstractArray{Tv, N},
        stencils::NTuple{N, _IdxStencil{2}},
        hs::NTuple{N},
        αs::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        ::Val{N}
    ) where {Tv, N}
    return _multilinear_sum_body(N, (d, bit) -> :(stencils[$d][$(bit + 1)]))
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
