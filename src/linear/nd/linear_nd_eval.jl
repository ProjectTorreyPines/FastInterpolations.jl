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
    return _eval_nd_at_point(itp, resolved, ops, policies, hints, mono)
end

# In-place batch evaluation (SoA + AoS) is handled by the unified
# AbstractInterpolantND callable in nd_interpolant_protocol.jl. Scalar
# evaluation routes through the generic `_eval_nd_at_point` there.
# Linear's "deriv ≥ 2 → 0" rule is handled inside the kernel itself
# (`_linear_weight(::EvalDeriv2+, ...) = zero(α)` — see below) so the
# cell-local corner sum produces a carrier-aware zero without any
# protocol-level short-circuit.

# ========================================
# CELL LOCATION (locate once, evaluate many)
# ========================================

# Generic N-dimensional. `extraps` is the per-axis effective extrap tuple —
# batch callers pass InBounds-promoted; scalar callers route through the
# 5-arg forwarder (interpolant_protocol.jl) which injects `itp.extraps`.
@inline function _locate_cell(
        itp::LinearInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    q_eval = _handle_all_extraps(query, itp.grids, extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, policies, hints, mono)
    inv_hs = map(_get_inv_h, itp.grids, indices)
    αs = map(_alpha_of, q_eval, Ls, inv_hs)
    stencils = map(i -> _IdxPair(i, i + 1), indices)
    return (itp.data, stencils, inv_hs, αs)
end

# N=2 specialization: direct destructuring eliminates ntuple closure overhead
@inline function _locate_cell(
        itp::LinearInterpolantND{Tg, Tv, 2},
        query::Tuple{Vararg{Real, 2}},
        extraps::Tuple{AbstractExtrap, AbstractExtrap},
        policies::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
        hints::Tuple{Base.RefValue{Int}, Base.RefValue{Int}},
        mono::Tuple{Bool, Bool},
    ) where {Tg, Tv}
    x_eval, y_eval, ix, iy, xL, yL = _locate_cell_2d_preamble(
        query, itp.grids, extraps, policies, hints, mono
    )

    inv_hx = _get_inv_h(itp.grids[1], ix)
    inv_hy = _get_inv_h(itp.grids[2], iy)
    αx = (x_eval - xL) * inv_hx
    αy = (y_eval - yL) * inv_hy
    return (itp.data, (_IdxPair(ix, ix + 1), _IdxPair(iy, iy + 1)), (inv_hx, inv_hy), (αx, αy))
end

# Evaluate kernel at a pre-located cell with given derivative ops.
# Deriv ≥ 2 → 0 is handled inside the kernel itself via
# `_linear_weight(::EvalDeriv2+, …) = zero(α)`, so the corner sum produces a
# cell-local, carrier-aware zero without a protocol-level short-circuit.
@inline function _eval_at_cell(
        itp::LinearInterpolantND{Tg, Tv, N},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tg, Tv, N}
    data, stencils, inv_hs, αs = cell
    return _multilinear_sum(data, stencils, inv_hs, αs, ops, Val(N))
end

# Per-method sample of `Tv` for fill-value paths (e.g. `_try_fill_oob`).
@inline _sample_data(itp::LinearInterpolantND) = @inbounds first(itp.data)

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

# `_alpha_of` (normalized cell coordinate, shared by 1D + ND) is defined in
# linear_kernels.jl. The `_ExclusivePeriodicAxis` overload lives in periodic_axis.jl.

# ========================================
# Multilinear Interpolation Kernel
# ========================================

"""
    _multilinear_sum(data, stencils, inv_hs, αs, ops, Val(N))

Compute the multilinear interpolation sum over 2^N corners.

`stencils[d]::_IdxStencil{2}` carries `(idx_L_d, idx_R_d)` — for each corner
bit pattern `b ∈ {0,1}^N`, the corner address on axis `d` is
`stencils[d][b_d + 1]` (bit 0 → left `idx_L_d`, bit 1 → right `idx_R_d`).
For non-periodic cells `idx_R == idx_L + 1`; for periodic-exclusive seam
cells `idx_R == 1` (wrap), so the kernel reads the wrapped neighbor without
any data extension.

Per-corner weight: `∏ᵢ _linear_weight(ops[i], αs[i], inv_hs[i], bᵢ)`. The
weight function depends on the evaluation op — EvalValue: `(1-α)` for `b=0`,
`α` for `b=1` (inv_h unused); EvalDeriv1: `-inv_h` for `b=0`, `+inv_h` for
`b=1` (precomputed reciprocal — no `inv()` at the kernel).

Single-overload, stencil-only kernel. Persistent callers wrap their
single-index `indices` via `map(i -> _IdxPair(i, i+1), indices)` at the
call site before invoking; BC oneshot callers receive seam-aware stencils
directly from `_search_all_intervals_stencil`.
"""
@generated function _multilinear_sum(
        data::AbstractArray{Tv, N},
        stencils::NTuple{N, _IdxStencil{2}},
        inv_hs::NTuple{N},
        αs::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        ::Val{N}
    ) where {Tv, N}
    num_corners = 1 << N  # 2^N
    corner_exprs = []
    for corner in 0:(num_corners - 1)
        bits = ntuple(d -> (corner >> (d - 1)) & 1, N)
        idx_expr = Expr(:tuple, [:(stencils[$d][$(bits[d] + 1)]) for d in 1:N]...)
        weight_exprs = [
            :(_linear_weight(ops[$d], αs[$d], inv_hs[$d], Val($(bits[d]))))
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

# ========================================
# Nested value kernel (value-only fast path)
# ========================================
# Repeated linear interpolation: collapse one axis at a time via
# `muladd(α, hi − lo, lo)`. Fewer mul-adds than the flat 2^N weight-expansion
# above, and the form cubic/quadratic ND already use. Value-only — every
# derivative / mixed-partial query keeps the flat method (more specific `ops`
# slot: `NTuple{N,EvalValue}` ⊂ `NTuple{N,AbstractEvalOp}`). Corners are read
# through `stencils[d][bit+1]`, so periodic-seam wrap (`idx_R == 1`) is preserved.

# N=2 hand-written: collapse axis 1 then axis 2.
@inline function _multilinear_sum(
        data::AbstractArray{Tv, 2},
        stencils::NTuple{2, _IdxStencil{2}},
        ::NTuple{2},
        αs::Tuple{Vararg{Real, 2}},
        ::Tuple{EvalValue, EvalValue},
        ::Val{2}
    ) where {Tv}
    sx, sy = stencils
    αx, αy = αs
    @inbounds begin
        c00 = data[sx[1], sy[1]]; c10 = data[sx[2], sy[1]]
        c01 = data[sx[1], sy[2]]; c11 = data[sx[2], sy[2]]
        g0 = muladd(αx, c10 - c00, c00)   # bit_y = 0 edge
        g1 = muladd(αx, c11 - c01, c01)   # bit_y = 1 edge
        return muladd(αy, g1 - g0, g0)
    end
end

# Generic-N: @generated staged collapse, mirrors cubic's `_eval_nd_cell` minus the
# derivative corners. Serves N ≥ 3 (N=2 hits the hand-written method above — more
# specific concrete `Val{2}`). Axis `s` is always the lowest remaining corner bit,
# so each stage pairs adjacent positions (even = bit 0 = lo, odd = bit 1 = hi).
@generated function _multilinear_sum(
        data::AbstractArray{Tv, N},
        stencils::NTuple{N, _IdxStencil{2}},
        ::NTuple{N},
        αs::Tuple{Vararg{Real, N}},
        ::NTuple{N, EvalValue},
        ::Val{N}
    ) where {Tv, N}
    stmts = Expr[]
    αsyms = ntuple(d -> Symbol("α_", d), N)
    ssyms = ntuple(d -> Symbol("s_", d), N)
    push!(stmts, :(($(αsyms...),) = αs))
    push!(stmts, :(($(ssyms...),) = stencils))
    # Stage 0: read the 2^N corners.
    num = 1 << N
    cur = Vector{Symbol}(undef, num)
    for c in 0:(num - 1)
        idx = [:($(ssyms[d])[$(((c >> (d - 1)) & 1) + 1)]) for d in 1:N]
        v = Symbol("g0_", c)
        push!(stmts, :($v = data[$(idx...)]))
        cur[c + 1] = v
    end
    # Stages 1..N: collapse one axis per stage.
    for s in 1:N
        α = αsyms[s]
        half = length(cur) ÷ 2
        nxt = Vector{Symbol}(undef, half)
        for j in 0:(half - 1)
            lo = cur[2j + 1]; hi = cur[2j + 2]
            v = Symbol("g", s, "_", j)
            push!(stmts, :($v = muladd($α, $hi - $lo, $lo)))
            nxt[j + 1] = v
        end
        cur = nxt
    end
    return quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
            return $(cur[1])
        end
    end
end

# ========================================
# Linear Weight Functions
# ========================================

# For value evaluation: (1-α) for bit=0, α for bit=1 (inv_h unused)
@inline _linear_weight(::EvalValue, α, inv_h, ::Val{0}) = one(α) - α
@inline _linear_weight(::EvalValue, α, inv_h, ::Val{1}) = α

# For first derivative: -inv_h for bit=0, +inv_h for bit=1. `* one(α)` threads
# Tq even when α does not appear in the weight expression (mixed-partial
# `(D1, D1)` etc.); for Float α the `1.0` factor is const-folded by LLVM.
@inline _linear_weight(::EvalDeriv1, α, inv_h, ::Val{0}) = -inv_h * one(α)
@inline _linear_weight(::EvalDeriv1, α, inv_h, ::Val{1}) = inv_h * one(α)

# Second and higher derivatives: weight is `zero(α)` at every corner. The
# multilinear sum then yields cell-local NaN propagation via IEEE `NaN * 0 = NaN`.
@inline _linear_weight(::EvalDeriv2, α, inv_h, ::Val{B}) where {B} = zero(α)
@inline _linear_weight(::EvalDeriv3, α, inv_h, ::Val{B}) where {B} = zero(α)

# Generic fallback: N-th derivative weight is zero for N ≥ 2
@inline _linear_weight(::DerivOp{N}, α, inv_h, ::Val{B}) where {N, B} = zero(α)
