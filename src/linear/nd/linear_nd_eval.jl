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
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _eval_nd_scalar_query(itp, query, deriv, extrap, search, hint)
end

# In-place batch evaluation (SoA + AoS) is handled by the unified
# AbstractInterpolantND callable in nd_interpolant_protocol.jl. Scalar
# evaluation routes through the generic `_eval_nd_at_point` there.
# Linear's "deriv ≥ 2 → 0" rule is handled inside the kernel itself
# (`_linear_kernel(::EvalDeriv2+, …)` → carrier-aware 0) so the nested
# collapse produces a cell-local, carrier-aware zero without any
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
    # 6-arg search: per-axis `extraps` let InBounds range axes take the lean direct
    # search (one-sided clamp; hint still written back) — bit-identical, per-axis.
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, policies, hints, mono, extraps)
    inv_hs = map(_get_inv_h, itp.grids, indices)
    αs = map(_alpha_of, q_eval, Ls, inv_hs)
    intervals = map(i -> _ExplicitIndices(i, i + 1), indices)
    return (itp.data, intervals, inv_hs, αs)
end

# No N=2 specialization: the generic-N locate above inlines to the same code at
# N=2 (the `map`/closure forms compile away), so a hand-destructured 2D variant
# is equal-or-slower — verified bit-identical via same-process method-swap A/B.

# Evaluate kernel at a pre-located cell with given derivative ops.
# Deriv ≥ 2 → 0 is handled inside the kernel itself via
# `_linear_kernel(::EvalDeriv2+, …)` → carrier-aware 0, so the nested collapse
# produces a cell-local, carrier-aware zero without a protocol-level short-circuit.
@inline function _eval_at_cell(
        itp::LinearInterpolantND{Tg, Tv, N},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tg, Tv, N}
    data, intervals, inv_hs, αs = cell
    return _multilinear_sum(data, intervals, inv_hs, αs, ops, Val(N))
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
    _multilinear_sum(data, intervals, inv_hs, αs, ops, Val(N))

N-dimensional multilinear interpolation by **nested repeated-linear collapse**:
read the 2^N cell corners, then collapse one axis per stage via the shared 1D
kernel `_linear_kernel(ops[s], lo, hi, inv_hs[s], αs[s])`. The op selects the
per-axis operation — EvalValue → `α·hi + (1−α)·lo` (convex value blend); EvalDeriv1
→ `(hi−lo)·inv_h·one(α)` (slope along that axis); EvalDeriv2+ → carrier-aware `0`
(preserves cell-local `NaN·0 = NaN`). Mixed partials fall out by using the slope
kernel on each differentiated axis. Costs `2^N − 1` `_linear_kernel` calls vs the
old flat weight-expansion's `N·2^N` products; matches cubic/quadratic ND's collapse.

`intervals[d]::_ExplicitIndices{2}` carries `(idx_L_d, idx_R_d)` — corner `b ∈ {0,1}^N`
reads `intervals[d][b_d + 1]` (bit 0 → `idx_L_d`, bit 1 → `idx_R_d`). Non-periodic
cells have `idx_R == idx_L + 1`; periodic-exclusive seam cells have `idx_R == 1`
(wrap), so the kernel reads the wrapped neighbor without data extension.

Single interval-only kernel for all ops. Persistent callers wrap single-index
`indices` via `map(i -> _ExplicitIndices(i, i+1), indices)`; BC oneshot callers receive
seam-aware intervals from `_search_all_axis_intervals`.
"""
@generated function _multilinear_sum(
        data::AbstractArray{Tv, N},
        intervals::NTuple{N, _ExplicitIndices{2}},
        inv_hs::Tuple{Vararg{Real, N}},   # heterogeneous-tolerant (raw mixed-precision grids); each axis used independently
        αs::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        ::Val{N}
    ) where {Tv, N}
    stmts = Expr[]
    αsyms = ntuple(d -> Symbol("α_", d), N)
    ssyms = ntuple(d -> Symbol("s_", d), N)
    ihsyms = ntuple(d -> Symbol("ih_", d), N)
    opsyms = ntuple(d -> Symbol("op_", d), N)
    push!(stmts, :(($(αsyms...),) = αs))
    push!(stmts, :(($(ssyms...),) = intervals))
    push!(stmts, :(($(ihsyms...),) = inv_hs))
    push!(stmts, :(($(opsyms...),) = ops))
    # Stage 0: read the 2^N corners through the (seam-aware) intervals.
    num = 1 << N
    cur = Vector{Symbol}(undef, num)
    for c in 0:(num - 1)
        idx = [:($(ssyms[d])[$(((c >> (d - 1)) & 1) + 1)]) for d in 1:N]
        v = Symbol("g0_", c)
        push!(stmts, :($v = data[$(idx...)]))
        cur[c + 1] = v
    end
    # Stages 1..N: collapse one axis per stage via the op-specific 1D kernel
    # `_linear_kernel` (EvalValue → convex value blend; EvalDeriv1 → slope ×inv_h;
    # EvalDeriv2+ → carrier-aware 0, preserving cell-local `NaN·0 = NaN`). Axis `s`
    # is the lowest remaining corner bit, so each stage pairs adjacent positions
    # (even = bit 0 = lo, odd = bit 1 = hi).
    for s in 1:N
        half = length(cur) ÷ 2
        nxt = Vector{Symbol}(undef, half)
        for j in 0:(half - 1)
            lo = cur[2j + 1]; hi = cur[2j + 2]
            v = Symbol("g", s, "_", j)
            push!(stmts, :($v = _linear_kernel($(opsyms[s]), $lo, $hi, $(ihsyms[s]), $(αsyms[s]))))
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

# NOTE: the per-stage 1D kernel `_linear_kernel(op, yL, yR, inv_h, α)` lives in
# `linear/linear_kernels.jl` (shared with the 1D path) and dispatches the op:
# EvalValue → `α·yR + (1−α)·yL` (convex), EvalDeriv1 → `(yR−yL)·inv_h·one(α)`,
# EvalDeriv2+ → carrier-aware `0`. The old flat-form `_linear_weight` helpers are
# gone — the nested collapse calls `_linear_kernel` directly per axis.
