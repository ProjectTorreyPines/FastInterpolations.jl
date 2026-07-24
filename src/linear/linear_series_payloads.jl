# ========================================
# Linear Series lean anchors (reuse gridded op-minimal payloads)
# ========================================
# The lean `_AxisAnchor{I, P}` Series layer for Linear. Unlike cubic (which owns
# bespoke payloads), Linear reuses the gridded op-minimal payloads verbatim —
# `_LinearValuePayload{Tα}` (α), `_LinearDeriv1Payload{Tα, Tinv}` (inv_h),
# `_LinearZeroPayload{Tα}` (empty) — and the gridded `_resolve_anchor(::LinearInterp, …)`
# weight formula and `_linear_kernel(op, yL, yR, a)` combine. Only the Series
# concerns are new: op/extrap→anchor-type selection (Clamp/Fill → `_StatefulPayload`)
# and the point-contiguous SIMD kernel + OOB adapter.
#
# Included AFTER gridded/gridded_linear.jl (payloads/resolve/kernel live there).
# Design: docs/design/series_lean_ports_plan.md

# Payload identity → op instance / carrier eltype (kernels + OOB arms stay op-free).
@inline _payload_op(::Type{<:_LinearValuePayload}) = EvalValue()
@inline _payload_op(::Type{<:_LinearDeriv1Payload}) = EvalDeriv1()
@inline _payload_op(::Type{<:_LinearZeroPayload}) = DerivOp(2)   # any N≥2 → carrier zero

@inline _payload_eltype(::Type{<:_LinearValuePayload{Tα}}) where {Tα} = Tα
@inline _payload_eltype(::Type{<:_LinearDeriv1Payload{Tα, Tinv}}) where {Tα, Tinv} = Tα
@inline _payload_eltype(::Type{<:_LinearZeroPayload{Tα}}) where {Tα} = Tα

# ─── op × extrap → anchor type (mirrors gridded `_axis_anchor_type(::LinearInterp)`
# geometry-type computation, with the Series query type in place of grid targets;
# Clamp/Fill wrap in `_StatefulPayload`). Compile-time; anchor type fully concrete.
@inline function _linear_series_anchor_type(
        op::AbstractEvalOp,
        extrap::AbstractExtrap,
        x::AbstractVector,
        ::Type{Tq}
    ) where {Tq}
    Tg = eltype(x)
    # inv_h is grid-only geometry — the query type must NOT widen it (the current
    # series stores `inv_h::Tg` and does the slope in grid precision, then widens
    # via `one(Tα)`). Only Tα (the carrier) takes the query type.
    Tinv = _promote_eltype(_inv_op, _promote_grid_float(Tg, Tg))
    # α = `(q - L)·inv_h` is dimensionless on unit grids; a raw `promote_type`
    # collapses to abstract `Quantity{Float64}`. Mirror gridded `_axis_anchor_type`:
    # the op-witness keeps α the concrete dimensionless carrier.
    Tα = _promote_eltype(_alpha_of, Tq, Tg, Tinv)
    P = _linear_payload_type(op, Tα, Tinv)
    return _AxisAnchor{_interval_type(x), _maybe_stateful_payload(extrap, P)}
end

# ─── Point-contiguous SIMD kernel (n_series × n_points): stream across the K
# series. Reuses the gridded `_linear_kernel(op, yL, yR, a)` combine (proven
# bit-identical to the pointwise `_linear_kernel(op, yL, yR, inv_h, α)`); the
# per-op field (`a.alpha` / `a.inv_h` / carrier) is loop-invariant, so LLVM
# hoists it and the k-loop vectorizes.
@inline function _linear_payload_kernel!(
        out::AbstractVector, y_point::Matrix, a::_AxisAnchor{I, P}
    ) where {I <: _AbstractIndices{2}, P}
    idxL = a.idxL
    idxR = a.idxR
    op = _payload_op(P)
    @inbounds @simd for k in axes(out, 1)
        out[k] = _linear_kernel(op, y_point[k, idxL], y_point[k, idxR], a)
    end
    return out
end

# Zero payload (DerivOp{N≥2}): single-term `0 * y[idxL]` per series — matches the
# Series batch/scalar convention (sign-preserving carrier zero, cell-local NaN),
# NOT the gridded two-term `(0*yL + 0*yR)` which loses `-0.0` at the boundary.
@inline function _linear_payload_kernel!(
        out::AbstractVector, y_point::Matrix, a::_AxisAnchor{I, <:_LinearZeroPayload}
    ) where {I <: _AbstractIndices{2}}
    idxL = a.idxL
    @inbounds @simd for k in axes(out, 1)
        out[k] = 0 * y_point[k, idxL]
    end
    return out
end

# ─── Extrap adapter (point-contiguous scalar/batch). Clamp/Fill own the OOB
# state branch (delegating to the shared `_fill_constant_extrap_simd!(…,::Type{Tq})`
# carrier form); in-domain rebuilds the bare anchor (isbits, zero-cost) and
# delegates to the bare kernel. Every other extrap uses the bare kernel directly.
@inline function _linear_series_eval!(
        out::AbstractVector, y_point::Matrix,
        a::_AxisAnchor{I, _StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        return _fill_constant_extrap_simd!(
            out, y_point, a.state, size(y_point, 2), _payload_op(P), extrap, _payload_eltype(P)
        )
    end
    return _linear_payload_kernel!(out, y_point, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _linear_series_eval!(out::AbstractVector, y_point::Matrix, a::_AxisAnchor, ::AbstractExtrap) =
    _linear_payload_kernel!(out, y_point, a)

# ─── Series-contiguous matrix kernel + adapter (batch loops: `y[idx, k]`) ─────
# Per-series scalar return (vs the point kernel's whole-`out` write) for the
# Q×K / K×Q batch loops. Mirrors the point kernel; only the load pattern
# (`y[idx, k]`) and the OOB helper (`_constant_extrap_boundary_value` scalar
# form) differ. Zero payload is single-term `0 * y[idxL, k]` (see point kernel).
@inline function _linear_payload_kernel(
        y::Matrix, k::Int, a::_AxisAnchor{I, P}
    ) where {I <: _AbstractIndices{2}, P}
    idxL = a.idxL
    idxR = a.idxR
    @inbounds return _linear_kernel(_payload_op(P), y[idxL, k], y[idxR, k], a)
end

@inline function _linear_payload_kernel(
        y::Matrix, k::Int, a::_AxisAnchor{I, <:_LinearZeroPayload}
    ) where {I <: _AbstractIndices{2}}
    @inbounds return 0 * y[a.idxL, k]
end

@inline function _linear_series_eval(
        y::Matrix, k::Int,
        a::_AxisAnchor{I, _StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        return _constant_extrap_boundary_value(
            y, a.state, size(y, 1), k, _payload_op(P), extrap, _payload_eltype(P)
        )
    end
    return _linear_payload_kernel(y, k, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _linear_series_eval(y::Matrix, k::Int, a::_AxisAnchor, ::AbstractExtrap) =
    _linear_payload_kernel(y, k, a)

# ─── Raw-vector kernel + adapter (one-shot: per series-vector `y[idx]`) ────────
# The one-shot surfaces eval each series y-vector independently. OOB formula is
# the one-shot form `_eval_extrapolation(op, y_bnd, extrap, zero(Tq))` (differs
# from the persistent `val*one(Tq)` on signed zero; `_eval_extrapolation` reads
# the carrier only via `zero`, so no `xq` is stored). Zero payload: `0 * y[idxL]`.
@inline function _linear_payload_kernel(
        y::AbstractVector, a::_AxisAnchor{I, P}
    ) where {I <: _AbstractIndices{2}, P}
    @inbounds return _linear_kernel(_payload_op(P), y[a.idxL], y[a.idxR], a)
end

@inline function _linear_payload_kernel(
        y::AbstractVector, a::_AxisAnchor{I, <:_LinearZeroPayload}
    ) where {I <: _AbstractIndices{2}}
    @inbounds return 0 * y[a.idxL]
end

@inline function _linear_series_eval(
        y::AbstractVector,
        a::_AxisAnchor{I, _StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        y_bnd = a.state == OOB_LEFT ? first(y) : last(y)
        return _eval_extrapolation(_payload_op(P), y_bnd, extrap, zero(_payload_eltype(P)))
    end
    return _linear_payload_kernel(y, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _linear_series_eval(y::AbstractVector, a::_AxisAnchor, ::AbstractExtrap) =
    _linear_payload_kernel(y, a)

# ─── Periodic (seam-aware) lean anchor ────────────────────────────────────────
# Periodic Series eval always wraps in-domain → bare payload. Mirrors the current
# periodic builder's 2-arg cached geometry (`_get_inv_h(x_eff, idxL)`) — NOT the
# gridded 4-arg `1/(xR-xL)`, which cancels on large-offset ranges (see the
# persistent path). The seam pair `(idxL, idxR)` (idxR == 1 at the seam) rides in
# the interval; kernels are oblivious.
@inline _resolve_linear_periodic(
    ::Type{_AxisAnchor{I, _LinearValuePayload{Tα}}}, interval, alpha, inv_h
) where {I, Tα} =
    _AxisAnchor{I, _LinearValuePayload{Tα}}(interval, _LinearValuePayload{Tα}(Tα(alpha)))
@inline _resolve_linear_periodic(
    ::Type{_AxisAnchor{I, _LinearDeriv1Payload{Tα, Tinv}}}, interval, alpha, inv_h
) where {I, Tα, Tinv} =
    _AxisAnchor{I, _LinearDeriv1Payload{Tα, Tinv}}(interval, _LinearDeriv1Payload{Tα, Tinv}(Tinv(inv_h)))
@inline _resolve_linear_periodic(
    ::Type{_AxisAnchor{I, _LinearZeroPayload{Tα}}}, interval, alpha, inv_h
) where {I, Tα} =
    _AxisAnchor{I, _LinearZeroPayload{Tα}}(interval, _LinearZeroPayload{Tα}())

@inline function _build_linear_periodic_series_anchor(
        ::Type{A}, x_eff, xq_wrapped, idxL::Int, idxR::Int, xL
    ) where {A <: _AxisAnchor}
    inv_h = _get_inv_h(x_eff, idxL)
    alpha = (xq_wrapped - xL) * inv_h
    return _resolve_linear_periodic(A, _interval_indices(x_eff, idxL, idxR), alpha, inv_h)
end
