# ========================================
# Quadratic Series lean anchors (bake dL, thread op through `_quadratic_kernel`)
# ========================================
# The lean `_AxisAnchor{I, P}` Series layer for Quadratic. Quadratic is
# coefficient-based (S(x) = a*dL² + d*dL + y) but the coefficients y/a/d are
# precomputed at build and stored per series — so, like Linear reading stored y,
# the anchor bakes only the OP-INDEPENDENT geometry `dL`, and the op-threaded
# `_quadratic_kernel` picks the formula. A SINGLE payload suffices (unlike
# Linear/Constant's op→payload): the op changes only the kernel formula, never
# the baked `dL`, so tagging the anchor type with the op would buy nothing.
#
# ExtendExtrap extends the boundary polynomial (bare kernel) — NO Clamp
# normalization (a quadratic has slope/curvature to extend, unlike a constant).
# Clamp/Fill OOB uses the persistent `_fill_*` / `_constant_extrap_boundary_value`
# carrier forms (shared with Constant/Linear; preserve signed zero) for the
# point/matrix surfaces, and the one-shot `_eval_extrapolation` for raw vectors.
# The three lean kernels all call `_quadratic_kernel`, so they are bit-identical
# to the current batch/one-shot path (which already routes through it); the
# scalar path's deriv2/deriv3 micro-optimizations unify to that canonical form.
#
# Included AFTER quadratic_kernels.jl + quadratic_anchor.jl and after
# core/series_lean_anchors.jl. Design: docs/design/series_lean_ports_plan.md

# Op-independent payload: bakes the offset `dL` (== `loc.xq - loc.xL`, carrier
# `_coord_eltype(Tq, Tg)`). `idx` rides in the interval.
struct _QuadraticPayload{Tdl} <: _AbstractAnchorPayload
    dL::Tdl
end

@inline _payload_eltype(::Type{<:_QuadraticPayload{Tdl}}) where {Tdl} = Tdl

# ─── extrap → anchor type (op-independent). Clamp/Fill wrap in `_StatefulPayload`.
@inline function _quadratic_series_anchor_type(
        op::DerivOp, extrap::AbstractExtrap, x::AbstractVector, ::Type{Tq}
    ) where {Tq}
    Tdl = _coord_eltype(Tq, eltype(x))
    Ts = typeof(_constant_axis_deriv_scale(oneunit(eltype(x)), op))
    return _AxisAnchor{
        _interval_type(x), _maybe_stateful_payload(extrap, _QuadraticPayload{Tdl}, Ts),
    }
end

@inline _quadratic_series_anchor_type(
    extrap::AbstractExtrap, x::AbstractVector, ::Type{Tq}
) where {Tq} = _quadratic_series_anchor_type(EvalValue(), extrap, x, Tq)

# ─── Resolution: bake `dL = xq - xL` (mirrors `_quadratic_anchor_query_impl`).
# Reuses the shared `_resolve_series_anchor` (bare + stateful) machinery.
@inline function _resolve_anchor(
        ::QuadraticInterp,
        ::Type{_AxisAnchor{I, _QuadraticPayload{Tdl}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        ::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, Tdl}
    return _AxisAnchor{I, _QuadraticPayload{Tdl}}(_interval_indices(grid, idxL, idxR), _QuadraticPayload{Tdl}(xq - xL))
end

# ─── Point-contiguous SIMD kernel (n_series × n_points). `_quadratic_kernel`
# dispatches on the compile-time `op`, so the k-loop specializes + vectorizes.
@inline function _quadratic_payload_kernel!(
        out::AbstractVector, y_point::Matrix, a_point::Matrix, d_point::Matrix,
        anchor::_AxisAnchor{I, _QuadraticPayload{Tdl}}, op::AbstractEvalOp
    ) where {I <: _AbstractIndices{2}, Tdl}
    idx = anchor.idxL
    dL = anchor.dL
    @inbounds @simd for k in eachindex(out)
        out[k] = _quadratic_kernel(op, a_point[k, idx], d_point[k, idx], y_point[k, idx], dL)
    end
    return out
end

# ─── Point adapter. Clamp/Fill own the OOB state branch (shared carrier-form
# `_fill_constant_extrap_simd!`, preserving signed zero — matches Constant/Linear);
# every other extrap (incl. Extend, which extends the polynomial) uses the bare
# kernel. NoExtrap already threw at build.
@inline function _quadratic_series_eval!(
        out::AbstractVector, y_point::Matrix, a_point::Matrix, d_point::Matrix,
        anchor::_AxisAnchor{I, _StatefulPayload{P, S}}, op::AbstractEvalOp, extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P, S}
    if anchor.state != IN_DOMAIN
        scale = _stateful_deriv_scale(typeof(anchor.payload))
        return _fill_constant_extrap_simd!(
            out, y_point, anchor.state, size(y_point, 2), op, extrap, _payload_eltype(P), scale
        )
    end
    return _quadratic_payload_kernel!(out, y_point, a_point, d_point, _AxisAnchor(getfield(anchor, :interval), anchor.inner), op)
end

@inline _quadratic_series_eval!(
    out::AbstractVector, y_point::Matrix, a_point::Matrix, d_point::Matrix,
    anchor::_AxisAnchor, op::AbstractEvalOp, ::AbstractExtrap
) = _quadratic_payload_kernel!(out, y_point, a_point, d_point, anchor, op)

# ─── Series-contiguous matrix kernel + adapter (batch loops: `y[idx, k]`).
@inline function _quadratic_payload_kernel(
        y::AbstractMatrix, a::AbstractMatrix, d::AbstractMatrix, k::Int,
        anchor::_AxisAnchor{I, _QuadraticPayload{Tdl}}, op::AbstractEvalOp
    ) where {I <: _AbstractIndices{2}, Tdl}
    idx = anchor.idxL
    @inbounds return _quadratic_kernel(op, a[idx, k], d[idx, k], y[idx, k], anchor.dL)
end

@inline function _quadratic_series_eval(
        y::AbstractMatrix, a::AbstractMatrix, d::AbstractMatrix, k::Int,
        anchor::_AxisAnchor{I, _StatefulPayload{P, S}}, op::AbstractEvalOp, extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P, S}
    if anchor.state != IN_DOMAIN
        scale = _stateful_deriv_scale(typeof(anchor.payload))
        return _constant_extrap_boundary_value(
            y, anchor.state, size(y, 1), k, op, extrap, _payload_eltype(P), scale
        )
    end
    return _quadratic_payload_kernel(y, a, d, k, _AxisAnchor(getfield(anchor, :interval), anchor.inner), op)
end

@inline _quadratic_series_eval(
    y::AbstractMatrix, a::AbstractMatrix, d::AbstractMatrix, k::Int,
    anchor::_AxisAnchor, op::AbstractEvalOp, ::AbstractExtrap
) = _quadratic_payload_kernel(y, a, d, k, anchor, op)

# ─── Raw-vector kernel + adapter (one-shot: per series-vector `y[idx]`). OOB uses
# the one-shot `_eval_extrapolation` form (normalizes signed zero; no `xq` stored).
@inline function _quadratic_payload_kernel(
        y::AbstractVector, a::AbstractVector, d::AbstractVector,
        anchor::_AxisAnchor{I, _QuadraticPayload{Tdl}}, op::AbstractEvalOp
    ) where {I <: _AbstractIndices{2}, Tdl}
    idx = anchor.idxL
    @inbounds return _quadratic_kernel(op, a[idx], d[idx], y[idx], anchor.dL)
end

@inline function _quadratic_series_eval(
        y::AbstractVector, a::AbstractVector, d::AbstractVector,
        anchor::_AxisAnchor{I, _StatefulPayload{P, S}}, op::AbstractEvalOp, extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P, S}
    if anchor.state != IN_DOMAIN
        y_bnd = anchor.state == OOB_LEFT ? first(y) : last(y)
        scale = _stateful_deriv_scale(typeof(anchor.payload))
        return _eval_extrapolation(op, y_bnd, extrap, zero(_payload_eltype(P)), scale)
    end
    return _quadratic_payload_kernel(y, a, d, _AxisAnchor(getfield(anchor, :interval), anchor.inner), op)
end

@inline _quadratic_series_eval(
    y::AbstractVector, a::AbstractVector, d::AbstractVector,
    anchor::_AxisAnchor, op::AbstractEvalOp, ::AbstractExtrap
) = _quadratic_payload_kernel(y, a, d, anchor, op)
