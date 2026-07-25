# ========================================
# Constant Series lean anchors (reuse gridded op-minimal gather payload)
# ========================================
# The lean `_AxisAnchor{I, P}` Series layer for Constant. Like Linear (and unlike
# cubic), Constant reuses the gridded gather payload `_ConstantValuePayload{Tq}`
# (bakes `select_right` — WHICH node — at build) verbatim. The only new pieces are
# the Series concerns: op→payload selection (deriv → `_ConstantZeroPayload`, a
# gather that scales by 0), the Series domain-max convention, and the point/matrix/
# raw-vector kernels + OOB adapters.
#
# Constant differs from Linear in two ways that shape this file:
#   1. `select_right` depends on `m.side` (not just geometry) AND on the Series
#      domain-max convention — at `xq == last(x)` EVERY side collapses to `y[n]`
#      (LeftSide's floor is overridden). The gridded ND resolve has no such
#      override, so Constant owns a dedicated `_resolve_series_anchor(::ConstantInterp)`
#      instead of reusing the gridded `_resolve_anchor`.
#   2. The derivative is zero everywhere but its NaN-cell source differs from the
#      value's node: `0 * y[idxL]` normally (side-independent), `0 * y[n]` at
#      domain-max. `_ConstantZeroPayload` bakes that (`select_right` = domain-max).
#
# Included AFTER gridded/gridded_constant.jl (`_ConstantValuePayload`/`_carrier`
# live there) and core/series_lean_anchors.jl.
# Design: docs/design/series_lean_ports_plan.md

# Deriv gather payload: the node is `idxL` normally, `idxR` (== n) at domain-max
# (matches the value payload there for NaN cell-locality). The kernel scales by 0.
struct _ConstantZeroPayload{Tq, Toneunit} <: _AbstractAnchorPayload
    select_right::Bool
end

# Payload identity → op instance / carrier eltype (kernels + OOB arms stay op-free).
@inline _payload_op(::Type{<:_ConstantValuePayload}) = EvalValue()
@inline _payload_op(::Type{<:_ConstantZeroPayload}) = DerivOp(1)

@inline _payload_eltype(::Type{<:_ConstantValuePayload{Tq}}) where {Tq} = Tq
@inline _payload_eltype(::Type{<:_ConstantZeroPayload{Tq}}) where {Tq} = Tq

# op → payload identity (value gather vs zero gather). `EvalValue == DerivOp{0}` is
# strictly more specific than the `DerivOp` deriv fallback.
@inline _constant_series_payload_type(::EvalValue, ::Type{Tq}, ::Type) where {Tq} = _ConstantValuePayload{Tq}
@inline function _constant_series_payload_type(op::DerivOp, ::Type{Tq}, ::Type{Tg}) where {Tq, Tg}
    Toneunit = typeof(_deriv_oneunit(oneunit(Tg), op))
    return _ConstantZeroPayload{Tq, Toneunit}
end

# ExtendExtrap ≡ ClampExtrap for a step function (constant slope is zero, so
# "extend the boundary" == "hold the boundary value"). This is the documented
# contract (see `constant_interp` docstring) and what the 1D scalar / one-shot
# paths already do; the persistent Series path historically extended via the
# bare kernel (wrong node for RightSide-OOB-left / LeftSide-OOB-right). The lean
# layer unifies every surface onto the Clamp behavior. Normalizing HERE (not at
# the call sites) keeps every caller — including the OOB helper below — correct.
@inline _norm_constant_extrap(::ExtendExtrap) = ClampExtrap()
@inline _norm_constant_extrap(e::AbstractExtrap) = e

# ─── op × extrap → anchor type. Carrier `Tone` mirrors the current anchor's
# `one(dL)` where `dL = loc.xq - loc.xL` (== `_coord_eltype(Tq, Tg)`): coordinate
# subtraction, so an Int grid + Int query keeps an Int carrier (no over-floating).
# Clamp/Fill wrap in `_StatefulPayload`. Compile-time; anchor type fully concrete.
@inline function _constant_series_anchor_type(
        op::AbstractEvalOp,
        extrap::AbstractExtrap,
        x::AbstractVector,
        ::Type{Tq}
    ) where {Tq}
    Tone = _coord_eltype(Tq, eltype(x))
    P = _constant_series_payload_type(op, Tone, eltype(x))
    Toneunit = typeof(_deriv_oneunit(oneunit(eltype(x)), op))
    return _AxisAnchor{_interval_type(x), _maybe_stateful_payload(_norm_constant_extrap(extrap), P, Toneunit)}
end

# ─── Resolution (Constant-specific: side-selected node + Series domain-max
# override). Dispatches on `m::ConstantInterp` (strictly more specific than the
# family-generic `_resolve_series_anchor`), so Constant never touches the gridded
# `_resolve_anchor`. `m.side` rides in via `ConstantInterp(sitp.side)`.

# Value gather: node = side selection, OR `idxR` (== n) at the domain max.
@inline function _resolve_constant_series(
        m::ConstantInterp,
        ::Type{_AxisAnchor{I, _ConstantValuePayload{Tone}}},
        grid::AbstractVector,
        loc,
        ::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, Tone}
    h = _get_h(grid, loc.idxL, loc.xL, loc.xR)
    dL = loc.xq - loc.xL
    domain_max = _extract_primal(loc.xq) == _extract_primal(loc.xR)
    select_right = domain_max || (_compute_single_offset(m.side, h, dL) == 1)
    return _AxisAnchor{I, _ConstantValuePayload{Tone}}(
        _interval_indices(grid, loc.idxL, loc.idxR), _ConstantValuePayload{Tone}(select_right)
    )
end

# Zero gather (deriv): node = `idxL`, OR `idxR` (== n) at the domain max (matches
# the value node there — so a NaN closing cell propagates identically). Node is
# side-independent otherwise; no `h`/`dL`/offset needed.
@inline function _resolve_constant_series(
        ::ConstantInterp,
        ::Type{_AxisAnchor{I, _ConstantZeroPayload{Tone, Toneunit}}},
        grid::AbstractVector,
        loc,
        ::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, Tone, Toneunit}
    domain_max = _extract_primal(loc.xq) == _extract_primal(loc.xR)
    return _AxisAnchor{I, _ConstantZeroPayload{Tone, Toneunit}}(
        _interval_indices(grid, loc.idxL, loc.idxR), _ConstantZeroPayload{Tone, Toneunit}(domain_max)
    )
end

@inline _resolve_series_anchor(
    m::ConstantInterp, ::Type{A}, grid::AbstractVector, loc, extrap::AbstractExtrap
) where {A <: _AxisAnchor} = _resolve_constant_series(m, A, grid, loc, extrap)

@inline function _resolve_series_anchor(
        m::ConstantInterp,
        ::Type{_AxisAnchor{I, _StatefulPayload{P, Toneunit}}},
        grid::AbstractVector,
        loc,
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P, Toneunit}
    bare = _resolve_constant_series(m, _AxisAnchor{I, P}, grid, loc, extrap)
    return _AxisAnchor{I, _StatefulPayload{P, Toneunit}}(
        getfield(bare, :interval), _StatefulPayload{P, Toneunit}(getfield(bare, :payload), loc.state)
    )
end

# ─── Point-contiguous SIMD gather (n_series × n_points): stream across the K
# series. `sel` (WHICH node) + carrier `one(Tone)` are loop-invariant → LLVM
# hoists them and the k-loop vectorizes. Value multiplies the carrier; the zero
# payload additionally scales by 0 (constant deriv, cell-local NaN from `y[sel]`).
@inline function _constant_payload_kernel!(
        out::AbstractVector, y_point::Matrix, a::_AxisAnchor{I, _ConstantValuePayload{Tone}}
    ) where {I <: _AbstractIndices{2}, Tone}
    sel = ifelse(a.select_right, a.idxR, a.idxL)
    c = one(Tone)
    @inbounds @simd for k in axes(out, 1)
        out[k] = y_point[k, sel] * c
    end
    return out
end

@inline function _constant_payload_kernel!(
        out::AbstractVector, y_point::Matrix, a::_AxisAnchor{I, _ConstantZeroPayload{Tone, Toneunit}}
    ) where {I <: _AbstractIndices{2}, Tone, Toneunit}
    sel = ifelse(a.select_right, a.idxR, a.idxL)
    c = one(Tone)
    @inbounds @simd for k in axes(out, 1)
        out[k] = 0 * y_point[k, sel] * c * oneunit(Toneunit)
    end
    return out
end

# ─── Extrap adapter (point-contiguous scalar/batch). Clamp/Fill own the OOB state
# branch (shared `_fill_constant_extrap_simd!(…, ::Type{Tq})` carrier form); every
# other extrap uses the bare gather directly.
@inline function _constant_series_eval!(
        out::AbstractVector, y_point::Matrix,
        a::_AxisAnchor{I, <:_StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        deriv_oneunit = _payload_deriv_oneunit(typeof(a.payload))
        return _fill_constant_extrap_simd!(
            out, y_point, a.state, size(y_point, 2), _payload_op(P), _norm_constant_extrap(extrap), _payload_eltype(P), deriv_oneunit
        )
    end
    return _constant_payload_kernel!(out, y_point, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _constant_series_eval!(out::AbstractVector, y_point::Matrix, a::_AxisAnchor, ::AbstractExtrap) =
    _constant_payload_kernel!(out, y_point, a)

# ─── Series-contiguous matrix gather + adapter (batch loops: `y[idx, k]`) ──────
# Per-series scalar return for the Q×K / K×Q batch loops. Mirrors the point
# gather; only the load pattern (`y[idx, k]`) and the OOB helper
# (`_constant_extrap_boundary_value` scalar form) differ.
@inline function _constant_payload_kernel(
        y::Matrix, k::Int, a::_AxisAnchor{I, _ConstantValuePayload{Tone}}
    ) where {I <: _AbstractIndices{2}, Tone}
    @inbounds return y[ifelse(a.select_right, a.idxR, a.idxL), k] * one(Tone)
end

@inline function _constant_payload_kernel(
        y::Matrix, k::Int, a::_AxisAnchor{I, _ConstantZeroPayload{Tone, Toneunit}}
    ) where {I <: _AbstractIndices{2}, Tone, Toneunit}
    @inbounds return 0 * y[ifelse(a.select_right, a.idxR, a.idxL), k] * one(Tone) * oneunit(Toneunit)
end

@inline function _constant_series_eval(
        y::Matrix, k::Int,
        a::_AxisAnchor{I, <:_StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        deriv_oneunit = _payload_deriv_oneunit(typeof(a.payload))
        return _constant_extrap_boundary_value(
            y, a.state, size(y, 1), k, _payload_op(P), _norm_constant_extrap(extrap), _payload_eltype(P), deriv_oneunit
        )
    end
    return _constant_payload_kernel(y, k, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _constant_series_eval(y::Matrix, k::Int, a::_AxisAnchor, ::AbstractExtrap) =
    _constant_payload_kernel(y, k, a)

# ─── Raw-vector gather + adapter (one-shot: per series-vector `y[idx]`) ─────────
# The one-shot surfaces eval each series y-vector independently. OOB uses the
# one-shot form `_eval_extrapolation(op, y_bnd, extrap, zero(Tq))` (differs from
# the persistent `y_bnd * one` on signed zero: `_promote_extrap_val`'s `+ 0`
# normalizes `-0.0`; `_eval_extrapolation` reads the carrier only via `zero`, so
# no `xq` is stored).
@inline function _constant_payload_kernel(
        y::AbstractVector, a::_AxisAnchor{I, _ConstantValuePayload{Tone}}
    ) where {I <: _AbstractIndices{2}, Tone}
    @inbounds return y[ifelse(a.select_right, a.idxR, a.idxL)] * one(Tone)
end

@inline function _constant_payload_kernel(
        y::AbstractVector, a::_AxisAnchor{I, _ConstantZeroPayload{Tone, Toneunit}}
    ) where {I <: _AbstractIndices{2}, Tone, Toneunit}
    @inbounds return 0 * y[ifelse(a.select_right, a.idxR, a.idxL)] * one(Tone) * oneunit(Toneunit)
end

@inline function _constant_series_eval(
        y::AbstractVector,
        a::_AxisAnchor{I, <:_StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        y_bnd = a.state == OOB_LEFT ? first(y) : last(y)
        deriv_oneunit = _payload_deriv_oneunit(typeof(a.payload))
        return _eval_extrapolation(
            _payload_op(P), y_bnd, _norm_constant_extrap(extrap), zero(_payload_eltype(P)), deriv_oneunit
        )
    end
    return _constant_payload_kernel(y, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _constant_series_eval(y::AbstractVector, a::_AxisAnchor, ::AbstractExtrap) =
    _constant_payload_kernel(y, a)

# ─── Periodic (seam-aware) lean anchor ────────────────────────────────────────
# Periodic Series eval always wraps in-domain → bare payload. Mirrors the current
# periodic builder's 2-arg cached geometry (`_get_h(x_eff, idxL)`) and the
# domain-max convention at the ORIGINAL right endpoint (`last(x_eff)` — virtual
# endpoint for `:exclusive`, `x[n]` for `:inclusive`; matches `_constant_eval_at_anchor`'s
# `aq.xq == x_last`). The seam pair `(idxL, idxR)` (idxR == 1 at the seam) rides in
# the interval; the gather kernel is oblivious.
@inline _resolve_constant_periodic(
    ::Type{_AxisAnchor{I, _ConstantValuePayload{Tone}}}, interval, h, dL, domain_max::Bool, side
) where {I, Tone} =
    _AxisAnchor{I, _ConstantValuePayload{Tone}}(
    interval, _ConstantValuePayload{Tone}(domain_max || (_compute_single_offset(side, h, dL) == 1))
)
@inline _resolve_constant_periodic(
    ::Type{_AxisAnchor{I, _ConstantZeroPayload{Tone, Toneunit}}}, interval, ::Any, ::Any, domain_max::Bool, ::Any
) where {I, Tone, Toneunit} =
    _AxisAnchor{I, _ConstantZeroPayload{Tone, Toneunit}}(interval, _ConstantZeroPayload{Tone, Toneunit}(domain_max))

@inline function _build_constant_periodic_series_anchor(
        ::Type{A}, x_eff, xq_wrapped, idxL::Int, idxR::Int, xL, m::ConstantInterp
    ) where {A <: _AxisAnchor}
    h = _get_h(x_eff, idxL)
    dL = xq_wrapped - xL
    domain_max = _extract_primal(xq_wrapped) == _extract_primal(last(x_eff))
    return _resolve_constant_periodic(A, _interval_indices(x_eff, idxL, idxR), h, dL, domain_max, m.side)
end
