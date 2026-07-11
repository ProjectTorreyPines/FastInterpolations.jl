# ========================================
# Cubic Series lean anchors (op/extrap-aware payloads)
# ========================================
# Lean `_AxisAnchor{I, P}` payloads for the Series batch surfaces, which know
# the eval op AND the extrap at anchor-build time. Each payload bakes only its
# op's weights (vs the 96 B all-ops `_CubicAnchoredQuery`); flat extraps
# (Clamp/Fill) wrap the payload in the generic `_StatefulPayload` so the OOB
# state branch stays eval-time, exactly like the full-anchor path.
# Design: docs/design/cubic_series_payload_anchor.md
#
# Weight formulas reuse `_compute_anchor_weights` verbatim → bit-identical to
# the corresponding `_CubicAnchoredQuery` field (w0/w1/w2/w3) by construction.

struct _CubicValuePayload1D{Tq}
    w::NTuple{4, Tq}
end
struct _CubicDeriv1Payload1D{Tq}
    w::NTuple{4, Tq}
end
struct _CubicDeriv2Payload1D{Tq}
    w::NTuple{2, Tq}
end
struct _CubicDeriv3Payload1D{Tq}
    w::NTuple{2, Tq}
end
struct _CubicZeroPayload1D{Tq} end   # DerivOp{N≥4}: result is a carrier zero, no weights

# Weighted payloads share one resolve arm; the zero payload has its own.
const _CubicWeightedPayload1D = Union{
    _CubicValuePayload1D, _CubicDeriv1Payload1D,
    _CubicDeriv2Payload1D, _CubicDeriv3Payload1D,
}

# Payload identity → op instance (kernels and OOB arms stay op-argument-free).
@inline _payload_op(::Type{<:_CubicValuePayload1D}) = EvalValue()
@inline _payload_op(::Type{<:_CubicDeriv1Payload1D}) = EvalDeriv1()
@inline _payload_op(::Type{<:_CubicDeriv2Payload1D}) = EvalDeriv2()
@inline _payload_op(::Type{<:_CubicDeriv3Payload1D}) = EvalDeriv3()
@inline _payload_op(::Type{<:_CubicZeroPayload1D}) = DerivOp(4)   # any N≥4 is equivalent downstream
@inline _payload_op(::Type{_StatefulPayload{P}}) where {P} = _payload_op(P)

# ─── Payload/anchor type selection ───────────────────────────────────────────
# op → payload; Clamp/Fill → stateful wrap. Both compile-time (op and extrap
# are known at the batch entry), so the anchor type is fully concrete.

@inline _cubic_series_payload_type(::EvalValue, ::Type{Tq}) where {Tq} = _CubicValuePayload1D{Tq}
@inline _cubic_series_payload_type(::EvalDeriv1, ::Type{Tq}) where {Tq} = _CubicDeriv1Payload1D{Tq}
@inline _cubic_series_payload_type(::EvalDeriv2, ::Type{Tq}) where {Tq} = _CubicDeriv2Payload1D{Tq}
@inline _cubic_series_payload_type(::EvalDeriv3, ::Type{Tq}) where {Tq} = _CubicDeriv3Payload1D{Tq}
@inline _cubic_series_payload_type(::DerivOp, ::Type{Tq}) where {Tq} = _CubicZeroPayload1D{Tq}

@inline _maybe_stateful_payload(::_ClampOrFill, ::Type{P}) where {P} = _StatefulPayload{P}
@inline _maybe_stateful_payload(::AbstractExtrap, ::Type{P}) where {P} = P

@inline function _cubic_series_anchor_type(
        op::AbstractEvalOp,
        extrap::AbstractExtrap,
        x::AbstractVector,
        ::Type{Tq}
    ) where {Tq}
    return _AxisAnchor{_interval_type(x), _maybe_stateful_payload(extrap, _cubic_series_payload_type(op, Tq))}
end

# ─── Resolution ──────────────────────────────────────────────────────────────
# Mirrors the ND partials overloads' shape (`gridded_partials.jl`); only the
# payload type inside `A` differs, so dispatch stays collision-free.

@inline function _resolve_anchor(
        ::CubicInterp,
        ::Type{_AxisAnchor{I, P}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        ::AbstractExtrap
    ) where {I, P <: _CubicWeightedPayload1D}
    h = _get_h(grid, idxL, xL, xR)
    inv_h = _get_inv_h(grid, idxL, xL, xR)
    dL = xq - xL
    dR = xR - xq
    w = _compute_anchor_weights(_payload_op(P), h, inv_h, dL, dR)
    return _AxisAnchor{I, P}(_interval_indices(grid, idxL, idxR), P(w))
end

@inline function _resolve_anchor(
        ::CubicInterp,
        ::Type{_AxisAnchor{I, _CubicZeroPayload1D{Tq}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        ::AbstractExtrap
    ) where {I, Tq}
    return _AxisAnchor{I, _CubicZeroPayload1D{Tq}}(
        _interval_indices(grid, idxL, idxR), _CubicZeroPayload1D{Tq}()
    )
end

# Stateful variant needs `loc.state`, which the gridded backbone loop does not
# thread — the Series-owned build loop below passes the whole `loc`.
@inline function _resolve_series_anchor(
        m::CubicInterp,
        ::Type{_AxisAnchor{I, _StatefulPayload{P}}},
        grid::AbstractVector,
        loc,
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    bare = _resolve_anchor(m, _AxisAnchor{I, P}, grid, loc.idxL, loc.idxR, loc.xq, loc.xL, loc.xR, extrap)
    return _AxisAnchor{I, _StatefulPayload{P}}(
        getfield(bare, :interval), _StatefulPayload(getfield(bare, :payload), loc.state)
    )
end

@inline function _resolve_series_anchor(
        m::CubicInterp,
        ::Type{A},
        grid::AbstractVector,
        loc,
        extrap::AbstractExtrap
    ) where {A <: _AxisAnchor}
    return _resolve_anchor(m, A, grid, loc.idxL, loc.idxR, loc.xq, loc.xL, loc.xR, extrap)
end

# ─── Series-owned build loop ─────────────────────────────────────────────────
# Mirrors `_fill_anchors!` (search → optional wrap → resolve), with two
# differences: the anchor type (hence op/extrap representation) comes from the
# buffer eltype, and NoExtrap throws HERE — before any output is written — via
# the untyped `_throw_domain_error` (mixed-precision-safe `DomainError`,
# axis-agnostic `dim = 0` phrasing).

# Single lean anchor for one query — the shared build body. Scalar surfaces call
# this directly; the batch loop below calls it per query. NoExtrap throws HERE,
# before any output is written, via the untyped `_throw_domain_error`.
@inline function _build_series_anchor(
        ::Type{A},
        x::AbstractVector{Tg},
        xq::Real,
        extrap::AbstractExtrap,
        wrap::Bool,
        searcher::Searcher
    ) where {A <: _AxisAnchor, Tg}
    loc = _anchor_loc(x, _promote_coord(xq, Tg), wrap, searcher)
    if extrap isa NoExtrap && loc.state != IN_DOMAIN
        _throw_domain_error(xq, x)
    end
    return _resolve_series_anchor(CubicInterp(), A, x, loc, extrap)
end

@inline function _fill_series_anchors!(
        buffer::AbstractVector{A},
        x::AbstractVector,
        xqs::AbstractVector{S},
        extrap::AbstractExtrap,
        wrap::Bool,
        searcher::SR
    ) where {A <: _AxisAnchor, S <: Real, SR <: Searcher}
    @inbounds for j in eachindex(xqs)
        buffer[j] = _build_series_anchor(A, x, xqs[j], extrap, wrap, searcher)
    end
    return buffer
end

# Periodic (seam-aware) lean anchor: mirrors `_build_periodic_cubic_anchor`
# exactly (wrap → 4-tuple search → 2-arg cached geometry) while baking only the
# requested op's weights. Periodic Series eval has no extrap dispatch (queries
# always wrap in-domain), so the payload is always bare.
@inline function _build_periodic_series_anchor(
        ::Type{_AxisAnchor{I, P}},
        cache::CubicSplineCache,
        xq,
        searcher::Searcher
    ) where {I <: _AbstractIndices{2}, P}
    xq_wrapped = _wrap_to_domain(xq, cache.x)
    idxL, idxR, xL, xR = search_interval(searcher, cache.x, xq_wrapped)
    h = _get_h(cache.x, idxL)
    inv_h = _get_inv_h(cache.x, idxL)
    dL = xq_wrapped - xL
    dR = xR - xq_wrapped
    return _AxisAnchor{I, P}(
        _interval_indices(cache.x, idxL, idxR), _make_series_payload(P, h, inv_h, dL, dR)
    )
end

@inline _make_series_payload(::Type{P}, h, inv_h, dL, dR) where {P <: _CubicWeightedPayload1D} =
    P(_compute_anchor_weights(_payload_op(P), h, inv_h, dL, dR))
@inline _make_series_payload(::Type{_CubicZeroPayload1D{Tq}}, h, inv_h, dL, dR) where {Tq} =
    _CubicZeroPayload1D{Tq}()

# ─── Bare payload kernels ────────────────────────────────────────────────────
# Bodies mirror the full-anchor kernels exactly — matrix layout mirrors
# `_eval_series_anchored`, raw-vector layout mirrors `_cubic_eval_kernel` —
# with weights read from the payload (`a.w`) instead of the op-selected field.
# Named for the payload family, not "series": any future lean-anchor consumer
# (e.g. unified `interp` batch routing) reuses them as-is.

# matrix layout (series-contiguous column k)
@inline function _cubic_payload_kernel(
        y::Matrix, z::Matrix, k::Int,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:Union{_CubicValuePayload1D, _CubicDeriv1Payload1D}}
    )
    wyL, wyR, wzL, wzR = a.w
    idxL = a.idxL
    idxR = a.idxR
    @inbounds return muladd(
        wyR, y[idxR, k], muladd(
            wyL, y[idxL, k],
            muladd(wzR, z[idxR, k], wzL * z[idxL, k])
        )
    )
end

@inline function _cubic_payload_kernel(
        y::Matrix, z::Matrix, k::Int,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:Union{_CubicDeriv2Payload1D, _CubicDeriv3Payload1D}}
    )
    wzL, wzR = a.w
    @inbounds return muladd(wzR, z[a.idxR, k], wzL * z[a.idxL, k])
end

@inline function _cubic_payload_kernel(
        y::Matrix, ::Matrix, k::Int,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:_CubicZeroPayload1D}
    )
    @inbounds return 0 * y[a.idxL, k]
end

# raw-vector layout (one-shot Series)
@inline function _cubic_payload_kernel(
        y::AbstractVector, z::AbstractVector,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:Union{_CubicValuePayload1D, _CubicDeriv1Payload1D}}
    )
    wyL, wyR, wzL, wzR = a.w
    idxL = a.idxL
    idxR = a.idxR
    @inbounds return muladd(
        wyR, y[idxR], muladd(
            wyL, y[idxL],
            muladd(wzR, z[idxR], wzL * z[idxL])
        )
    )
end

@inline function _cubic_payload_kernel(
        ::AbstractVector, z::AbstractVector,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:Union{_CubicDeriv2Payload1D, _CubicDeriv3Payload1D}}
    )
    wzL, wzR = a.w
    @inbounds return muladd(wzR, z[a.idxR], wzL * z[a.idxL])
end

@inline function _cubic_payload_kernel(
        y::AbstractVector, ::AbstractVector,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:_CubicZeroPayload1D}
    )
    return 0 * (@inbounds y[a.idxL])
end

# ─── Surface extrap adapters ─────────────────────────────────────────────────
# These ARE surface-specific: each preserves its surface's CURRENT OOB formula
# bit-exactly (they differ on signed zero — see the OOB pins):
#   * persistent matrix:  `val * one(Tq)`         (series_utils helpers)
#   * one-shot raw-vector: `_eval_extrapolation(op, val, extrap, zero(Tq))`
# `one(Tq)`/`zero(Tq)` are value-independent, so no `xq` is stored. In-domain
# delegates to the bare kernel through a rebuilt bare anchor (isbits, zero-cost).

@inline _payload_eltype(::Type{_CubicValuePayload1D{Tq}}) where {Tq} = Tq
@inline _payload_eltype(::Type{_CubicDeriv1Payload1D{Tq}}) where {Tq} = Tq
@inline _payload_eltype(::Type{_CubicDeriv2Payload1D{Tq}}) where {Tq} = Tq
@inline _payload_eltype(::Type{_CubicDeriv3Payload1D{Tq}}) where {Tq} = Tq
@inline _payload_eltype(::Type{_CubicZeroPayload1D{Tq}}) where {Tq} = Tq

@inline function _cubic_series_eval(
        y::Matrix, z::Matrix, k::Int,
        a::_AxisAnchor{I, _StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        return _constant_extrap_boundary_value(
            y, a.state, size(y, 1), k, _payload_op(P), extrap, _payload_eltype(P)
        )
    end
    return _cubic_payload_kernel(y, z, k, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _cubic_series_eval(y::Matrix, z::Matrix, k::Int, a::_AxisAnchor, ::AbstractExtrap) =
    _cubic_payload_kernel(y, z, k, a)

@inline function _cubic_series_eval(
        y::AbstractVector, z::AbstractVector,
        a::_AxisAnchor{I, _StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        y_bnd = a.state == OOB_LEFT ? first(y) : last(y)
        return _eval_extrapolation(_payload_op(P), y_bnd, extrap, zero(_payload_eltype(P)))
    end
    return _cubic_payload_kernel(y, z, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _cubic_series_eval(y::AbstractVector, z::AbstractVector, a::_AxisAnchor, ::AbstractExtrap) =
    _cubic_payload_kernel(y, z, a)

# ─── Point-contiguous kernels + adapter (persistent SCALAR) ──────────────────
# Bodies mirror the matrix kernels/adapter above, transposed to point-contiguous
# `y_point[k, idx]` (n_series × n_points) so the k-loop SIMD-streams across the
# series dimension — the layout the scalar surface needs. The `!` bang marks the
# whole-`out` write (vs the matrix kernel's per-k scalar return). Weights/indices
# still come from the shared payload/anchor; only the load pattern differs. The
# zero-payload arm is per-k (`0 * y_point[k, idxL]`) so scalar matches the matrix
# kernel bit-for-bit, including the derivative-≥4 signed zero.
@inline function _cubic_payload_kernel!(
        out::AbstractVector, y_point::Matrix, z_point::Matrix,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:Union{_CubicValuePayload1D, _CubicDeriv1Payload1D}}
    )
    wyL, wyR, wzL, wzR = a.w
    idxL = a.idxL
    idxR = a.idxR
    @inbounds @simd for k in axes(out, 1)
        out[k] = muladd(
            wyR, y_point[k, idxR], muladd(
                wyL, y_point[k, idxL],
                muladd(wzR, z_point[k, idxR], wzL * z_point[k, idxL])
            )
        )
    end
    return out
end

@inline function _cubic_payload_kernel!(
        out::AbstractVector, ::Matrix, z_point::Matrix,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:Union{_CubicDeriv2Payload1D, _CubicDeriv3Payload1D}}
    )
    wzL, wzR = a.w
    idxL = a.idxL
    idxR = a.idxR
    @inbounds @simd for k in axes(out, 1)
        out[k] = muladd(wzR, z_point[k, idxR], wzL * z_point[k, idxL])
    end
    return out
end

@inline function _cubic_payload_kernel!(
        out::AbstractVector, y_point::Matrix, ::Matrix,
        a::_AxisAnchor{<:_AbstractIndices{2}, <:_CubicZeroPayload1D}
    )
    idxL = a.idxL
    @inbounds @simd for k in axes(out, 1)
        out[k] = 0 * y_point[k, idxL]
    end
    return out
end

@inline function _cubic_series_eval!(
        out::AbstractVector, y_point::Matrix, z_point::Matrix,
        a::_AxisAnchor{I, _StatefulPayload{P}},
        extrap::AbstractExtrap
    ) where {I <: _AbstractIndices{2}, P}
    if a.state != IN_DOMAIN
        return _fill_constant_extrap_simd!(
            out, y_point, a.state, size(y_point, 2), _payload_op(P), extrap, _payload_eltype(P)
        )
    end
    return _cubic_payload_kernel!(out, y_point, z_point, _AxisAnchor(getfield(a, :interval), a.inner))
end

@inline _cubic_series_eval!(out::AbstractVector, y_point::Matrix, z_point::Matrix, a::_AxisAnchor, ::AbstractExtrap) =
    _cubic_payload_kernel!(out, y_point, z_point, a)
