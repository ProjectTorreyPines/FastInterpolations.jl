# ========================================
# Shared lean `_AxisAnchor` Series build machinery (family-agnostic)
# ========================================
# The op/extrap-aware `_AxisAnchor{I, P}` Series backbone, shared by every
# interpolation family (cubic first, then linear/constant/quadratic). Each family
# owns its payloads, `_resolve_anchor(m, …)` weight formula, and eval kernels; the
# build/search/wrap/NoExtrap-throw scaffolding lives here and dispatches on the
# interp method `m` so a family only adds dedicated methods, never edits this loop.
#
# Included from core (before the family files) because cubic is included LAST, so
# a family file cannot see helpers defined in `cubic_series_payloads.jl`.
# Depends on: `_AxisAnchor`/`_StatefulPayload` (axis_anchor_types.jl), `_anchor_loc`
# (anchor_common.jl), `_throw_domain_error` (utils.jl), `_promote_coord`.

# Clamp/Fill wrap the bare payload so the OOB state branch stays eval-time; every
# other extrap uses the bare payload with a branch-free kernel. Compile-time
# (extrap is known at the Series entry), so the anchor type is fully concrete.
@inline _maybe_stateful_payload(::_ClampOrFill, ::Type{P}) where {P} = _StatefulPayload{P}
@inline _maybe_stateful_payload(::AbstractExtrap, ::Type{P}) where {P} = P

# ─── Resolution (method-generic; family provides `_resolve_anchor(m, …)`) ─────
# Stateful variant needs `loc.state`, which the gridded backbone loop does not
# thread — the Series build loop below passes the whole `loc`.
@inline function _resolve_series_anchor(
        m::AbstractInterpMethod,
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
        m::AbstractInterpMethod,
        ::Type{A},
        grid::AbstractVector,
        loc,
        extrap::AbstractExtrap
    ) where {A <: _AxisAnchor}
    return _resolve_anchor(m, A, grid, loc.idxL, loc.idxR, loc.xq, loc.xL, loc.xR, extrap)
end

# ─── Build loop ──────────────────────────────────────────────────────────────
# Mirrors `_fill_anchors!` (search → optional wrap → resolve). The anchor type
# (hence op/extrap representation) comes from the caller; NoExtrap throws HERE —
# before any output is written — via the untyped `_throw_domain_error`
# (mixed-precision-safe `DomainError`, axis-agnostic `dim = 0` phrasing).

# Single lean anchor for one query — the shared build body. Scalar surfaces call
# this directly; the batch loop below calls it per query.
@inline function _build_series_anchor(
        m::AbstractInterpMethod,
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
    return _resolve_series_anchor(m, A, x, loc, extrap)
end

@inline function _fill_series_anchors!(
        m::AbstractInterpMethod,
        buffer::AbstractVector{A},
        x::AbstractVector,
        xqs::AbstractVector{S},
        extrap::AbstractExtrap,
        wrap::Bool,
        searcher::SR
    ) where {A <: _AxisAnchor, S <: Real, SR <: Searcher}
    @inbounds for j in eachindex(xqs)
        buffer[j] = _build_series_anchor(m, A, x, xqs[j], extrap, wrap, searcher)
    end
    return buffer
end
