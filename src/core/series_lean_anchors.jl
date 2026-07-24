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
@inline _maybe_stateful_payload(::_ClampOrFill, ::Type{P}) where {P <: _AbstractAnchorPayload} = _StatefulPayload{P}
@inline _maybe_stateful_payload(::AbstractExtrap, ::Type{P}) where {P <: _AbstractAnchorPayload} = P

# ─── Resolution (method-generic; family provides `_resolve_anchor(m, …)`) ─────
# `m::M` (type parameter) forces specialization on the concrete interp singleton so
# `_resolve_anchor(m, …)` dispatches statically — mirroring the concretely-typed
# dispatch these methods had before P0 made them family-generic.
# Stateful variant needs `loc.state`, which the gridded backbone loop does not
# thread — the Series build loop below passes the whole `loc`.
@inline function _resolve_series_anchor(
        m::M,
        ::Type{_AxisAnchor{I, _StatefulPayload{P}}},
        grid::AbstractVector,
        loc,
        extrap::AbstractExtrap
    ) where {M <: AbstractInterpMethod, I <: _AbstractIndices{2}, P}
    bare = _resolve_anchor(m, _AxisAnchor{I, P}, grid, loc.idxL, loc.idxR, loc.xq, loc.xL, loc.xR, extrap)
    return _AxisAnchor{I, _StatefulPayload{P}}(
        getfield(bare, :interval), _StatefulPayload(getfield(bare, :payload), loc.state)
    )
end

@inline function _resolve_series_anchor(
        m::M,
        ::Type{A},
        grid::AbstractVector,
        loc,
        extrap::AbstractExtrap
    ) where {M <: AbstractInterpMethod, A <: _AxisAnchor}
    return _resolve_anchor(m, A, grid, loc.idxL, loc.idxR, loc.xq, loc.xL, loc.xR, extrap)
end

# ─── Build loop ──────────────────────────────────────────────────────────────
# Mirrors `_fill_anchors!` (search → optional wrap → resolve). The anchor type
# (hence op/extrap representation) comes from the caller; NoExtrap throws HERE —
# before any output is written — via the untyped `_throw_domain_error`
# (mixed-precision-safe `DomainError`, axis-agnostic `dim = 0` phrasing).

# Single lean anchor for one query — the shared build body. Scalar surfaces call
# this directly: a scalar `AutoSearch` resolves straight to a concrete `BinarySearch`
# (search.jl), so no Union forms and no barrier is needed. `searcher::SR` (type param)
# keeps the concrete searcher monomorphic; the batch loop below hands it one query at
# a time via `_fill_series_anchors_resolved!`.
@inline function _build_series_anchor(
        m::M,
        ::Type{A},
        x::AbstractVector{Tg},
        xq::Number,
        extrap::AbstractExtrap,
        wrap::Bool,
        searcher::SR
    ) where {M <: AbstractInterpMethod, A <: _AxisAnchor, Tg, SR <: Searcher}
    loc = _anchor_loc(x, _promote_coord(xq, Tg), wrap, searcher)
    if extrap isa NoExtrap && loc.state != IN_DOMAIN
        _throw_domain_error(xq, x)
    end
    return _resolve_series_anchor(m, A, x, loc, extrap)
end

# Batch fill. `searcher::SR` (type param): the caller must hand a *concrete* searcher
# — go through `_fill_series_anchors_resolved!`, which picks the policy and never lets
# the adaptive `AutoSearch` Union reach this loop (it would heap-box under coverage).
@inline function _fill_series_anchors!(
        m::M,
        buffer::AbstractVector{A},
        x::AbstractVector,
        xqs::AbstractVector{S},
        extrap::AbstractExtrap,
        wrap::Bool,
        searcher::SR
    ) where {M <: AbstractInterpMethod, A <: _AxisAnchor, S <: Number, SR <: Searcher}
    @inbounds for j in eachindex(xqs)
        buffer[j] = _build_series_anchor(m, A, x, xqs[j], extrap, wrap, searcher)
    end
    return buffer
end

# ─── Policy-consuming search barrier (THE batch entry) ────────────────────────
# Every batch Series surface (persistent + one-shot) calls this instead of resolving
# a searcher itself. It resolves the search policy and hands `_fill_series_anchors!`
# a *concrete* `Searcher` in each branch, so the adaptive `AutoSearch` choice never
# exists as a `Union` value in the caller's frame. That matters because
# `_resolve_search(grid, xqs, AutoSearch(), nothing)` on a Vector grid returns
# `Union{Searcher{BinarySearch,NoHint}, Searcher{LinearBinarySearch,RefHint}}` (the
# policy is a runtime monotonicity check); passed into the non-inlined build loop it
# heap-boxes (16 B) — normally union-split away, but `--code-coverage` disables that
# pass. Branching *before* constructing the searcher keeps every argument concrete.
# Mirrors the ND `_search_axis_adaptive` barrier (nd_utils.jl).

# Range grid: O(1) DirectSearch — no adaptive Union to begin with.
@inline function _fill_series_anchors_resolved!(
        m, buffer, grid::AbstractRange, xqs, extrap, wrap, ::AutoSearch, ::Nothing
    )
    return _fill_series_anchors!(m, buffer, grid, xqs, extrap, wrap, _to_searcher(DirectSearch()))
end

# Vector grid + AutoSearch + no hint: the one case that would form the Union. Pick the
# policy here so each arm passes a concrete `Searcher` down.
@inline function _fill_series_anchors_resolved!(
        m, buffer, grid::AbstractVector, xqs, extrap, wrap, ::AutoSearch, ::Nothing
    )
    return _is_likely_monotone(xqs) ?
        _fill_series_anchors!(m, buffer, grid, xqs, extrap, wrap, _to_searcher(LinearBinarySearch())) :
        _fill_series_anchors!(m, buffer, grid, xqs, extrap, wrap, _to_searcher(BinarySearch()))
end

# Explicit policy, or AutoSearch + hint: `_resolve_search` already yields a concrete
# `Searcher` (no Union), so a single pass-through suffices.
@inline function _fill_series_anchors_resolved!(
        m, buffer, grid, xqs, extrap, wrap, search, hint
    )
    return _fill_series_anchors!(m, buffer, grid, xqs, extrap, wrap, _resolve_search(grid, xqs, search, hint))
end

# ─── QK small-NQ variant (Q-outer × K-inner, no anchor buffer) ────────────────
# The small-NQ one-shot path builds one stack-resident anchor per query and evals all
# K series before the next query — no pool buffer (that's the whole point for tiny
# NQ). It needs the same concrete searcher as the pooled path, so it goes through the
# same policy barrier; the family supplies the point eval via `_series_point_eval`.
@inline function _series_qk_fill!(
        m, outputs, x, vecs, xqs, A, extrap, wrap, searcher::SR
    ) where {SR <: Searcher}
    K = length(vecs)
    @inbounds for j in eachindex(xqs)
        a = _build_series_anchor(m, A, x, xqs[j], extrap, wrap, searcher)
        for k in 1:K
            outputs[k][j] = _series_point_eval(m, vecs[k], a, extrap)
        end
    end
    return outputs
end

@inline function _series_qk_fill_resolved!(m, outputs, x, vecs, xqs, A, extrap, wrap, ::AutoSearch, ::Nothing)
    if x isa AbstractRange
        return _series_qk_fill!(m, outputs, x, vecs, xqs, A, extrap, wrap, _to_searcher(DirectSearch()))
    elseif _is_likely_monotone(xqs)
        return _series_qk_fill!(m, outputs, x, vecs, xqs, A, extrap, wrap, _to_searcher(LinearBinarySearch()))
    else
        return _series_qk_fill!(m, outputs, x, vecs, xqs, A, extrap, wrap, _to_searcher(BinarySearch()))
    end
end

@inline function _series_qk_fill_resolved!(m, outputs, x, vecs, xqs, A, extrap, wrap, search, hint)
    return _series_qk_fill!(m, outputs, x, vecs, xqs, A, extrap, wrap, _resolve_search(x, xqs, search, hint))
end
