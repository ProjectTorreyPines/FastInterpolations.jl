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

# ─── Caller-frame searcher union-split ───────────────────────────────────────
# `_resolve_search` returns a small Union of `Searcher` types on a Vector grid +
# AutoSearch ({BinarySearch/NoHint, LinearBinarySearch/RefHint}) — the policy is
# picked by a runtime monotonicity check. Julia normally union-splits it away, but
# `--code-coverage` disables that optimization pass, and the larger cubic/quadratic
# build loops then do not inline — so the Union `searcher` heap-boxes (16 B) as it
# escapes the functor into the loop, breaking the zero-alloc contract. (Only under
# coverage-instrumented builds, where the union-split/inline optimization is off.)
#
# This macro does the split in *source* in the caller's own frame, so a concrete
# searcher reaches the loop and never crosses the boundary as a Union. Both branches
# are identical — the `isa` is only there to narrow the type. Range grids resolve to
# a concrete `DirectSearch` searcher, where the branch folds away.
macro _narrow_searcher(searcher, call)
    return quote
        if $(esc(searcher)) isa Searcher{BinarySearch, NoHint}
            $(esc(call))
        else
            $(esc(call))
        end
    end
end

# ─── Resolution (method-generic; family provides `_resolve_anchor(m, …)`) ─────
# `m::M` (type parameter) forces specialization on the concrete interp singleton so
# `_resolve_anchor(m, …)` dispatches statically — mirroring the concretely-typed
# dispatch these methods had before P0 made them family-generic. (The zero-alloc fix
# for the coverage-instrumented build is `@_narrow_searcher`, not this.)
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
# this directly (through `@_narrow_searcher`); the batch loop below calls it per query.
# `searcher::SR` (type param) so the concrete searcher stays monomorphic through the
# loop — see `@_narrow_searcher` for why the caller must hand it a concrete type.
@inline function _build_series_anchor(
        m::M,
        ::Type{A},
        x::AbstractVector{Tg},
        xq::Real,
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
# (via `@_narrow_searcher`); a Union searcher that escapes this loop heap-boxes under
# coverage-instrumented builds (see the macro's note).
@inline function _fill_series_anchors!(
        m::M,
        buffer::AbstractVector{A},
        x::AbstractVector,
        xqs::AbstractVector{S},
        extrap::AbstractExtrap,
        wrap::Bool,
        searcher::SR
    ) where {M <: AbstractInterpMethod, A <: _AxisAnchor, S <: Real, SR <: Searcher}
    @inbounds for j in eachindex(xqs)
        buffer[j] = _build_series_anchor(m, A, x, xqs[j], extrap, wrap, searcher)
    end
    return buffer
end
