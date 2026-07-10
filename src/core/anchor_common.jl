# ========================================
# Shared Per-Axis Anchor Location
# ========================================
# Single shared building block for all anchor construction (cubic, linear,
# quadratic, constant). Extracts the common wrap → state → search logic.
# Method-specific geometry/weight computation remains in each method's anchor file.
#
# Include order: utils.jl → periodic.jl → anchor_common.jl
# Dependencies: _extract_primal (utils.jl), _wrap_to_domain (periodic.jl),
#               search_interval + Searcher (search.jl)

# ========================================
# Domain state constants
# ========================================

const IN_DOMAIN = 0x00
const OOB_LEFT = 0x01
const OOB_RIGHT = 0x02

# ========================================
# _oob_state: shared OOB classifier (single source of truth)
# ========================================

"""
    _oob_state(x, xq) -> UInt8

Classify query `xq` against grid `x` as `IN_DOMAIN`, `OOB_LEFT`, or `OOB_RIGHT`.

Single source of truth for the in-domain / which-side decision: bounds come from
[`_domain_bounds`](@ref), so it matches the batch path and the `_CachedRange`
widened bracket is handled in one place (a query at the true endpoint is
`IN_DOMAIN`). `_extract_primal` on bounds and query keeps it partial-sign
independent for Dual grids.
"""
@inline function _oob_state(x::AbstractVector, xq::Real)
    lo, hi = _domain_bounds(x)
    xqp = _extract_primal(xq)
    # `_lt`/`_gt` promote-compare: dodge Base's exact mixed `<(Int, Float)` on an
    # Int/Rational grid (no-op on a Float grid). See ordering helpers in search.jl.
    _lt(xqp, _extract_primal(lo)) && return OOB_LEFT
    _gt(xqp, _extract_primal(hi)) && return OOB_RIGHT
    return IN_DOMAIN
end

# ========================================
# _AnchorLoc: Location-Only Result
# ========================================

"""
    _AnchorLoc{I, Tg, Tq}

Result of `_anchor_loc`: physical search interval + domain state classification,
with NO geometry (h, inv_h, dL, dR). Geometry is each method's internal concern.

# Type Parameters
- `I <: _AbstractIndices{2}`: concrete interval representation — `_ContiguousIndices{2}`
  for ordinary axes, `_ExplicitIndices{2}` for exclusive-periodic axes. Selected
  per axis type (see [`_interval_indices`](@ref)), so the parameter is concrete.
- `Tg <: AbstractFloat`: grid element type (for `xL`, `xR`)
- `Tq <: Real`: query type (preserves ForwardDiff.Dual for AD)

# Fields
- `interval::I`: the physical `(idxL, idxR)` cell returned by `search_interval`.
  Ordinary axes store only the left index (`idxR == idxL + 1`); exclusive-periodic
  axes store both, so the seam cell `(n, 1)` survives.
- `xq::Tq`: query point (possibly wrapped), preserves Dual for AD
- `state::UInt8`: domain state — `IN_DOMAIN`, `OOB_LEFT`, or `OOB_RIGHT`
- `xL::Tg`: left node coordinate paired with `idxL`
- `xR::Tg`: right node coordinate paired with `idxR`

`idxL`/`idxR` are Val-dispatched virtual properties reading through `interval`.
"""
struct _AnchorLoc{
        I <: _AbstractIndices{2},
        Tg,
        Tq <: Real,
    }
    interval::I
    xq::Tq
    state::UInt8
    xL::Tg
    xR::Tg
end

# ──────────────────────────────────────────────────────────────
# Virtual property accessors — `loc.idxL` / `loc.idxR` read through `interval`
# ──────────────────────────────────────────────────────────────
# `idxL`/`idxR` always mean the physical search-interval endpoints. Val-dispatch
# forces compile-time specialization so each access inlines to `getfield` + a
# folded index op (a single-method Symbol branch would union the return types
# — Int / Tq / UInt8 / Tg / I — and defeat inference in hot loops).
@inline Base.getproperty(loc::_AnchorLoc, s::Symbol) = _get_anchor_loc_property(loc, Val(s))
@inline _get_anchor_loc_property(loc::_AnchorLoc, ::Val{:idxL}) = getfield(loc, :interval)[Val(1)]
@inline _get_anchor_loc_property(loc::_AnchorLoc, ::Val{:idxR}) = getfield(loc, :interval)[Val(2)]
@inline _get_anchor_loc_property(loc::_AnchorLoc, ::Val{s}) where {s} = getfield(loc, s)
@inline Base.propertynames(::_AnchorLoc) = (:interval, :idxL, :idxR, :xq, :state, :xL, :xR)

# ──────────────────────────────────────────────────────────────
# Interval representation selector — chosen per axis type, not per query
# ──────────────────────────────────────────────────────────────
# Ordinary axes never store a redundant right index; exclusive-periodic axes
# always store the explicit pair (including interior queries) so one anchor
# vector has one concrete element type.
@inline _interval_indices(::AbstractVector, idxL::Int, idxR::Int) = _ContiguousIndices{2}(idxL)
@inline _interval_indices(::_ExclusivePeriodicAxis, idxL::Int, idxR::Int) = _ExplicitIndices(idxL, idxR)

# ========================================
# _anchor_loc: Shared Location Function
# ========================================

"""
    _anchor_loc(x, xq, wrap, policy) -> _AnchorLoc{I, Tg, Tq}

Shared interval location for all interpolation methods.
Performs: domain-state classification (`_oob_state`, widened `_CachedRange`
bracket) → wrap-fold only for OOB queries when `wrap` (then re-classify) →
interval search (lean `InBounds` for in-domain, guarded `search_interval` for
OOB).

Returns `_AnchorLoc` with NO geometry — each method computes its own
h/inv_h/dL/dR from `xL`, `xR`, `xq` as needed.

# Arguments
- `x::AbstractVector{Tg}`: sorted grid points
- `xq::Tq`: query point (Real — can be Float or ForwardDiff.Dual)
- `wrap::Bool`: whether to wrap query to domain (for periodic/WrapExtrap)
- `policy::Searcher`: search policy (default: `DEFAULT_SEARCHER`)

# AD Support
When `xq` is a ForwardDiff.Dual, the returned `_AnchorLoc.xq` preserves the
Dual type. The interval search uses `_extract_primal(xq)` for comparisons.
"""
@inline function _anchor_loc(
        x::AbstractVector{Tg},
        xq::Tq,
        wrap::Bool,
        policy::P = DEFAULT_SEARCHER
    ) where {Tg, Tq <: Real, P <: Searcher}
    # Actual grid span — used only for wrap-fold geometry (period stays exactly
    # `last - first`, not the widened bracket).
    x_min, x_max = first(x), last(x)

    # Use primal value for comparisons (supports ForwardDiff.Dual)
    xq_primal = _extract_primal(xq)

    # Classify via the shared `_oob_state` (widened `_CachedRange` bracket → a
    # query at the true endpoint is IN_DOMAIN, consistent with the batch path).
    state = _oob_state(x, xq_primal)

    # WrapExtrap/periodic: an IN_DOMAIN query never wraps. Strictly-OOB queries
    # take the `mod()` path (folds against the actual span x_min/x_max, returns
    # Tg with AD primal handled), then re-classify.
    if wrap && state != IN_DOMAIN
        xq = _wrap_to_domain(xq, x_min, x_max)
        xq_primal = xq  # xq is now Tg, no need for _extract_primal
        state = _oob_state(x, xq_primal)
    end

    # Find interval. For ordinary axes, the right index is always `idx + 1`, so
    # `_interval_indices` stores only the left index (contiguous).
    idx, xL, xR = if state == OOB_LEFT
        @inbounds (1, x[1], x[2])
    elseif state == OOB_RIGHT
        n = length(x)
        @inbounds (n - 1, x[n - 1], x[n])
    else
        # IN_DOMAIN: `_oob_state` already established the (possibly wrapped) query is in-domain, so
        # the interval search takes the lean path (InBounds) — bit-identical to guarded,
        # coupled to the `_oob_state` classification above.
        _i, _, _xL, _xR = search_interval(policy, x, xq_primal, InBounds())
        (_i, _xL, _xR)
    end

    interval = _interval_indices(x, idx, idx + 1)
    return _AnchorLoc{typeof(interval), Tg, typeof(xq)}(interval, xq, state, xL, xR)
end

@inline function _anchor_loc(
        x::_ExclusivePeriodicAxis{Tg},
        xq::Tq,
        wrap::Bool,
        policy::P = DEFAULT_SEARCHER
    ) where {Tg, Tq <: Real, P <: Searcher}
    # Actual grid span — used only for wrap-fold geometry (period stays exactly
    # `last - first`, not the widened bracket).
    x_min, x_max = first(x), last(x)

    # Use primal value for comparisons (supports ForwardDiff.Dual)
    xq_primal = _extract_primal(xq)

    # Classify via the shared `_oob_state` (widened `_CachedRange` bracket → a
    # query at the true endpoint is IN_DOMAIN, consistent with the batch path).
    state = _oob_state(x, xq_primal)

    # WrapExtrap/periodic: an IN_DOMAIN query never wraps. Strictly-OOB queries
    # take the `mod()` path (folds against the actual span x_min/x_max, returns
    # Tg with AD primal handled), then re-classify.
    if wrap && state != IN_DOMAIN
        xq = _wrap_to_domain(xq, x_min, x_max)
        xq_primal = xq  # xq is now Tg, no need for _extract_primal
        state = _oob_state(x, xq_primal)
    end

    # Seam-aware axes must preserve the right index returned by their search
    # overload: the seam cell is `(idx, idxR) == (n, 1)`. `_interval_indices`
    # stores both indices explicitly for every query on this axis.
    idx, idxR, xL, xR = if state == IN_DOMAIN
        search_interval(policy, x, xq_primal, InBounds())
    else
        search_interval(policy, x, xq_primal)
    end

    interval = _interval_indices(x, idx, idxR)
    return _AnchorLoc{typeof(interval), Tg, typeof(xq)}(interval, xq, state, xL, xR)
end
