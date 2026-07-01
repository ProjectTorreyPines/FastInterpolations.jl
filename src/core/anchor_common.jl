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
    _AnchorLoc{Tg, Tq}

Result of `_anchor_loc`: interval location + domain state classification,
with NO geometry (h, inv_h, dL, dR). Geometry is each method's internal concern.

# Type Parameters
- `Tg <: AbstractFloat`: grid element type (for `xL`, `xR`)
- `Tq <: Real`: query type (preserves ForwardDiff.Dual for AD)

# Fields
- `idx::Int`: interval index ∈ 1:(n-1)
- `xq::Tq`: query point (possibly wrapped), preserves Dual for AD
- `state::UInt8`: domain state — `IN_DOMAIN`, `OOB_LEFT`, or `OOB_RIGHT`
- `xL::Tg`: left node x[idx]
- `xR::Tg`: right node x[idx+1]
"""
struct _AnchorLoc{Tg, Tq <: Real}
    idx::Int
    xq::Tq
    state::UInt8
    xL::Tg
    xR::Tg
end

# ========================================
# _anchor_loc: Shared Location Function
# ========================================

"""
    _anchor_loc(x, xq, wrap, policy) -> _AnchorLoc{Tg, Tq}

Shared interval location for all interpolation methods.
Performs: domain-state classification (`_oob_state`, widened `_CachedRange`
bracket) → wrap-fold only for OOB queries when `wrap` (then re-classify) →
interval search (boundary cell for OOB, `search_interval` otherwise).

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

    # Find interval
    # For outside-domain points, use boundary intervals for weight computation
    idx, xL, xR = if state == OOB_LEFT
        @inbounds (1, x[1], x[2])
    elseif state == OOB_RIGHT
        n = length(x)
        @inbounds (n - 1, x[n - 1], x[n])
    else
        # IN_DOMAIN: `_oob_state` already established the (possibly wrapped) query is in-domain, so
        # the interval search takes the guard-free lean path (InBounds) — bit-identical to guarded,
        # coupled to the `_oob_state` classification above. A `_ExclusivePeriodicAxis` routes to its
        # seam-aware search via the InBounds overload (periodic_axis.jl). (Perf-neutral in practice:
        # the search is amortized over K series and `_oob_state` already pays the boundary compares;
        # kept for consistency with the other in-domain search paths.)
        _i, _, _xL, _xR = search_interval(policy, x, xq_primal, InBounds())
        (_i, _xL, _xR)
    end

    return _AnchorLoc{Tg, typeof(xq)}(idx, xq, state, xL, xR)
end
