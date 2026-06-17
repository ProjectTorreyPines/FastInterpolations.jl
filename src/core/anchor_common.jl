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

The single source of truth for every "is `xq` in domain, and if not which side?"
decision — `_anchor_loc` and every scalar/one-shot eval site route through here.
Bounds come from [`_domain_bounds`](@ref), so classification matches the batch
path (`_is_all_inbounds`) and the `_CachedRange` widened bracket (≈1 ULP past the
stored `first`/`last` on the x86_64 fast path) is handled in exactly one place —
a query at the *true* endpoint is `IN_DOMAIN`, not falsely OOB.

`_extract_primal` on both bounds and query so a Float query at the boundary
against a Dual grid endpoint classifies on primal value alone (no-op on plain
Float / `_CachedRange` fields).
"""
@inline function _oob_state(x::AbstractVector, xq::Real)
    lo, hi = _domain_bounds(x)
    xqp = _extract_primal(xq)
    xqp < _extract_primal(lo) && return OOB_LEFT
    xqp > _extract_primal(hi) && return OOB_RIGHT
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
    # Actual grid span (= x[1], x[n]) — used only for wrap-fold geometry, so the
    # periodic period stays exactly `last - first` (not the widened bracket).
    x_min, x_max = first(x), last(x)

    # Use primal value for comparisons (supports ForwardDiff.Dual)
    xq_primal = _extract_primal(xq)

    # Classify via the shared `_oob_state` (widened `_CachedRange` bracket → a
    # query at the true endpoint is IN_DOMAIN, consistent with the batch path).
    state = _oob_state(x, xq_primal)

    # Handle wrapping (for extrap=WrapExtrap() or periodic mode)
    # Generic _wrap_to_domain handles AD primal extraction and returns Tg.
    # Closed-domain convention: an IN_DOMAIN query never wraps. Only strictly-OOB
    # queries take the slow `mod()` path (which folds against the actual grid
    # span x_min/x_max), then re-classify (the folded query is in-domain).
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
        _i, _, _xL, _xR = search_interval(policy, x, xq_primal)
        (_i, _xL, _xR)
    end

    return _AnchorLoc{Tg, typeof(xq)}(idx, xq, state, xL, xR)
end
