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
Performs: wrap → domain state classification → interval search.

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
    x_min, x_max = first(x), last(x)

    # Use primal value for comparisons (supports ForwardDiff.Dual)
    xq_primal = _extract_primal(xq)

    # Handle wrapping (for extrap=WrapExtrap() or periodic mode)
    # Generic _wrap_to_domain handles AD primal extraction and returns Tg.
    # Closed-domain convention: `xq == x_max` is in-domain — no wrap needed.
    # Only strictly-OOB queries (`xq < x_min` or `xq > x_max`) take the slow
    # `mod()` path inside `_wrap_to_domain` (which itself uses `xi <= x_max`).
    if wrap && (xq_primal < x_min || xq_primal > x_max)
        xq = _wrap_to_domain(xq, x_min, x_max)
        xq_primal = xq  # xq is now Tg, no need for _extract_primal
    end

    # Classify domain state
    state = if xq_primal < x_min
        OOB_LEFT
    elseif xq_primal > x_max
        OOB_RIGHT
    else
        IN_DOMAIN
    end

    # Find interval
    # For outside-domain points, use boundary intervals for weight computation
    idx, xL, xR = if xq_primal < x_min
        @inbounds (1, x[1], x[2])
    elseif xq_primal > x_max
        n = length(x)
        @inbounds (n - 1, x[n - 1], x[n])
    else
        _i, _, _xL, _xR = search_interval(policy, x, xq_primal)
        (_i, _xL, _xR)
    end

    return _AnchorLoc{Tg, typeof(xq)}(idx, xq, state, xL, xR)
end
