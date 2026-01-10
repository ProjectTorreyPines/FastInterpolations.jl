# ========================================
# Cubic Anchored Query
# ========================================
# Precomputed geometry weights for ultra-fast cubic spline evaluation
# at a fixed query point. Enables 2-4x speedup by eliminating interval
# search and geometry setup for repeated evaluations.
#
# Include order: ops.jl → ... → cubic_types.jl → cubic_anchor.jl → cubic_interpolant.jl
#
# ========================================
# _CubicAnchoredQuery Type (Internal)
# ========================================

"""
    _CubicAnchoredQuery{T}

Precomputed weights for ultra-fast cubic spline evaluation at a fixed query point.
Internal API: no runtime grid validation; callers must ensure the anchor
matches the interpolant grid.

# Type Parameters
- `T`: Float type (Float32 or Float64)
# Fields
- `idx`: Interval index where xq falls
- `xq`: Original query point (or wrapped value for periodic)
- `side`: Domain position (0=inside, 1=left, 2=right)
- `w0`: Precomputed weights for value
- `w1`: Precomputed weights for first derivative
- `w2`: Precomputed weights for second derivative

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
aq = _anchor_query(x, 0.35)

itp1 = cubic_interp(x, sin.(2π .* x))
itp2 = cubic_interp(x, cos.(2π .* x))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=1)     # Reuses same anchor for derivative
```

# Performance
Anchored evaluation is 2-4x faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search and geometry setup.
"""
struct _CubicAnchoredQuery{T<:AbstractFloat}
    idx::Int                   # interval index
    xq::T                      # query point (possibly wrapped)
    side::UInt8                # 0=inside, 1=below_min, 2=above_max
    w0::NTuple{4,T}            # (wyL, wyR, wzL, wzR) for value
    w1::NTuple{4,T}            # (wyL, wyR, wzL, wzR) for first deriv
    w2::NTuple{4,T}            # (wyL, wyR, wzL, wzR) for second deriv
end

# ========================================
# Weight Computation
# ========================================

"""
    _compute_anchor_weights(::EvalValue, h, inv_h, dL, dR) -> NTuple{4,T}

Compute weights for cubic spline value evaluation.

Weights satisfy: S(xq) = wyL*yL + wyR*yR + wzL*zL + wzR*zR
"""
@inline function _compute_anchor_weights(::EvalValue, h::T, inv_h::T, dL::T, dR::T) where {T}
    wyL = dR * inv_h
    wyR = dL * inv_h
    # wzL = (inv_h * dR^3 - h * dR) / 6
    # wzR = (inv_h * dL^3 - h * dL) / 6
    div6 = inv(T(6))
    wzL = (inv_h * dR^3 - h * dR) * div6
    wzR = (inv_h * dL^3 - h * dL) * div6
    return (wyL, wyR, wzL, wzR)
end

"""
    _compute_anchor_weights(::EvalDeriv1, h, inv_h, dL, dR) -> NTuple{4,T}

Compute weights for cubic spline first derivative evaluation.

Weights satisfy: S'(xq) = wyL*yL + wyR*yR + wzL*zL + wzR*zR
"""
@inline function _compute_anchor_weights(::EvalDeriv1, h::T, inv_h::T, dL::T, dR::T) where {T}
    wyL = -inv_h
    wyR =  inv_h
    inv_2h = inv_h * inv(T(2))
    h_div6 = h * inv(T(6))
    wzL = -dR^2 * inv_2h + h_div6
    wzR =  dL^2 * inv_2h - h_div6
    return (wyL, wyR, wzL, wzR)
end

"""
    _compute_anchor_weights(::EvalDeriv2, h, inv_h, dL, dR) -> NTuple{4,T}

Compute weights for cubic spline second derivative evaluation.

Weights satisfy: S''(xq) = wzL*zL + wzR*zR (no y contribution)
"""
@inline function _compute_anchor_weights(::EvalDeriv2, ::T, inv_h::T, dL::T, dR::T) where {T}
    wyL = zero(T)
    wyR = zero(T)
    wzL = dR * inv_h
    wzR = dL * inv_h
    return (wyL, wyR, wzL, wzR)
end

# ========================================
# Anchor Construction
# ========================================

"""
    _anchor_query(x::AbstractVector{T}, xq::T; wrap::Bool=false) -> _CubicAnchoredQuery

Create an anchored query for ultra-fast cubic spline evaluation at a fixed point.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar)
- `wrap`: If true, wrap `xq` to domain [x[1], x[end]) before anchoring.
          Used for `extrap=:wrap` mode. Distinct from `PeriodicBC` (boundary condition).

# Returns
`_CubicAnchoredQuery{T}` with precomputed geometry weights for value and derivatives.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = cubic_interp(collect(x), sin.(2π .* x))
itp2 = cubic_interp(collect(x), cos.(2π .* x))

aq = _anchor_query(collect(x), 0.35)

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=1)     # Reuses same anchor for derivative
```

# Performance
Anchored evaluation is 2-4x faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search and geometry setup.
"""
function _anchor_query(x::AbstractVector{T}, xq::T; wrap::Bool=false) where {T<:AbstractFloat}
    return _anchor_query_impl(x, xq, wrap)
end

# Real wrapper for convenience (scalar)
function _anchor_query(x::AbstractVector{T}, xq::S; wrap::Bool=false) where {T<:AbstractFloat, S<:Real}
    _anchor_query(x, T(xq); wrap=wrap)
end

"""
    _anchor_query(x::AbstractVector{T}, xq::AbstractVector; wrap::Bool=false) -> Vector{_CubicAnchoredQuery{T}}

Create anchored queries for multiple query points.

Internal API: No runtime grid validation. Caller must ensure `x` matches
the grid used for interpolant construction.

# Arguments
- `x`: Grid points (must match interpolant's grid)
- `xq`: Query points (any Real type, auto-promoted to T)
- `wrap`: If true, wrap query points to domain [x[1], x[end]) before anchoring.
          Used for `extrap=:wrap` mode. Distinct from `PeriodicBC` (boundary condition).

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.75])

itp1 = cubic_interp(x, sin.(2π .* x))
itp2 = cubic_interp(x, cos.(2π .* x))

vals1 = itp1(aq_vec)  # Batch evaluation
vals2 = itp2(aq_vec)  # Reuse same anchors
```
"""
function _anchor_query(
    x::AbstractVector{T},
    xq::AbstractVector{S};
    wrap::Bool=false
) where {T<:AbstractFloat, S<:Real}
    output = Vector{_CubicAnchoredQuery{T}}(undef, length(xq))
    @inbounds for k in eachindex(xq)
        output[k] = _anchor_query_impl(x, T(xq[k]), wrap)
    end
    return output
end

"""
    _anchor_query_impl(x, xq, wrap) -> _CubicAnchoredQuery

Internal implementation of _anchor_query.
"""
@inline function _anchor_query_impl(
    x::AbstractVector{T},
    xq::T,
    wrap::Bool
) where {T<:AbstractFloat}
    x_min, x_max = first(x), last(x)
    # Handle wrapping (for extrap=:wrap mode)
    if wrap && (xq < x_min || xq >= x_max)
        xq = _wrap_to_domain(xq, x_min, x_max)
    end

    # Determine side (domain position)
    side = if xq < x_min
        0x01  # below min
    elseif xq > x_max
        0x02  # above max
    else
        0x00  # inside
    end

    # Find interval and compute geometry
    # For outside-domain points, use boundary intervals for weight computation
    idx, xL, xR = if xq < x_min
        # Below domain: use first interval
        @inbounds (1, x[1], x[2])
    elseif xq > x_max
        # Above domain: use last interval
        n = length(x)
        @inbounds (n - 1, x[n-1], x[n])
    else
        # Inside domain: normal interval search
        _find_interval(x, xq)
    end

    # Compute geometry
    h = xR - xL
    inv_h = one(T) / h
    dL = xq - xL  # distance from Left endpoint
    dR = xR - xq  # distance from Right endpoint

    # Compute weights for value and derivatives
    w0 = _compute_anchor_weights(EvalValue(), h, inv_h, dL, dR)
    w1 = _compute_anchor_weights(EvalDeriv1(), h, inv_h, dL, dR)
    w2 = _compute_anchor_weights(EvalDeriv2(), h, inv_h, dL, dR)

    return _CubicAnchoredQuery{T}(idx, xq, side, w0, w1, w2)
end
