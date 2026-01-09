# ========================================
# Cubic Anchored Query
# ========================================
# Precomputed geometry weights for ultra-fast cubic spline evaluation
# at a fixed query point. Enables 2-4x speedup by eliminating interval
# search and geometry setup for repeated evaluations.
#
# Include order: ops.jl → ... → cubic_types.jl → cubic_anchor.jl → cubic_interpolant.jl
#
# Note: _grid_id(x) is defined in utils.jl (shared with CubicInterpolant)

# ========================================
# CubicAnchoredQuery Type
# ========================================

"""
    CubicAnchoredQuery{T,Op}

Precomputed weights for ultra-fast cubic spline evaluation at a fixed query point.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `Op`: Operation type (EvalValue, EvalDeriv1, EvalDeriv2) for compile-time dispatch

# Fields
- `grid_id`: Grid identity token (length, hash) for validation
- `idx`: Interval index where xq falls
- `xq`: Original query point (or wrapped value for periodic)
- `side`: Domain position (0=inside, 1=left, 2=right)
- `w`: Precomputed weights (wyL, wyR, wzL, wzR)

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
aq = anchor_query(x, 0.35)

itp1 = cubic_interp(x, sin.(2π .* x))
itp2 = cubic_interp(x, cos.(2π .* x))

itp1(aq)  # Ultra-fast: skips interval search
itp2(aq)  # Reuses same anchor
```

# Performance
Anchored evaluation is 2-4x faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search and geometry setup.
"""
struct CubicAnchoredQuery{T<:AbstractFloat, Op<:AbstractEvalOp}
    grid_id::Tuple{Int,UInt}   # (length, hash) for validation
    idx::Int                   # interval index
    xq::T                      # query point (possibly wrapped)
    side::UInt8                # 0=inside, 1=below_min, 2=above_max
    w::NTuple{4,T}             # (wyL, wyR, wzL, wzR)
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
    anchor_query(x::AbstractVector{T}, xq::T; deriv::Int=0, periodic::Bool=false) -> CubicAnchoredQuery

Create an anchored query for ultra-fast cubic spline evaluation at a fixed point.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar)
- `deriv`: Derivative order (0=value, 1=first, 2=second)
- `periodic`: If true, wrap `xq` to domain before anchoring

# Returns
`CubicAnchoredQuery{T,Op}` with precomputed geometry weights.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = cubic_interp(collect(x), sin.(2π .* x))
itp2 = cubic_interp(collect(x), cos.(2π .* x))

aq = anchor_query(collect(x), 0.35)

itp1(aq)  # Ultra-fast: skips interval search
itp2(aq)  # Reuses same anchor
```

# Performance
Anchored evaluation is 2-4x faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search and geometry setup.
"""
function anchor_query(x::AbstractVector{T}, xq::T; deriv::Int=0, periodic::Bool=false) where {T<:AbstractFloat}
    # Validate deriv argument
    (deriv < 0 || deriv > 2) && throw(ArgumentError("deriv must be 0, 1, or 2; got $deriv"))

    # Dispatch to typed implementation
    if deriv == 0
        return _anchor_query_impl(x, xq, EvalValue(), periodic)
    elseif deriv == 1
        return _anchor_query_impl(x, xq, EvalDeriv1(), periodic)
    else  # deriv == 2
        return _anchor_query_impl(x, xq, EvalDeriv2(), periodic)
    end
end

# Real wrapper for convenience
function anchor_query(x::AbstractVector{T}, xq::S; deriv::Int=0, periodic::Bool=false) where {T<:AbstractFloat, S<:Real}
    anchor_query(x, T(xq); deriv=deriv, periodic=periodic)
end

"""
    _anchor_query_impl(x, xq, op, periodic) -> CubicAnchoredQuery

Internal implementation of anchor_query with concrete Op type.
"""
@inline function _anchor_query_impl(
    x::AbstractVector{T},
    xq::T,
    op::Op,
    periodic::Bool
) where {T<:AbstractFloat, Op<:AbstractEvalOp}
    x_min, x_max = first(x), last(x)
    grid_id = _grid_id(x)

    # Handle periodic wrapping
    if periodic && (xq < x_min || xq >= x_max)
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

    # Compute weights
    w = _compute_anchor_weights(op, h, inv_h, dL, dR)

    return CubicAnchoredQuery{T,Op}(grid_id, idx, xq, side, w)
end
