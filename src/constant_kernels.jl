# ========================================
# Constant Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for constant (step) interpolation.
# No dependencies - can be tested independently.
#
# Unified signature: _constant_kernel(op, y_left, y_right, h, dt1, side)
# - h = x1 - x0 (interval width)
# - dt1 = xi - x0 (offset from left boundary)
# - side = Val(:nearest) | Val(:left) | Val(:right)
#
# Grid point behavior: When dt1 == 0 (exactly at grid point),
# all side modes return y_left (the value at that grid point).

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dt1, ::Val{:left})

Constant interpolation with left-continuous (floor) convention.
Always returns the left boundary value `y_left`.
"""
@inline function _constant_kernel(::EvalValue, y_left::T, ::T, ::T, ::T, ::Val{:left}) where {T}
    return y_left
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dt1, ::Val{:right})

Constant interpolation with right-continuous (ceiling) convention.
Returns `y_left` at grid point (dt1 == 0), `y_right` otherwise.
"""
@inline function _constant_kernel(::EvalValue, y_left::T, y_right::T, ::T, dt1::T, ::Val{:right}) where {T}
    return iszero(dt1) ? y_left : y_right
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dt1, ::Val{:nearest})

Constant interpolation with nearest-neighbor convention and left tie-breaking.
Returns `y_left` if dt1 <= h/2 (including midpoint), `y_right` otherwise.
"""
@inline function _constant_kernel(::EvalValue, y_left::T, y_right::T, h::T, dt1::T, ::Val{:nearest}) where {T}
    return dt1 <= h / 2 ? y_left : y_right
end

"""
    _constant_kernel(::EvalDeriv1, y_left, y_right, h, dt1, side)

First derivative of constant interpolation.
Always returns zero (constant function has no slope).
"""
@inline function _constant_kernel(::EvalDeriv1, y_left::T, ::T, ::T, ::T, ::SideVal) where {T}
    return zero(T)
end

"""
    _constant_kernel(::EvalDeriv2, y_left, y_right, h, dt1, side)

Second derivative of constant interpolation.
Always returns zero (constant function has no curvature).
"""
@inline function _constant_kernel(::EvalDeriv2, y_left::T, ::T, ::T, ::T, ::SideVal) where {T}
    return zero(T)
end
