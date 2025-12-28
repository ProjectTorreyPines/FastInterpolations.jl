# ========================================
# Linear Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for linear interpolation.
# No dependencies - can be tested independently.
#
# Unified signature: _linear_kernel(op, y0, y1, h, dt1)
# - h = x1 - x0 (interval width)
# - dt1 = xi - x0 (offset from left boundary)
# - α = dt1 / h (computed internally for Value)

"""
    _linear_kernel(::EvalValue, y0, y1, h, dt1)

Evaluate linear interpolation value at offset dt1 from left boundary.
Returns: y0 * (1 - α) + y1 * α where α = dt1 / h
"""
@inline function _linear_kernel(::EvalValue, y0::T, y1::T, h::T, dt1::T) where {T}
    α = dt1 / h
    return y0 * (one(T) - α) + y1 * α
end

"""
    _linear_kernel(::EvalDeriv1, y0, y1, h, dt1)

Evaluate first derivative (slope) of linear interpolation.
Returns constant slope: (y1 - y0) / h
"""
@inline function _linear_kernel(::EvalDeriv1, y0::T, y1::T, h::T, ::T) where {T}
    return (y1 - y0) / h
end

"""
    _linear_kernel(::EvalDeriv2, y0, y1, h, dt1)

Evaluate second derivative of linear interpolation.
Always returns zero (linear function has no curvature).

Note: Mathematically, the second derivative is a Dirac delta at knots,
but we return zero everywhere as a practical approximation.
"""
@inline function _linear_kernel(::EvalDeriv2, ::T, ::T, ::T, ::T) where {T}
    return zero(T)
end
