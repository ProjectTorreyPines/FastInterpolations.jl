# ========================================
# Linear Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for linear interpolation.
# No dependencies - can be tested independently.
#
# Unified signature: _linear_kernel(op, yL, yR, h, dL)
# - h = x1 - x0 (interval width)
# - dL = xq - x0 (offset from left boundary)
# - α = dL / h (computed internally for Value)

"""
    _linear_kernel(::EvalValue, yL, yR, h, dL)

Evaluate linear interpolation value at offset dL from left boundary.
Returns: yL * (1 - α) + yR * α where α = dL / h
"""
@inline function _linear_kernel(::EvalValue, yL::T, yR::T, h::T, dL::T) where {T}
    α = dL / h
    return muladd(α, yR - yL, yL)  # yL + α*(yR-yL)
end

"""
    _linear_kernel(::EvalDeriv1, yL, yR, h, dL)

Evaluate first derivative (slope) of linear interpolation.
Returns constant slope: (yR - yL) / h
"""
@inline function _linear_kernel(::EvalDeriv1, yL::T, yR::T, h::T, ::T) where {T}
    return (yR - yL) / h
end

"""
    _linear_kernel(::EvalDeriv2, yL, yR, h, dL)

Evaluate second derivative of linear interpolation.
Always returns zero (linear function has no curvature).

Note: Mathematically, the second derivative is a Dirac delta at knots,
but we return zero everywhere as a practical approximation.
"""
@inline function _linear_kernel(::EvalDeriv2, ::T, ::T, ::T, ::T) where {T}
    return zero(T)
end
