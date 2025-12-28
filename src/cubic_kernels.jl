# ========================================
# Cubic Spline Kernels
# ========================================
# Pure mathematical kernel functions for cubic spline evaluation.
# No dependencies - can be tested independently.
#
# Signature: _cubic_kernel(op, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)
# - z_i, z_ip1: second derivative (moment) values at interval endpoints
# - y_i, y_ip1: function values at interval endpoints
# - h_i: interval width (x[i+1] - x[i])
# - dt1: xi - x[i] (offset from left)
# - dt2: x[i+1] - xi (offset from right)

"""
    _cubic_kernel(::EvalValue, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)

Evaluate cubic spline value using moment (z) formulation.

Formula:
    S(x) = z_i*(dt2³)/(6h) + z_{i+1}*(dt1³)/(6h)
         + (y_{i+1}/h - z_{i+1}*h/6)*dt1
         + (y_i/h - z_i*h/6)*dt2
"""
@inline function _cubic_kernel(
    ::EvalValue,
    z_i::T, z_ip1::T, y_i::T, y_ip1::T, h_i::T, dt1::T, dt2::T
) where {T}
    I = (z_i * dt2^3 + z_ip1 * dt1^3) / (6 * h_i)
    C = (y_ip1 / h_i - z_ip1 * h_i / 6) * dt1
    D = (y_i / h_i - z_i * h_i / 6) * dt2
    return I + C + D
end

"""
    _cubic_kernel(::EvalDeriv1, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)

Evaluate first derivative of cubic spline.

Formula:
    S'(x) = (-z_i*dt2² + z_{i+1}*dt1²)/(2h)
          + (y_{i+1} - y_i)/h
          + h*(z_i - z_{i+1})/6
"""
@inline function _cubic_kernel(
    ::EvalDeriv1,
    z_i::T, z_ip1::T, y_i::T, y_ip1::T, h_i::T, dt1::T, dt2::T
) where {T}
    term1 = (-z_i * dt2^2 + z_ip1 * dt1^2) / (2 * h_i)
    term2 = (y_ip1 - y_i) / h_i
    term3 = (z_i - z_ip1) * h_i / 6
    return term1 + term2 + term3
end

"""
    _cubic_kernel(::EvalDeriv2, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)

Evaluate second derivative of cubic spline.
This is simply a linear interpolation of the z (moment) values.

Formula:
    S''(x) = (z_i*dt2 + z_{i+1}*dt1) / h
"""
@inline function _cubic_kernel(
    ::EvalDeriv2,
    z_i::T, z_ip1::T, ::T, ::T, h_i::T, dt1::T, dt2::T
) where {T}
    return (z_i * dt2 + z_ip1 * dt1) / h_i
end
