# ========================================
# Quadratic Spline Kernels
# ========================================
# Pure mathematical kernel functions for quadratic spline interpolation.
# No dependencies - can be tested independently.
#
# Quadratic spline model for interval [x_i, x_{i+1}]:
#   S_i(xq) = a_i*(xq - x_i)² + d_i*(xq - x_i) + y_i
#
# Where:
#   - a_i: quadratic coefficient (controls curvature)
#   - d_i: slope at x_i (first derivative at interval start)
#   - y_i: value at x_i
#   - dL = xq - x_i (offset from interval start)

"""
    _quadratic_kernel(::EvalValue, a, d, y, dL) -> value

Evaluate quadratic polynomial at offset dL from interval start.

Formula: S(x) = a*dL² + d*dL + y = muladd(muladd(a, dL, d), dL, y)
# Arguments
- `a::T`: Quadratic coefficient
- `d::T`: Slope at interval start
- `y::T`: Value at interval start
- `dL::T`: Offset from interval start (x - x_i)
"""
@inline function _quadratic_kernel(::EvalValue, a::T, d::T, y::T, dL::T) where {T<:AbstractFloat}
    return muladd(muladd(a, dL, d), dL, y)  # a*dL² + d*dL + y
end

"""
    _quadratic_kernel(::EvalDeriv1, a, d, y, dL) -> first derivative

Evaluate first derivative of quadratic polynomial.

Formula: S'(x) = 2*a*dL + d = muladd(2*a, dL, d)
"""
@inline function _quadratic_kernel(::EvalDeriv1, a::T, d::T, ::T, dL::T) where {T<:AbstractFloat}
    return muladd(T(2)*a, dL, d)  # 2*a*dL + d
end

"""
    _quadratic_kernel(::EvalDeriv2, a, d, y, dL) -> second derivative

Evaluate second derivative of quadratic polynomial.

Formula: S''(x) = 2*a (constant within interval)
"""
@inline function _quadratic_kernel(::EvalDeriv2, a::T, ::T, ::T, ::T) where {T<:AbstractFloat}
    return T(2)*a  # constant
end

"""
    _quadratic_kernel(::EvalDeriv3, a, d, y, dL) -> zero(T)

Third derivative of quadratic spline is always zero.
Quadratic polynomials have constant second derivative, zero third derivative.
"""
@inline function _quadratic_kernel(::EvalDeriv3, ::T, ::T, ::T, ::T) where {T<:AbstractFloat}
    return zero(T)
end
