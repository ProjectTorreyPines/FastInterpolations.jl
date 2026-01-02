# ========================================
# Quadratic Spline Kernels
# ========================================
# Pure mathematical kernel functions for quadratic spline interpolation.
# No dependencies - can be tested independently.
#
# Quadratic spline model for interval [x_i, x_{i+1}]:
#   S_i(x) = a_i*(x - x_i)² + d_i*(x - x_i) + y_i
#
# Where:
#   - a_i: quadratic coefficient (controls curvature)
#   - d_i: slope at x_i (first derivative at interval start)
#   - y_i: value at x_i
#   - dt = x - x_i (offset from interval start)

"""
    _quadratic_kernel(::EvalValue, a, d, y, dt) -> value

Evaluate quadratic polynomial at offset dt from interval start.

Formula: S(x) = a*dt² + d*dt + y = muladd(muladd(a, dt, d), dt, y)

# Arguments
- `a::T`: Quadratic coefficient
- `d::T`: Slope at interval start
- `y::T`: Value at interval start
- `dt::T`: Offset from interval start (x - x_i)
"""
@inline function _quadratic_kernel(::EvalValue, a::T, d::T, y::T, dt::T) where {T<:AbstractFloat}
    return muladd(muladd(a, dt, d), dt, y)  # a*dt² + d*dt + y
end

"""
    _quadratic_kernel(::EvalDeriv1, a, d, y, dt) -> first derivative

Evaluate first derivative of quadratic polynomial.

Formula: S'(x) = 2*a*dt + d = muladd(2*a, dt, d)
"""
@inline function _quadratic_kernel(::EvalDeriv1, a::T, d::T, ::T, dt::T) where {T<:AbstractFloat}
    return muladd(2*a, dt, d)  # 2*a*dt + d
end

"""
    _quadratic_kernel(::EvalDeriv2, a, d, y, dt) -> second derivative

Evaluate second derivative of quadratic polynomial.

Formula: S''(x) = 2*a (constant within interval)
"""
@inline function _quadratic_kernel(::EvalDeriv2, a::T, ::T, ::T, ::T) where {T<:AbstractFloat}
    return 2*a  # constant
end
