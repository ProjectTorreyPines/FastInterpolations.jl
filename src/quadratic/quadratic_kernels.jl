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
#
# Type parameters:
# - Tg<:AbstractFloat: Grid type (geometry) for dL
# - Tv: Value type (can be Tg, Complex{Tg}, or other Number) for a, d, y

"""
    _quadratic_kernel(::EvalValue, a, d, y, dL) -> value

Evaluate quadratic polynomial at offset dL from interval start.

Formula: S(x) = a*dL² + d*dL + y = muladd(muladd(a, dL, d), dL, y)
# Arguments
- `a::Tv`: Quadratic coefficient (value-derived)
- `d::Tv`: Slope at interval start (value-derived)
- `y::Tv`: Value at interval start
- `dL::Tg`: Offset from interval start (x - x_i, grid-derived)
"""
@inline function _quadratic_kernel(::EvalValue, a::Tv, d::Tv, y::Tv, dL::Tg) where {Tv, Tg<:AbstractFloat}
    return muladd(muladd(a, dL, d), dL, y)  # a*dL² + d*dL + y, returns Tv
end

"""
    _quadratic_kernel(::EvalDeriv1, a, d, y, dL) -> first derivative

Evaluate first derivative of quadratic polynomial.

Formula: S'(x) = 2*a*dL + d = muladd(2*a, dL, d)
Returns Tv (preserves Complex type when applicable).
"""
@inline function _quadratic_kernel(::EvalDeriv1, a::Tv, d::Tv, ::Tv, dL::Tg) where {Tv, Tg<:AbstractFloat}
    return muladd(Tg(2)*a, dL, d)  # 2*a*dL + d, returns Tv
end

"""
    _quadratic_kernel(::EvalDeriv2, a, d, y, dL) -> second derivative

Evaluate second derivative of quadratic polynomial.

Formula: S''(x) = 2*a (constant within interval)
Returns Tv (preserves Complex type when applicable).
"""
@inline function _quadratic_kernel(::EvalDeriv2, a::Tv, ::Tv, ::Tv, ::Tg) where {Tv, Tg<:AbstractFloat}
    return a + a  # 2*a, returns Tv (avoids type conversion issues)
end

"""
    _quadratic_kernel(::EvalDeriv3, a, d, y, dL) -> zero(Tv)

Third derivative of quadratic spline is always zero.
Quadratic polynomials have constant second derivative, zero third derivative.
Returns `zero(Tv)` to preserve Complex type when applicable.
"""
@inline function _quadratic_kernel(::EvalDeriv3, a::Tv, ::Tv, ::Tv, ::Tg) where {Tv, Tg<:AbstractFloat}
    return zero(Tv)
end
