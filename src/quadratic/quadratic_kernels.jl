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
# - Td: Offset type for dL (can be Tg, or promoted Tg×Tq for AD)
# - Value args (a, d, y) are untyped for duck-type support (Dual coefficients
#   on duck grids). The typed `dL::Td` provides the LLVM specialization anchor
#   (same pattern as _hermite_kernel_1d's h::Tg, inv_h::Tinv, dL::Tq).

"""
    _quadratic_kernel(::EvalValue, a, d, y, dL) -> value

Evaluate quadratic polynomial at offset dL from interval start.

Formula: S(x) = a*dL² + d*dL + y = muladd(muladd(a, dL, d), dL, y)
# Arguments
- `a`: Quadratic coefficient (value-derived, may be Dual for duck grids)
- `d`: Slope at interval start (value-derived)
- `y`: Value at interval start
- `dL::Td`: Offset from interval start (x - x_i, typed for specialization)
"""
@inline function _quadratic_kernel(::EvalValue, a, d, y, dL::Td) where {Td}
    return muladd(muladd(a, dL, d), dL, y)  # a*dL² + d*dL + y
end

"""
    _quadratic_kernel(::EvalDeriv1, a, d, y, dL) -> first derivative

Evaluate first derivative of quadratic polynomial.

Formula: S'(x) = 2*a*dL + d = muladd(2*a, dL, d)
"""
@inline function _quadratic_kernel(::EvalDeriv1, a, d, _, dL::Td) where {Td}
    return muladd(2 * a, dL, d)  # 2*a*dL + d
end

"""
    _quadratic_kernel(::EvalDeriv2, a, d, y, dL) -> second derivative

Evaluate second derivative of quadratic polynomial.

Formula: S''(x) = 2*a (constant within interval)
"""
@inline function _quadratic_kernel(::EvalDeriv2, a, _, _, dL::Td) where {Td}
    return (a + a) * one(dL)  # 2*a, carrier-aware via `* one(dL)`
end

"""
    _quadratic_kernel(::EvalDeriv3, a, d, y, dL)

Third derivative of quadratic spline is always zero.
Uses `0 * a` for duck-typing support and NaN propagation.
"""
@inline function _quadratic_kernel(::EvalDeriv3, a, _, _, dL::Td) where {Td}
    return 0 * a * one(dL)
end

"""Generic fallback: N-th derivative of degree-2 polynomial is zero for N ≥ 3."""
@inline function _quadratic_kernel(::DerivOp{N}, a, _, _, dL::Td) where {N, Td}
    return 0 * a * one(dL)
end
