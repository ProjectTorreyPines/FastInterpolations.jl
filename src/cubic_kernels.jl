# ========================================
# Cubic Spline Kernels
# ========================================
# Pure mathematical kernel functions for cubic spline evaluation.
# No dependencies - can be tested independently.
#
# Signature: _cubic_kernel(op, z0, z1, y0, y1, h, inv_h, α0, α1)
# - z0, z1: second derivative (moment) values at interval endpoints
# - y0, y1: function values at interval endpoints
# - h: interval width (x[i+1] - x[i])
# - inv_h: precomputed 1/h (eliminates fdiv in kernel)
# - α0: xi - x[i] (offset from left)
# - α1: x[i+1] - xi (offset from right)

"""
    _cubic_kernel(::EvalValue, z0, z1, y0, y1, h, inv_h, α0, α1)

Evaluate cubic spline value using moment (z) formulation.

# Formula
    S(x) = z0*(α1³)/(6h) + z1*(α0³)/(6h)
         + (y1/h - z1*h/6)*α0
         + (y0/h - z0*h/6)*α1

The computation is restructured to group common terms and leverage `muladd`
for FMA (Fused Multiply-Add) hardware instructions, reducing total FP operations.

# Operation counts (ARM64 native)
    0 fdiv + 9 fmul + 4 fmadd + 1 fmsub = 14 FP ops
"""
@inline function _cubic_kernel(
    ::EvalValue,
    z0::T, z1::T, y0::T, y1::T, h::T, inv_h::T, α0::T, α1::T
) where {T}
    # Native (ARM64) instruction breakdown:
    div6 = inv(T(6))                                    # (const-folded)
    # inv_h passed as parameter (fdiv eliminated)

    α0_cu = α0^3                                        # fmul, fmul
    α1_cu = α1^3                                        # fmul, fmul

    y_mix = muladd(y1, α0, y0 * α1)                     # fmul, fmadd
    z_mix1 = muladd(z0, α1_cu, z1 * α0_cu)              # fmul, fmadd
    z_mix2 = muladd(z1, α0, z0 * α1)                    # fmul, fmadd

    z_term = muladd(-h, z_mix2, inv_h * z_mix1) * div6  # fmul, fmsub, fmul
    return muladd(inv_h, y_mix, z_term)                 # fmadd
end
# Total: 0 fdiv + 9 fmul + 4 fmadd + 1 fmsub = 14 FP ops

"""
    _cubic_kernel(::EvalDeriv1, z0, z1, y0, y1, h, inv_h, α0, α1)

Evaluate first derivative of cubic spline.

Formula:
    S'(x) = (-z0*α1² + z1*α0²)/(2h)
          + (y1 - y0)/h
          + h*(z0 - z1)/6
"""
@inline function _cubic_kernel(
    ::EvalDeriv1,
    z0::T, z1::T, y0::T, y1::T, h::T, inv_h::T, α0::T, α1::T
) where {T}
    # inv_h passed as parameter (fdiv eliminated)

    inv_2h  = inv_h * inv(T(2))
    h_div6 = h  * inv(T(6))

    α0_sq = α0 * α0
    α1_sq = α1 * α1

    # z1*α0^2 - z0*α1^2
    z_mix   = muladd(z1, α0_sq, -z0 * α1_sq)

    # z_term = (z_mix)/(2h) + (z0 - z1)*(h/6)
    z_term  = muladd(inv_2h, z_mix, (z0 - z1) * h_div6)

    # (y1-y0)/h + z_term
    return muladd(inv_h, y1 - y0, z_term)
end

"""
    _cubic_kernel(::EvalDeriv2, z0, z1, y0, y1, h, inv_h, α0, α1)

Evaluate second derivative of cubic spline.
This is simply a linear interpolation of the z (moment) values.

Formula:
    S''(x) = (z0*α1 + z1*α0) / h
"""
@inline function _cubic_kernel(
    ::EvalDeriv2,
    z0::T, z1::T, ::T, ::T, h::T, inv_h::T, α0::T, α1::T
) where {T}
    return muladd(z0, α1, z1 * α0) * inv_h
end