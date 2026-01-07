# ========================================
# Cubic Spline Kernels
# ========================================
# Pure mathematical kernel functions for cubic spline evaluation.
# No dependencies - can be tested independently.
#
# Signature: _cubic_kernel(op, z0, z1, y0, y1, h, inv_h, dL, dR)
# - z0, z1: second derivative (moment) values at interval endpoints
# - y0, y1: function values at interval endpoints
# - h: interval width (x[i+1] - x[i])
# - inv_h: precomputed 1/h (eliminates fdiv in kernel)
# - dL: xq - x[i] (distance from Left endpoint)
# - dR: x[i+1] - xq (distance from Right endpoint)

"""
    _cubic_kernel(::EvalValue, z0, z1, y0, y1, h, inv_h, dL, dR)

Evaluate cubic spline value using moment (z) formulation.

# Formula
    S(x) = z0*(dR³)/(6h) + z1*(dL³)/(6h)
         + (y1/h - z1*h/6)*dL
         + (y0/h - z0*h/6)*dR

The computation is restructured to group common terms and leverage `muladd`
for FMA (Fused Multiply-Add) hardware instructions, reducing total FP operations.

# Operation counts (ARM64 native)
    0 fdiv + 9 fmul + 4 fmadd + 1 fmsub = 14 FP ops
"""
@inline function _cubic_kernel(
    ::EvalValue,
    z0::T, z1::T, y0::T, y1::T, h::T, inv_h::T, dL::T, dR::T
) where {T}
    # Native (ARM64) instruction breakdown:
    div6 = inv(T(6))                                    # (const-folded)
    # inv_h passed as parameter (fdiv eliminated)

    dL_cu = dL^3                                        # fmul, fmul
    dR_cu = dR^3                                        # fmul, fmul

    y_mix = muladd(y1, dL, y0 * dR)                     # fmul, fmadd
    z_mix1 = muladd(z0, dR_cu, z1 * dL_cu)              # fmul, fmadd
    z_mix2 = muladd(z1, dL, z0 * dR)                    # fmul, fmadd

    z_term = muladd(-h, z_mix2, inv_h * z_mix1) * div6  # fmul, fmsub, fmul
    return muladd(inv_h, y_mix, z_term)                 # fmadd
end
# Total: 0 fdiv + 9 fmul + 4 fmadd + 1 fmsub = 14 FP ops

"""
    _cubic_kernel(::EvalDeriv1, z0, z1, y0, y1, h, inv_h, dL, dR)

Evaluate first derivative of cubic spline.

Formula:
    S'(x) = (-z0*dR² + z1*dL²)/(2h)
          + (y1 - y0)/h
          + h*(z0 - z1)/6
"""
@inline function _cubic_kernel(
    ::EvalDeriv1,
    z0::T, z1::T, y0::T, y1::T, h::T, inv_h::T, dL::T, dR::T
) where {T}
    # inv_h passed as parameter (fdiv eliminated)

    inv_2h  = inv_h * inv(T(2))
    h_div6 = h  * inv(T(6))

    dL_sq = dL * dL
    dR_sq = dR * dR

    # z1*dL^2 - z0*dR^2
    z_mix   = muladd(z1, dL_sq, -z0 * dR_sq)

    # z_term = (z_mix)/(2h) + (z0 - z1)*(h/6)
    z_term  = muladd(inv_2h, z_mix, (z0 - z1) * h_div6)

    # (y1-y0)/h + z_term
    return muladd(inv_h, y1 - y0, z_term)
end

"""
    _cubic_kernel(::EvalDeriv2, z0, z1, y0, y1, h, inv_h, dL, dR)

Evaluate second derivative of cubic spline.
This is simply a linear interpolation of the z (moment) values.

Formula:
    S''(x) = (z0*dR + z1*dL) / h
"""
@inline function _cubic_kernel(
    ::EvalDeriv2,
    z0::T, z1::T, ::T, ::T, h::T, inv_h::T, dL::T, dR::T
) where {T}
    return muladd(z0, dR, z1 * dL) * inv_h
end