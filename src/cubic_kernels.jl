# ========================================
# Cubic Spline Kernels
# ========================================
# Pure mathematical kernel functions for cubic spline evaluation.
# No dependencies - can be tested independently.
#
# Signature: _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
# - zL, zR: second derivative (moment) values at interval endpoints
# - yL, yR: function values at interval endpoints
# - h: interval width (x[i+1] - x[i])
# - inv_h: precomputed 1/h (eliminates fdiv in kernel)
# - dL: xq - x[i] (distance from Left endpoint)
# - dR: x[i+1] - xq (distance from Right endpoint)

"""
    _cubic_kernel(::EvalValue, zL, zR, yL, yR, h, inv_h, dL, dR)

Evaluate cubic spline value using moment (z) formulation.

# Formula
    S(x) = zL*(dR³)/(6h) + zR*(dL³)/(6h)
         + (yR/h - zR*h/6)*dL
         + (yL/h - zL*h/6)*dR

The computation is restructured to group common terms and leverage `muladd`
for FMA (Fused Multiply-Add) hardware instructions, reducing total FP operations.

# Operation counts (ARM64 native)
    0 fdiv + 9 fmul + 4 fmadd + 1 fmsub = 14 FP ops
"""
@inline function _cubic_kernel(
    ::EvalValue,
    zL::T, zR::T, yL::T, yR::T, h::T, inv_h::T, dL::T, dR::T
) where {T}
    # Native (ARM64) instruction breakdown:
    div6 = inv(T(6))                                    # (const-folded)
    # inv_h passed as parameter (fdiv eliminated)

    dL_cu = dL^3                                        # fmul, fmul
    dR_cu = dR^3                                        # fmul, fmul

    y_mix = muladd(yR, dL, yL * dR)                     # fmul, fmadd
    z_mix1 = muladd(zL, dR_cu, zR * dL_cu)              # fmul, fmadd
    z_mix2 = muladd(zR, dL, zL * dR)                    # fmul, fmadd

    z_term = muladd(-h, z_mix2, inv_h * z_mix1) * div6  # fmul, fmsub, fmul
    return muladd(inv_h, y_mix, z_term)                 # fmadd
end
# Total: 0 fdiv + 9 fmul + 4 fmadd + 1 fmsub = 14 FP ops

"""
    _cubic_kernel(::EvalDeriv1, zL, zR, yL, yR, h, inv_h, dL, dR)

Evaluate first derivative of cubic spline.

Formula:
    S'(x) = (-zL*dR² + zR*dL²)/(2h)
          + (yR - yL)/h
          + h*(zL - zR)/6
"""
@inline function _cubic_kernel(
    ::EvalDeriv1,
    zL::T, zR::T, yL::T, yR::T, h::T, inv_h::T, dL::T, dR::T
) where {T}
    # inv_h passed as parameter (fdiv eliminated)

    inv_2h  = inv_h * inv(T(2))
    h_div6 = h * inv(T(6))

    dL_sq = dL * dL
    dR_sq = dR * dR

    # zR*dL^2 - zL*dR^2
    z_mix   = muladd(zR, dL_sq, -zL * dR_sq)

    # z_term = (z_mix)/(2h) + (zL - zR)*(h/6)
    z_term  = muladd(inv_2h, z_mix, (zL - zR) * h_div6)

    # (yR-yL)/h + z_term
    return muladd(inv_h, yR - yL, z_term)
end

"""
    _cubic_kernel(::EvalDeriv2, zL, zR, yL, yR, h, inv_h, dL, dR)

Evaluate second derivative of cubic spline.
This is simply a linear interpolation of the z (moment) values.

Formula:
    S''(x) = (zL*dR + zR*dL) / h
"""
@inline function _cubic_kernel(
    ::EvalDeriv2,
    zL::T, zR::T, ::T, ::T, ::T, inv_h::T, dL::T, dR::T
) where {T}
    return muladd(zL, dR, zR * dL) * inv_h
end
