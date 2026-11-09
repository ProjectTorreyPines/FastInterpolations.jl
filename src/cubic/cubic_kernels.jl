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

# Type Parameters
- `Tg`: Grid type for h, inv_h (AbstractFloat or duck-typed, e.g. ForwardDiff.Dual)
- `Tz`: Coefficient type for zL, zR (= `_promote_eltype(_coeff_op2, Tg, Tv)` — Dual when grid is Dual)
- `Tv`: Value type for yL, yR (unconstrained, typically Float)
- `Td`: Offset type for dL, dR (Tg, ForwardDiff.Dual for AD, or a unit-carrying grid type)

# Formula
With `t = dL/h` and `u = 1 - t`:

    S(x) = u*yL + t*yR - (h²/6) * t*u * ((2 - t)*zL + (1 + t)*zR)

i.e. the linear blend plus a moment correction that vanishes at both cell ends.
Node-exact: at `dL == 0`, `t` and `t*u` are exact zeros, so `S == yL` exactly (and
`S == yR` whenever `t` rounds to 1) under any FMA contraction. Value equality, not
bits: adding the zero terms turns a `-0.0` node value into `+0.0`, as in the linear and
Hermite kernels. `dR` is unused.

# Operation counts (ARM64 native)
    0 fdiv + 4 fmul + 6 fmadd/fmsub + 1 fsub = 11 FP ops
"""
@inline function _cubic_kernel(
        ::EvalValue,
        zL::Tz, zR::Tz, yL::Tv, yR::Tv,
        h::Tg, inv_h::Ti, dL::Td, ::Td
    ) where {Tg, Ti, Tz, Tv, Td}
    t = dL * inv_h                                  # fmul
    lin = _linear_value_blend(t, yL, yR)            # fmsub, fmadd (exact at t ∈ {0, 1})
    tu = muladd(-t, t, t)                           # fmsub: t(1-t), exact 0 at t ∈ {0, 1}
    zs = muladd(t, zR - zL, muladd(2, zL, zR))      # fsub, fmadd, fmadd: (2-t)zL + (1+t)zR
    c = h * h * _inv_const(Tg, 6)                   # fmul, fmul
    return muladd(-(c * tu), zs, lin)               # fmul, fmsub
end

"""
    _cubic_kernel(::EvalDeriv1, zL, zR, yL, yR, h, inv_h, dL, dR)

Evaluate first derivative of cubic spline.

# Type Parameters
- `Tg`: Grid type for h, inv_h (AbstractFloat or duck-typed, e.g. ForwardDiff.Dual)
- `Tz`: Coefficient type for zL, zR (= `_promote_eltype(_coeff_op2, Tg, Tv)` — Dual when grid is Dual)
- `Tv`: Value type for yL, yR (unconstrained, typically Float)
- `Td`: Offset type for dL, dR (Tg, ForwardDiff.Dual for AD, or a unit-carrying grid type)

Formula:
    S'(x) = (-zL*dR² + zR*dL²)/(2h)
          + (yR - yL)/h
          + h*(zL - zR)/6
"""
@inline function _cubic_kernel(
        ::EvalDeriv1,
        zL::Tz, zR::Tz, yL::Tv, yR::Tv,
        h::Tg, inv_h::Ti, dL::Td, dR::Td
    ) where {Tg, Ti, Tz, Tv, Td}
    # inv_h passed as parameter (fdiv eliminated)

    inv_2h = inv_h * _inv_const(Tg, 2)
    h_div6 = h * _inv_const(Tg, 6)

    dL_sq = dL * dL
    dR_sq = dR * dR

    # zR*dL^2 - zL*dR^2
    z_mix = muladd(zR, dL_sq, (-dR_sq) * zL)

    # z_term = (z_mix)/(2h) + (zL - zR)*(h/6)
    z_term = muladd(inv_2h, z_mix, (zL - zR) * h_div6)

    # (yR-yL)/h + z_term — diff widens in VALUE space (Tz is z-space; converting
    # unit-carrying y into it would be dimensionally wrong)
    Tw = _value_space_eltype(Tg, Tv)
    return muladd(inv_h, _fielddiff(Tw, yR, yL), z_term)
end

"""
    _cubic_kernel(::EvalDeriv2, zL, zR, yL, yR, h, inv_h, dL, dR)

Evaluate second derivative of cubic spline.
This is simply a linear interpolation of the z (moment) values.

# Type Parameters
- `Tg`: Grid type for h, inv_h (AbstractFloat or duck-typed, e.g. ForwardDiff.Dual)
- `Tz`: Coefficient type for zL, zR (= `_promote_eltype(_coeff_op2, Tg, Tv)` — Dual when grid is Dual)
- `Tv`: Value type for yL, yR (unconstrained, typically Float)
- `Td`: Offset type for dL, dR (Tg, ForwardDiff.Dual for AD, or a unit-carrying grid type)

Formula:
    S''(x) = (zL*dR + zR*dL) / h
"""
@inline function _cubic_kernel(
        ::EvalDeriv2,
        zL::Tz, zR::Tz, _, _,
        ::Tg, inv_h::Ti, dL::Td, dR::Td
    ) where {Tg, Ti, Tz, Td}
    return muladd(zL, dR, zR * dL) * inv_h
end

"""
    _cubic_kernel(::EvalDeriv3, zL, zR, yL, yR, h, inv_h, dL, dR)

Third derivative of cubic spline (constant within each interval).

# Type Parameters
- `Tg`: Grid type for h, inv_h (AbstractFloat or duck-typed, e.g. ForwardDiff.Dual)
- `Tz`: Coefficient type for zL, zR (= `_promote_eltype(_coeff_op2, Tg, Tv)` — Dual when grid is Dual)
- `Tv`: Value type for yL, yR (unconstrained, typically Float)
- `Td`: Offset type for dL, dR (Tg, ForwardDiff.Dual for AD, or a unit-carrying grid type)

# Formula
    S'''(x) = (zR - zL) / h

# Mathematical Background
The cubic spline in moment form is:
    S(x) = zL*(dR³)/(6h) + zR*(dL³)/(6h) + (yR/h - zR*h/6)*dL + (yL/h - zL*h/6)*dR

Third derivative (constant, independent of x within interval):
    S'''(x) = (zR - zL) / h

# Operation Count
    0 fdiv + 1 fmul + 1 fsub = 2 FP ops
"""
@inline function _cubic_kernel(
        ::EvalDeriv3,
        zL::Tz, zR::Tz, _, _,
        ::Tg, inv_h::Ti, dL::Td, ::Td
    ) where {Tg, Ti, Tz, Td}
    return (zR - zL) * inv_h * one(dL)
end

"""
    _cubic_kernel(::DerivOp{N}, ...) where {N}

Generic fallback: N-th derivative of a degree-3 polynomial is zero for N ≥ 4.
Julia dispatch ensures `DerivOp{0..3}` methods (more specific) are selected first.
"""
@inline function _cubic_kernel(
        ::DerivOp{N},
        zL::Tz, _, _, _,
        ::Tg, ::Ti, dL::Td, ::Td
    ) where {N, Tg, Ti, Tz, Td}
    # `zL` is order-2 (value/grid²); an N-th derivative adds `inv(grid)^(N-2)`.
    return 0 * zL * Base.literal_pow(^, inv(oneunit(dL)), Val(N - 2))
end
