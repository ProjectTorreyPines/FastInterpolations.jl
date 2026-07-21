# Cubic spline integration kernels (private)
#
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Spline: S(u) = a·u³ + b·u² + c·u + d
#   a = (zR - zL) / (6h)
#   b = zL / 2
#   c = (yR - yL)/h - h·(2zL + zR)/6
#   d = yL

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
# Horner form of antiderivative: F(u) = u·@evalpoly(u, d, c2, b3, a4)
# with pre-absorbed coefficients a4=a/4, b3=b/3, c2=c/2.
@inline function _cubic_integral_kernel(
        ::_EvalIntegralPartial,
        zL::Tz, zR::Tz, yL::Ty, yR::Ty,
        h::Tg, u0::Td, u1::Td
    ) where {Tz, Ty, Tg, Td}
    inv_h = inv(h)
    a4 = (zR - zL) * (inv_h * inv(Tg(24)))   # a/4 = (zR-zL)/(24h)
    b3 = inv(Tg(6)) * zL                     # b/3 = zL/6
    c2 = _fielddiff(Tz, yR, yL) * (inv_h / 2) - (h * inv(Tg(12))) * (2zL + zR)  # c/2
    d = yL
    return u1 * @evalpoly(u1, d, c2, b3, a4) -
        u0 * @evalpoly(u0, d, c2, b3, a4)
end

# --- Full-cell integral: ∫_0^h S(u) du = h/2·(yL+yR) - h³/24·(zL+zR) ---
@inline function _cubic_integral_kernel(
        ::_EvalIntegralCell,
        zL::Tz, zR::Tz, yL::Ty, yR::Ty, h::Tg
    ) where {Tz, Ty, Tg}
    h2 = h * h
    return (h / 2) * muladd(-(h2 * inv(Tg(12))), zL + zR, _fieldsum(Tz, yL, yR))
end

# ═══════════════════════════════════════════════════════════════
# Linear integration kernels
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Piecewise linear: S(u) = yL + (yR - yL)/h · u
# ═══════════════════════════════════════════════════════════════

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
# Uses u1²-u0² = (u1-u0)(u1+u0) factorization + muladd.
@inline function _linear_integral_kernel(
        ::_EvalIntegralPartial,
        yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td
    ) where {Tv, Tg, Td}
    du = u1 - u0
    # Value-space widened field (wrap-free): the diff stays in value units; the
    # slope's 1/X dimension enters via `inv(2h)` (coeff-space Tc would convert
    # unit-carrying values into slope units — DimensionError).
    Tw = _promote_eltype(_interp_op, Tg, Tv, Tg)
    half_slope = _fielddiff(Tw, yR, yL) * inv(2h)
    return du * muladd(half_slope, u1 + u0, yL)
end

# --- Full-cell integral: ∫_0^h S(u) du = h/2·(yL + yR) ---
@inline function _linear_integral_kernel(
        ::_EvalIntegralCell,
        yL::Tv, yR::Tv, h::Tg
    ) where {Tv, Tg}
    Tw = _promote_eltype(_interp_op, Tg, Tv, Tg)   # value-space (see partial-cell note)
    return (h / 2) * _fieldsum(Tw, yL, yR)
end

# ═══════════════════════════════════════════════════════════════
# Quadratic integration kernels
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Piecewise quadratic: S(u) = a·u² + d·u + y₀
# (a, d are pre-computed coefficients stored in QuadraticInterpolant)
# ═══════════════════════════════════════════════════════════════

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
# Horner form: F(u) = u·@evalpoly(u, y0, d/2, a/3)
@inline function _quadratic_integral_kernel(
        ::_EvalIntegralPartial,
        a::Ta, d::Td2, y0::Ty, u0::Td, u1::Td
    ) where {Ta, Td2, Ty, Td}
    a_3 = inv(Td(3)) * a
    d_2 = inv(Td(2)) * d
    return u1 * @evalpoly(u1, y0, d_2, a_3) -
        u0 * @evalpoly(u0, y0, d_2, a_3)
end

# --- Full-cell integral: ∫_0^h S(u) du = a/3·h³ + d/2·h² + y₀·h ---
@inline function _quadratic_integral_kernel(
        ::_EvalIntegralCell,
        a::Ta, d::Td2, y0::Ty, h::Tg
    ) where {Ta, Td2, Ty, Tg}
    return h * @evalpoly(h, y0, inv(Tg(2)) * d, inv(Tg(3)) * a)
end

# ═══════════════════════════════════════════════════════════════
# Constant integration kernels
# Piecewise constant: value depends on side mode (LeftSide, RightSide, NearestSide)
# ═══════════════════════════════════════════════════════════════

# --- LeftSide — always use left value yL ---
@inline function _constant_integral_kernel(
        ::_EvalIntegralPartial,
        yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::LeftSide
    ) where {Tv, Tg, Td}
    return yL * (u1 - u0)
end

# --- RightSide — always use right value yR ---
@inline function _constant_integral_kernel(
        ::_EvalIntegralPartial,
        yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::RightSide
    ) where {Tv, Tg, Td}
    return yR * (u1 - u0)
end

# --- NearestSide — split at midpoint h/2 ---
@inline function _constant_integral_kernel(
        ::_EvalIntegralPartial,
        yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::NearestSide
    ) where {Tv, Tg, Td}
    mid = h / 2
    if u1 <= mid
        return yL * (u1 - u0)
    elseif u0 >= mid
        return yR * (u1 - u0)
    else
        return yL * (mid - u0) + yR * (u1 - mid)
    end
end

# ═══════════════════════════════════════════════════════════════
# ND Hermite integration kernels
#
# Antiderivatives of the four cubic Hermite basis functions on [0,1]:
#   H₀₀(t) = 2t³ - 3t² + 1     →  ∫H₀₀ dt = t⁴/2 - t³ + t
#   H₁₀(t) = t³ - 2t² + t      →  ∫H₁₀ dt = t⁴/4 - 2t³/3 + t²/2
#   H₀₁(t) = -2t³ + 3t²        →  ∫H₀₁ dt = -t⁴/2 + t³
#   H₁₁(t) = t³ - t²           →  ∫H₁₁ dt = t⁴/4 - t³/3
#
# Used by the @generated ND tensor-product cell integral kernel.
# ═══════════════════════════════════════════════════════════════

# Horner form: F(t) = t · @evalpoly(t, c₁, c₂, c₃) — eliminates explicit t^3, t^4.
@inline _IH00(t) = t * @evalpoly(t, 1, 0, -1, 1 / 2)
@inline _IH10(t) = t * @evalpoly(t, 0, 1 / 2, -2 / 3, 1 / 4)
@inline _IH01(t) = t * @evalpoly(t, 0, 0, 1, -1 / 2)
@inline _IH11(t) = t * @evalpoly(t, 0, 0, -1 / 3, 1 / 4)

# --- 1D Hermite integral over [u0, u1] in local coordinates ---
# Computes ∫_{u0}^{u1} P(x) dx where P is the cubic Hermite polynomial:
#   P(t) = fL·H₀₀(t) + fR·H₀₁(t) + h·(dfL·H₁₀(t) + dfR·H₁₁(t))
# with t = u/h, dx = h dt
@inline function _hermite_integral_kernel_1d(
        fL, fR, dfL, dfR,
        h::Tg, inv_h::Tg,
        u0::Real, u1::Real
    ) where {Tg}
    t0 = u0 * inv_h
    t1 = u1 * inv_h
    dH00 = _IH00(t1) - _IH00(t0)
    dH10 = _IH10(t1) - _IH10(t0)
    dH01 = _IH01(t1) - _IH01(t0)
    dH11 = _IH11(t1) - _IH11(t0)
    inner = muladd(dfR, dH11, dfL * dH10)      # dfL·ΔH₁₀ + dfR·ΔH₁₁
    outer = muladd(fR, dH01, fL * dH00)        # fL·ΔH₀₀ + fR·ΔH₀₁
    return h * muladd(h, inner, outer)
end

# ═══════════════════════════════════════════════════════════════
# Partial-cell 1D weight helpers (consumed by the separable ND engine's bounded
# node weights in integrate_fulldomain.jl). The 2^N per-cell ND kernels these
# once fed were retired with the generic engine.
# ═══════════════════════════════════════════════════════════════

# Multilinear basis weights integrated over [u0, u1]:
#   w₀(u0,u1,h) = ∫(1 - u/h) du = (u1-u0) - (u1²-u0²)/(2h)
#   w₁(u0,u1,h) = ∫(u/h) du     = (u1²-u0²)/(2h)
# Factored form: u1²-u0² = (u1-u0)(u1+u0), then muladd to avoid explicit squaring.
@inline function _w0_int(u0, u1, h)
    du = u1 - u0
    su = u1 + u0
    return du * muladd(-su, inv(2h), one(su))   # du·(1 - (u1+u0)/(2h))
end
@inline function _w1_int(u0, u1, h)
    du = u1 - u0
    return du * (u1 + u0) / (2h)
end

# Constant side-dependent partial-cell weights:
#   LeftSide → all weight to left corner; RightSide → right corner;
#   NearestSide → split at midpoint h/2.
@inline _cw0(u0, u1, h, ::LeftSide) = u1 - u0
@inline _cw1(u0, u1, h, ::LeftSide) = zero(u1 - u0)
@inline _cw0(u0, u1, h, ::RightSide) = zero(u1 - u0)
@inline _cw1(u0, u1, h, ::RightSide) = u1 - u0
@inline _cw0(u0, u1, h, ::NearestSide) = max(zero(u0), min(u1, h / 2) - u0)
@inline _cw1(u0, u1, h, ::NearestSide) = max(zero(u0), u1 - max(u0, h / 2))
