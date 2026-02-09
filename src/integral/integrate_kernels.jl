# Cubic spline integration kernels (private)
#
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Spline: S(u) = a·u³ + b·u² + c·u + d
#   a = (zR - zL) / (6h)
#   b = zL / 2
#   c = (yR - yL)/h - h·(2zL + zR)/6
#   d = yL

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
@inline function _cubic_integral_kernel(
    ::_EvalIntegralPartial,
    zL::Tv, zR::Tv, yL::Tv, yR::Tv,
    h::Tg, u0::Td, u1::Td
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    a = (zR - zL) / (6h)
    b = zL / 2
    c = (yR - yL) / h - h * (2zL + zR) / 6
    d = yL
    return (a / 4) * (u1^4 - u0^4) +
           (b / 3) * (u1^3 - u0^3) +
           (c / 2) * (u1^2 - u0^2) +
            d      * (u1 - u0)
end

# --- Full-cell integral: ∫_0^h S(u) du = h/2·(yL+yR) - h³/24·(zL+zR) ---
@inline function _cubic_integral_kernel(
    ::_EvalIntegralCell,
    zL::Tv, zR::Tv, yL::Tv, yR::Tv, h::Tg
) where {Tv, Tg<:AbstractFloat}
    return h / 2 * (yL + yR) - h^3 / 24 * (zL + zR)
end

# ═══════════════════════════════════════════════════════════════
# Linear integration kernels
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Piecewise linear: S(u) = yL + (yR - yL)/h · u
# ═══════════════════════════════════════════════════════════════

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
@inline function _linear_integral_kernel(
    ::_EvalIntegralPartial,
    yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return yL * (u1 - u0) + (yR - yL) / (2h) * (u1^2 - u0^2)
end

# --- Full-cell integral: ∫_0^h S(u) du = h/2·(yL + yR) ---
@inline function _linear_integral_kernel(
    ::_EvalIntegralCell,
    yL::Tv, yR::Tv, h::Tg
) where {Tv, Tg<:AbstractFloat}
    return h * (yL + yR) / 2
end

# ═══════════════════════════════════════════════════════════════
# Quadratic integration kernels
# Local coordinate: u = x - xL, so u ∈ [0, h] for a full cell.
# Piecewise quadratic: S(u) = a·u² + d·u + y₀
# (a, d are pre-computed coefficients stored in QuadraticInterpolant)
# ═══════════════════════════════════════════════════════════════

# --- Partial-cell integral: ∫_{u0}^{u1} S(u) du ---
@inline function _quadratic_integral_kernel(
    ::_EvalIntegralPartial,
    a::Tv, d::Tv, y0::Tv, u0::Td, u1::Td
) where {Tv, Td<:Real}
    return (a / 3) * (u1^3 - u0^3) + (d / 2) * (u1^2 - u0^2) + y0 * (u1 - u0)
end

# --- Full-cell integral: ∫_0^h S(u) du = a/3·h³ + d/2·h² + y₀·h ---
@inline function _quadratic_integral_kernel(
    ::_EvalIntegralCell,
    a::Tv, d::Tv, y0::Tv, h::Tg
) where {Tv, Tg<:AbstractFloat}
    return (a / 3) * h^3 + (d / 2) * h^2 + y0 * h
end

# ═══════════════════════════════════════════════════════════════
# Constant integration kernels
# Piecewise constant: value depends on side mode (:left, :right, :nearest)
# ═══════════════════════════════════════════════════════════════

# --- :left side — always use left value yL ---
@inline function _constant_integral_kernel(
    ::_EvalIntegralPartial,
    yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::Val{:left}
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return yL * (u1 - u0)
end

# --- :right side — always use right value yR ---
@inline function _constant_integral_kernel(
    ::_EvalIntegralPartial,
    yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::Val{:right}
) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return yR * (u1 - u0)
end

# --- :nearest side — split at midpoint h/2 ---
@inline function _constant_integral_kernel(
    ::_EvalIntegralPartial,
    yL::Tv, yR::Tv, h::Tg, u0::Td, u1::Td, ::Val{:nearest}
) where {Tv, Tg<:AbstractFloat, Td<:Real}
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

@inline _IH00(t) = t - t^3 + t^4 / 2
@inline _IH10(t) = t^2 / 2 - 2t^3 / 3 + t^4 / 4
@inline _IH01(t) = t^3 - t^4 / 2
@inline _IH11(t) = -t^3 / 3 + t^4 / 4

# --- 1D Hermite integral over [u0, u1] in local coordinates ---
# Computes ∫_{u0}^{u1} P(x) dx where P is the cubic Hermite polynomial:
#   P(t) = fL·H₀₀(t) + fR·H₀₁(t) + h·(dfL·H₁₀(t) + dfR·H₁₁(t))
# with t = u/h, dx = h dt
@inline function _hermite_integral_kernel_1d(
    fL, fR, dfL, dfR,
    h::Tg, inv_h::Tg,
    u0::Real, u1::Real
) where {Tg<:AbstractFloat}
    t0 = u0 * inv_h
    t1 = u1 * inv_h
    dH00 = _IH00(t1) - _IH00(t0)
    dH10 = _IH10(t1) - _IH10(t0)
    dH01 = _IH01(t1) - _IH01(t0)
    dH11 = _IH11(t1) - _IH11(t0)
    return h * (fL * dH00 + fR * dH01 + h * (dfL * dH10 + dfR * dH11))
end
