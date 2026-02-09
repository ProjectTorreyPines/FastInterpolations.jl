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
