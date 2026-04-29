# ========================================
# PHS Radial Kernels and Weight Functions
# ========================================
#
# Pure math functions for polyharmonic spline interpolation.
# No dependencies. All @inline for zero-overhead inlining into hot loops.
#
# Three groups:
#   A. Radial basis function φ(r) = r^K (odd K), Val{K}-dispatched
#   B. Blend weight function w(d, a) with first and second derivatives
#   C. Smoothing transform unrollers (Eqs. 21–23 from paper)

# ╔══════════════════════════════════════╗
# ║   Group A: Radial Basis Function     ║
# ╚══════════════════════════════════════╝
#
# φ(r) = r^K for odd positive integer K.
# Val{K} dispatch: compiler constant-folds the exponent (e.g. r^3 → r*r*r).
# r = 0 guard: return zero to avoid NaN in derivatives.

"""
    _phs_phi(r, ::Val{K}) -> T

Evaluate the polyharmonic radial basis function φ(r) = r^K.
Returns zero for r ≤ ε to avoid NaN at coincident points.
"""
@inline function _phs_phi(r::T, ::Val{K}) where {T, K}
    r <= zero(T) && return zero(T)
    return r^K
end

"""
    _phs_phi_prime(r, ::Val{K}) -> T

First derivative φ'(r) = K * r^(K-1).
Returns zero for r ≤ ε (derivative undefined/infinite at origin for K=1,
but the weight w_i=0 there so the product w_i*φ'(r)/r is well-defined via L'Hôpital).
"""
@inline function _phs_phi_prime(r::T, ::Val{K}) where {T, K}
    r <= zero(T) && return zero(T)
    return K * r^(K - 1)
end

"""
    _phs_phi_dprime(r, ::Val{K}) -> T

Second derivative φ''(r) = K*(K-1) * r^(K-2).
Returns zero for r ≤ ε and for K=1 (since K*(K-1)=0).
"""
@inline function _phs_phi_dprime(r::T, ::Val{K}) where {T, K}
    (r <= zero(T) || K == 1) && return zero(T)
    return K * (K - 1) * r^(K - 2)
end

# Convenience scalar dispatch (runtime degree) — used during construction only
@inline _phs_phi(r, k::Int) = _phs_phi(r, Val(k))
@inline _phs_phi_prime(r, k::Int) = _phs_phi_prime(r, Val(k))
@inline _phs_phi_dprime(r, k::Int) = _phs_phi_dprime(r, Val(k))

# ╔══════════════════════════════════════╗
# ║    Group B: Blend Weight Function    ║
# ╚══════════════════════════════════════╝
#
# w(d, a) = exp(d³ / (d³ - a³))   for d < a
#          = 0                     for d ≥ a
#
# This is the form from the implementation snippets, equivalent to the paper's
# Eq. 27 formulation.  It has a maximum of e at d→0⁺ (note: the paper normalizes
# the sum, so the absolute scale cancels), goes smoothly to 0 at d=a with
# continuous first and second derivatives.

"""
    _phs_blend_weight(d::T, a::T) -> T

Evaluate the blend weight w(d, a). Returns zero for d ≥ a.
"""
@inline function _phs_blend_weight(d::T, a::T) where {T}
    d >= a && return zero(T)
    d3 = d * d * d
    a3 = a * a * a
    return exp(d3 / (d3 - a3))
end

"""
    _phs_blend_weight_and_prime(d::T, a::T) -> (w, wp)

Evaluate w and its first derivative w'(d) simultaneously.
w'(d) = -3a³d² * w / (d³ - a³)²
"""
@inline function _phs_blend_weight_and_prime(d::T, a::T) where {T}
    if d >= a
        return zero(T), zero(T)
    end
    d2 = d * d
    d3 = d2 * d
    a3 = a * a * a
    denom = d3 - a3        # negative (d < a)
    w = exp(d3 / denom)
    wp = -3 * a3 * d2 * w / (denom * denom)
    return w, wp
end

"""
    _phs_blend_weight_and_derivs(d::T, a::T) -> (w, wp, wpp)

Evaluate w, w', and w'' simultaneously for use in second-derivative blending.
wpp = 3a³*d * (-2a⁶ + a³*d³ + 4*d⁶) * w / (d³ - a³)⁴
(This matches the snippet's weifun second-derivative formula.)
"""
@inline function _phs_blend_weight_and_derivs(d::T, a::T) where {T}
    if d >= a
        return zero(T), zero(T), zero(T)
    end
    d2 = d * d
    d3 = d2 * d
    a3 = a * a * a
    denom = d3 - a3        # negative
    denom2 = denom * denom
    denom4 = denom2 * denom2
    w = exp(d3 / denom)
    wp = -3 * a3 * d2 * w / denom2
    # inner = -2*a⁶ + a³*d³ + 4*d⁴  (matches snippet's (-2*a3^2 + a3*x3 + 4*x2^2))
    inner = muladd(4 * d2, d2, muladd(a3, d3, -2 * a3 * a3))
    wpp = 3 * a3 * d * inner * w / denom4
    return w, wp, wpp
end

# ╔══════════════════════════════════════╗
# ║   Group C: Smoothing Transform       ║
# ╚══════════════════════════════════════╝
#
# The log-density smoothing transformation f(x) = ln(ρ(x)/ρ₀(x)).
# Interpolation is performed on f, then the result is unrolled to ρ via:
#   ρ̃  = ρ₀ * exp(f)           [Eq. 21]
#   ρ̃ξ = ρ̃ * (fξ + ρ₀ξ/ρ₀)  [Eq. 22]
#   ρ̃ξζ = ρ̃ * (fξζ + ρ̃ξρ̃ζ/ρ̃² + ρ₀ξζ/ρ₀ - ρ₀ξρ₀ζ/ρ₀²) [Eq. 23]

"""
    _phs_unroll_value(f::T, rho0::T) -> T

Recover interpolated density from smooth function: ρ̃ = ρ₀ * exp(f).
"""
@inline function _phs_unroll_value(f::T, rho0::T) where {T}
    (rho0 < 1e-40 || f > 100) && return zero(T)
    return rho0 * exp(f)
end

"""
    _phs_unroll_grad_component(rho, f_grad_xi, rho0_grad_xi, rho0) -> T

Eq. 22: ρ̃ξ = ρ̃ * (fξ + ρ₀ξ/ρ₀).
Compute one component of the gradient of the recovered density.
"""
@inline function _phs_unroll_grad_component(rho::T, f_grad_xi::T, rho0_grad_xi::T, rho0::T) where {T}
    rho0_safe = max(rho0, T(1e-40))
    return rho * (f_grad_xi + rho0_grad_xi / rho0_safe)
end

"""
    _phs_unroll_hess_component(rho, f_hess_xixj, rho_grad_xi, rho_grad_xj, rho0, rho0_grad_xi, rho0_grad_xj, rho0_hess_xixj) -> T

Eq. 23: ρ̃ξζ = ρ̃ * (fξζ + ρ̃ξρ̃ζ/ρ̃² + ρ₀ξζ/ρ₀ - ρ₀ξρ₀ζ/ρ₀²).
Compute one component (ξ,ζ) of the Hessian of the recovered density.
"""
@inline function _phs_unroll_hess_component(
        rho::T,
        f_hess_xixj::T,
        rho_grad_xi::T,
        rho_grad_xj::T,
        rho0::T,
        rho0_grad_xi::T,
        rho0_grad_xj::T,
        rho0_hess_xixj::T,
    ) where {T}
    rho_safe = max(rho, T(1e-40))
    rho0_safe = max(rho0, T(1e-40))
    return rho * (
        f_hess_xixj +
        rho_grad_xi * rho_grad_xj / (rho_safe * rho_safe) +
        rho0_hess_xixj / rho0_safe -
        rho0_grad_xi * rho0_grad_xj / (rho0_safe * rho0_safe)
    )
end
