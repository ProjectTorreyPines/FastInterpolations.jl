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
Returns zero for r ≤ 0 (coincident points) to avoid NaN.
"""
@inline function _phs_phi(r::T, ::Val{K}) where {T, K}
    r <= zero(T) && return zero(T)
    return r^K
end
# Specializations for common odd degrees — explicit multiplications avoid the
# general pow path and enable better compiler optimisation.
# No r≤0 guard needed: for K≥1 the multiplication chains evaluate to 0 at r=0,
# so removing the guard makes these functions branch-free and SIMD-friendly.
@inline _phs_phi(r::T, ::Val{1}) where {T} = r
@inline _phs_phi(r::T, ::Val{3}) where {T} = r * r * r
@inline function _phs_phi(r::T, ::Val{5}) where {T}
    r2 = r * r
    return r2 * r2 * r
end
@inline function _phs_phi(r::T, ::Val{7}) where {T}
    r2 = r * r
    r4 = r2 * r2
    return r4 * r2 * r
end

"""
    _phs_phi_prime(r, ::Val{K}) -> T

First derivative φ'(r) = K * r^(K-1).
Returns zero for r ≤ 0. (φ'(r)/r — the quantity used downstream — is singular at
the origin for K=1, but w_i=0 there so the product w_i*φ'(r)/r is well-defined.)
"""
@inline function _phs_phi_prime(r::T, ::Val{K}) where {T, K}
    r <= zero(T) && return zero(T)
    return K * r^(K - 1)
end
@inline function _phs_phi_prime(r::T, ::Val{1}) where {T}
    r <= zero(T) && return zero(T)
    return one(T)
end
# No guard for K=3,5,7: 3*0²=0, 5*0⁴=0, 7*0⁶=0 — always correct at r=0.
@inline _phs_phi_prime(r::T, ::Val{3}) where {T} = 3 * r * r
@inline function _phs_phi_prime(r::T, ::Val{5}) where {T}
    r2 = r * r
    return 5 * r2 * r2
end
@inline function _phs_phi_prime(r::T, ::Val{7}) where {T}
    r2 = r * r
    return 7 * r2 * r2 * r2
end

"""
    _phs_phi_dprime(r, ::Val{K}) -> T

Second derivative φ''(r) = K*(K-1) * r^(K-2).
Returns zero for r ≤ 0 and for K=1 (since K*(K-1)=0).
"""
@inline function _phs_phi_dprime(r::T, ::Val{K}) where {T, K}
    (r <= zero(T) || K == 1) && return zero(T)
    return K * (K - 1) * r^(K - 2)
end
@inline _phs_phi_dprime(r::T, ::Val{1}) where {T} = zero(T)
# No guard for K=3,5,7: 6*0=0, 20*0³=0, 42*0⁵=0 — always correct at r=0.
@inline _phs_phi_dprime(r::T, ::Val{3}) where {T} = 6 * r
@inline function _phs_phi_dprime(r::T, ::Val{5}) where {T}
    return 20 * r * r * r
end
@inline function _phs_phi_dprime(r::T, ::Val{7}) where {T}
    r2 = r * r
    return 42 * r2 * r2 * r
end

# Convenience scalar dispatch (runtime degree) — used during construction only
@inline _phs_phi(r, k::Int) = _phs_phi(r, Val(k))
@inline _phs_phi_prime(r, k::Int) = _phs_phi_prime(r, Val(k))
@inline _phs_phi_dprime(r, k::Int) = _phs_phi_dprime(r, Val(k))

# ╔══════════════════════════════════════╗
# ║    Group B: Blend Weight Function    ║
# ╚══════════════════════════════════════╝
#
# The reference Fortran implementation (critic2 / grinterp_smr) uses a
# dimensionless (scale-invariant) form that differs from the printed paper
# Eq. 27.  The Fortran comment explicitly states:
#   "this version of weifun is different from the article.  The argument
#    of the exponential is adimensional in this version, and prevents
#    problems with underflows in very fine grids."
#
# Fortran weifun (what actually produces the paper's figures):
#   w(d, a) = exp( d³ / (d³ - a³) )           for d < a
#            = 0                                for d ≥ a
#
# The argument d³/(d³-a³) is dimensionless: it depends only on the ratio d/a,
# not on the physical scale of a.  The paper formula exp(d³/(a³(d³-a³))) suffers
# from underflow/overflow when a is very small (fine grids) because the a³
# factor in the denominator can make the exponent enormous.
#
# Derivatives (let u = d³/(d³-a³), so du/dd = -3a³d²/(d³-a³)²):
#   w'(d)  = -3a³d² w / (d³ - a³)²
#   w''(d) = 3a³d(-2a⁶ + a³d³ + 4d⁶) w / (d³ - a³)⁴

"""
    _phs_blend_weight(d::T, a::T) -> T

Evaluate the blend weight w(d, a) using the dimensionless Fortran formula
w = exp(d³/(d³-a³)).  Returns zero for d ≥ a.
"""
@inline function _phs_blend_weight(d::T, a::T) where {T}
    d >= a && return zero(T)
    d3 = d * d * d
    a3 = a * a * a
    return exp(d3 / (d3 - a3))
end

@inline function _phs_blend_weight(d::T, a::T, a3::T) where {T}
    d >= a && return zero(T)
    d3 = d * d * d
    return exp(d3 / (d3 - a3))
end

"""
    _phs_blend_weight_and_prime(d::T, a::T) -> (w, wp)

Evaluate w and its first derivative w'(d) = -3a³d² w / (d³ - a³)² simultaneously.
"""
@inline function _phs_blend_weight_and_prime(d::T, a::T) where {T}
    if d >= a
        return zero(T), zero(T)
    end
    d2 = d * d
    d3 = d2 * d
    a3 = a * a * a
    denom = d3 - a3          # negative (d < a)
    denom2 = denom * denom
    w = exp(d3 / denom)
    wp = -3 * a3 * d2 * w / denom2
    return w, wp
end

@inline function _phs_blend_weight_and_prime(d::T, a::T, a3::T) where {T}
    if d >= a
        return zero(T), zero(T)
    end
    d2 = d * d
    d3 = d2 * d
    denom = d3 - a3
    denom2 = denom * denom
    w = exp(d3 / denom)
    wp = -3 * a3 * d2 * w / denom2
    return w, wp
end

"""
    _phs_blend_weight_and_derivs(d::T, a::T) -> (w, wp, wpp)

Evaluate w, w', and w'' simultaneously for use in second-derivative blending.

  w'(d)  = -3a³d² w / (d³ - a³)²
  w''(d) = 3a³d(-2a⁶ + a³d³ + 4d⁶) w / (d³ - a³)⁴
"""
@inline function _phs_blend_weight_and_derivs(d::T, a::T) where {T}
    if d >= a
        return zero(T), zero(T), zero(T)
    end
    d2 = d * d
    d3 = d2 * d
    a3 = a * a * a
    denom = d3 - a3
    inv_denom = one(T) / denom
    inv_denom2 = inv_denom * inv_denom
    inv_denom4 = inv_denom2 * inv_denom2
    w = exp(d3 * inv_denom)
    wp = -3 * a3 * d2 * w * inv_denom2
    wpp = 3 * a3 * d * (muladd(4 * d3, d3, muladd(a3, d3, -2 * a3 * a3))) * w * inv_denom4
    return w, wp, wpp
end

@inline function _phs_blend_weight_and_derivs(d::T, a::T, a3::T) where {T}
    if d >= a
        return zero(T), zero(T), zero(T)
    end
    d2 = d * d
    d3 = d2 * d
    denom = d3 - a3
    inv_denom = one(T) / denom
    inv_denom2 = inv_denom * inv_denom
    inv_denom4 = inv_denom2 * inv_denom2
    w = exp(d3 * inv_denom)
    wp = -3 * a3 * d2 * w * inv_denom2
    wpp = 3 * a3 * d * (muladd(4 * d3, d3, muladd(a3, d3, -2 * a3 * a3))) * w * inv_denom4
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
    (rho0 < 1.0e-40 || f > 100) && return zero(T)
    return rho0 * exp(f)
end

"""
    _phs_unroll_grad_component(rho, f_grad_xi, rho0_grad_xi, rho0) -> T

Eq. 22: ρ̃ξ = ρ̃ * (fξ + ρ₀ξ/ρ₀).
Compute one component of the gradient of the recovered density.
"""
@inline function _phs_unroll_grad_component(rho::T, f_grad_xi::T, rho0_grad_xi::T, rho0::T) where {T}
    rho0_safe = max(rho0, T(1.0e-40))
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
    rho_safe = max(rho, T(1.0e-40))
    rho0_safe = max(rho0, T(1.0e-40))
    return rho * (
        f_hess_xixj +
            rho_grad_xi * rho_grad_xj / (rho_safe * rho_safe) +
            rho0_hess_xixj / rho0_safe -
            rho0_grad_xi * rho0_grad_xj / (rho0_safe * rho0_safe)
    )
end

# ╔════════════════════════════════════════════╗
# ║   Group D: Polynomial Augmentation Helpers ║
# ╚════════════════════════════════════════════╝
#
# For r^K PHS interpolation the polynomial augmentation must have degree
# at least  poly_deg = (K-1)÷2:
#   K=1  →  poly_deg=0  (constants only,   1 term in N dims)
#   K=3  →  poly_deg=1  (linear,          N+1 terms)
#   K=5  →  poly_deg=2  (quadratic, C(N+2,2) terms)
#   K=7  →  poly_deg=3  (cubic,     C(N+3,3) terms)
#
# Monomial ordering: increasing total degree, last index varies fastest.
#   N=2, poly_deg=2: (0,0),(0,1),(1,0),(0,2),(1,1),(2,0)

"""
    _phs_n_poly(N, poly_deg) -> Int

Number of polynomial basis functions in N dimensions up to total degree `poly_deg`.
Equal to C(N+poly_deg, poly_deg).
"""
@inline _phs_n_poly(N::Int, poly_deg::Int) = binomial(N + poly_deg, poly_deg)

"""
    _phs_all_exponents(::Val{N}, poly_deg) -> Vector{NTuple{N,Int}}

All exponent vectors for monomials of total degree ≤ `poly_deg` in N dimensions.
Called at stencil-construction time only (not the hot path).
"""
function _phs_all_exponents(::Val{N}, poly_deg::Int) where {N}
    result = NTuple{N, Int}[]
    current = zeros(Int, N - 1)   # dims 1…N-1; dim N is `remaining`
    function gen(d::Int, remaining::Int)
        if d == N
            push!(result, ntuple(i -> i < N ? current[i] : remaining, Val(N)))
            return
        end
        for k in 0:remaining
            current[d] = k
            gen(d + 1, remaining - k)
        end
        return
    end
    for total in 0:poly_deg
        gen(1, total)
    end
    return result
end

"""
    _phs_poly_exps_tuple(::Val{N}, ::Val{K})

Return a compile-time `NTuple` of `NTuple{N,Int}` exponents for the polynomial
augmentation of the r^K PHS interpolant (poly_deg = (K-1)÷2).
Generated at compile time — zero allocation, loops fully unrolled.
"""
@generated function _phs_poly_exps_tuple(::Val{N}, ::Val{K}) where {N, K}
    m = (K - 1) ÷ 2
    exps = NTuple{N, Int}[]
    current = zeros(Int, N - 1)
    function gen(d, remaining)
        if d == N
            push!(exps, ntuple(i -> i < N ? current[i] : remaining, N))
            return
        end
        for k in 0:remaining
            current[d] = k
            gen(d + 1, remaining - k)
        end
        return
    end
    for total in 0:m
        gen(1, total)
    end
    tup = Expr(:tuple, [Expr(:tuple, α...) for α in exps]...)
    return :($tup)
end

"""
    _phs_eval_poly(Δx, poly_exps, coeffs, ns) -> scalar

Evaluate the polynomial augmentation: `Σ_k c[ns+k] · Δx^α_k`.
`poly_exps` is a (compile-time) tuple of exponent NTuples from
`_phs_poly_exps_tuple`.
"""
@inline function _phs_eval_poly(
        Δx::NTuple{N, Tg},
        poly_exps::Tuple,
        coeffs::AbstractVector{Tv},
        ns::Int,
    ) where {N, Tg, Tv}
    y = zero(Tv)
    @inbounds for k in 1:length(poly_exps)
        α = poly_exps[k]
        mono = one(Tg)
        for d in 1:N
            α[d] != 0 && (mono *= Δx[d]^α[d])
        end
        y += coeffs[ns + k] * mono
    end
    return y
end

"""
    _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, axis) -> scalar

Evaluate `∂/∂x_axis` of the polynomial augmentation.
"""
@inline function _phs_eval_poly_deriv1(
        Δx::NTuple{N, Tg},
        poly_exps::Tuple,
        coeffs::AbstractVector{Tv},
        ns::Int,
        axis::Int,
    ) where {N, Tg, Tv}
    return _phs_eval_poly_deriv1(Δx, poly_exps, coeffs, ns, Val(axis))
end

@inline function _phs_eval_poly_deriv1(
        Δx::NTuple{N, Tg},
        poly_exps::Tuple,
        coeffs::AbstractVector{Tv},
        ns::Int,
        ::Val{axis},
    ) where {N, Tg, Tv, axis}
    y = zero(Tv)
    @inbounds for k in 1:length(poly_exps)
        α = poly_exps[k]
        α[axis] == 0 && continue
        mono = Tg(α[axis])
        for d in 1:N
            exp = d == axis ? α[d] - 1 : α[d]
            exp != 0 && (mono *= Δx[d]^exp)
        end
        y += coeffs[ns + k] * mono
    end
    return y
end

"""
    _phs_eval_poly_deriv2(Δx, poly_exps, coeffs, ns, ax1, ax2) -> scalar

Evaluate `∂²/∂x_ax1 ∂x_ax2` of the polynomial augmentation.
"""
@inline function _phs_eval_poly_deriv2(
        Δx::NTuple{N, Tg},
        poly_exps::Tuple,
        coeffs::AbstractVector{Tv},
        ns::Int,
        ax1::Int,
        ax2::Int,
    ) where {N, Tg, Tv}
    return _phs_eval_poly_deriv2(Δx, poly_exps, coeffs, ns, Val(ax1), Val(ax2))
end

@inline function _phs_eval_poly_deriv2(
        Δx::NTuple{N, Tg},
        poly_exps::Tuple,
        coeffs::AbstractVector{Tv},
        ns::Int,
        ::Val{ax1},
        ::Val{ax2},
    ) where {N, Tg, Tv, ax1, ax2}
    y = zero(Tv)
    @inbounds for k in 1:length(poly_exps)
        α = poly_exps[k]
        if ax1 == ax2
            α[ax1] < 2 && continue
            mono = Tg(α[ax1] * (α[ax1] - 1))
            for d in 1:N
                exp = d == ax1 ? α[d] - 2 : α[d]
                exp != 0 && (mono *= Δx[d]^exp)
            end
            y += coeffs[ns + k] * mono
        else
            (α[ax1] == 0 || α[ax2] == 0) && continue
            mono = Tg(α[ax1] * α[ax2])
            for d in 1:N
                exp = (d == ax1 || d == ax2) ? α[d] - 1 : α[d]
                exp != 0 && (mono *= Δx[d]^exp)
            end
            y += coeffs[ns + k] * mono
        end
    end
    return y
end

# ========================================
# Generated/Unrolled Tuple Helpers for SIMD
# ========================================

"""
    _phs_diff(query, base_coords, off, hs_local) -> NTuple{N, Tg}

Compute physical distance vectors: (query - node_i).
Generated at compile-time for arbitrary dimensions to ensure full unrolling
and optimal SIMD register allocation.
"""
@generated function _phs_diff(
        query::NTuple{N, <:Real},
        base_coords::NTuple{N, Tg},
        off::NTuple{N, Int},
        hs_local::NTuple{N, Tg},
    ) where {N, Tg}
    exprs = [:(Tg(query[$d]) - (base_coords[$d] + Tg(off[$d]) * hs_local[$d])) for d in 1:N]
    return Expr(:tuple, exprs...)
end

"""
    _phs_sum_sq(x) -> Tg

Compile-time unrolled sum of squares for tuples.
"""
@generated function _phs_sum_sq(x::NTuple{N, T}) where {N, T}
    exprs = [:(x[$d] * x[$d]) for d in 1:N]
    return Expr(:call, :+, exprs...)
end

"""
    _phs_diff_Δ(Δx, phys_off) -> NTuple{N, Tg}

Compute physical distance vectors from physical coordinate difference Δx (query - base_node) and precomputed physical offset.
Generated at compile-time for arbitrary dimensions to ensure full unrolling and optimal SIMD register allocation.
"""
@generated function _phs_diff_Δ(
        Δx::NTuple{N, Tg},
        phys_off::NTuple{N, Tg},
    ) where {N, Tg}
    exprs = [:(Δx[$d] - phys_off[$d]) for d in 1:N]
    return Expr(:tuple, exprs...)
end
