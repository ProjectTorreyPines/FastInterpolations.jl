# ========================================
# Hermite Mathematical Functions (Generic)
# ========================================
#
# Pure mathematical functions for Hermite interpolation shared across all dimensions:
# - Hermite basis functions (h00, h10, h01, h11)
# - 1D Hermite kernel for value and derivatives
# - Moment-to-derivative conversion (1D)
#
# Type-generic: Works with Float32, Float64, Complex, and ForwardDiff.Dual
#
# 2D-specific batch solvers are in nd_math_2d.jl (temporary).
#
# Note: These functions use the OUTPUT type (Tv or promoted type) for
# intermediate calculations to support complex values and AD.

# ========================================
# HERMITE BASIS FUNCTIONS
# ========================================
#
# Cubic Hermite interpolation uses function values AND derivative values
# at interval endpoints to construct a smooth C1-continuous interpolant.
#
# Given interval [xL, xR] with:
#   - yL, yR: function values at endpoints
#   - dyL, dyR: derivative values (dy/dx, unscaled)
#   - normalized position t ∈ [0,1]
#
# The Hermite polynomial is:
#   P(t) = h00(t)*yL + h10(t)*(h*dyL) + h01(t)*yR + h11(t)*(h*dyR)
#
# Basis functions:
#   h00(t) = 2t³ - 3t² + 1   (left endpoint value basis)
#   h10(t) = t³ - 2t² + t    (left endpoint derivative basis)
#   h01(t) = -2t³ + 3t²      (right endpoint value basis)
#   h11(t) = t³ - t²         (right endpoint derivative basis)

"""
    _hermite_h00(t)

Left endpoint value basis function: h00(t) = 2t³ - 3t² + 1
"""
@inline function _hermite_h00(t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    return muladd(T(2), t3, muladd(T(-3), t2, one(T)))
end

"""
    _hermite_h10(t)

Left endpoint derivative basis function: h10(t) = t³ - 2t² + t
"""
@inline function _hermite_h10(t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    return muladd(t3, one(T), muladd(T(-2), t2, t))
end

"""
    _hermite_h01(t)

Right endpoint value basis function: h01(t) = -2t³ + 3t²
"""
@inline function _hermite_h01(t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    return muladd(T(-2), t3, muladd(T(3), t2, zero(T)))
end

"""
    _hermite_h11(t)

Right endpoint derivative basis function: h11(t) = t³ - t²
"""
@inline function _hermite_h11(t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    return t3 - t2
end

# ========================================
# HERMITE 1D EVALUATION KERNELS
# ========================================
# Dispatch on EvalValue, EvalDeriv1, EvalDeriv2, EvalDeriv3
#
# Arguments:
# - yL, yR: Function values at left/right endpoints (type Tv)
# - dyL, dyR: Derivative values at endpoints (type Tv)
# - h: Interval width (type Tg)
# - inv_h: 1/h precomputed (type Tg)
# - dL: Distance from left endpoint xq - xL (type Tg or query type)

"""
    _hermite_kernel_1d(::EvalValue, yL, yR, dyL, dyR, h, inv_h, dL)

Evaluate cubic Hermite interpolation value at a point within interval.

Returns interpolated function value.
"""
@inline function _hermite_kernel_1d(
    ::EvalValue,
    yL::Tv, yR::Tv, dyL::Tv, dyR::Tv,
    h::Tg, inv_h::Tg, dL::Tq
) where {Tv, Tg, Tq}
    # Promote to output type for intermediate calculations
    T = promote_type(Tv, Tq)
    t = T(dL) * T(inv_h)

    h00_val = _hermite_h00(t)
    h10_val = _hermite_h10(t)
    h01_val = _hermite_h01(t)
    h11_val = _hermite_h11(t)

    value_contrib = muladd(h00_val, T(yL), h01_val * T(yR))
    deriv_contrib = muladd(h10_val, T(dyL), h11_val * T(dyR)) * T(h)

    return value_contrib + deriv_contrib
end

"""
    _hermite_kernel_1d(::EvalDeriv1, yL, yR, dyL, dyR, h, inv_h, dL)

Evaluate first derivative: dP/dx = (dP/dt) / h
"""
@inline function _hermite_kernel_1d(
    ::EvalDeriv1,
    yL::Tv, yR::Tv, dyL::Tv, dyR::Tv,
    h::Tg, inv_h::Tg, dL::Tq
) where {Tv, Tg, Tq}
    T = promote_type(Tv, Tq)
    t = T(dL) * T(inv_h)
    t2 = t * t

    # Derivatives of basis functions w.r.t. t
    dh00_dt = muladd(T(6), t2, T(-6) * t)         # 6t² - 6t
    dh10_dt = muladd(T(3), t2, muladd(T(-4), t, one(T)))  # 3t² - 4t + 1
    dh01_dt = muladd(T(-6), t2, T(6) * t)         # -6t² + 6t
    dh11_dt = muladd(T(3), t2, T(-2) * t)         # 3t² - 2t

    value_contrib = muladd(dh00_dt, T(yL), dh01_dt * T(yR))
    deriv_contrib = muladd(dh10_dt, T(dyL), dh11_dt * T(dyR)) * T(h)

    dP_dt = value_contrib + deriv_contrib
    return dP_dt * T(inv_h)
end

"""
    _hermite_kernel_1d(::EvalDeriv2, yL, yR, dyL, dyR, h, inv_h, dL)

Evaluate second derivative: d²P/dx² = (d²P/dt²) / h²
"""
@inline function _hermite_kernel_1d(
    ::EvalDeriv2,
    yL::Tv, yR::Tv, dyL::Tv, dyR::Tv,
    h::Tg, inv_h::Tg, dL::Tq
) where {Tv, Tg, Tq}
    T = promote_type(Tv, Tq)
    t = T(dL) * T(inv_h)

    # Second derivatives of basis functions w.r.t. t
    d2h00_dt2 = muladd(T(12), t, T(-6))     # 12t - 6
    d2h10_dt2 = muladd(T(6), t, T(-4))      # 6t - 4
    d2h01_dt2 = muladd(T(-12), t, T(6))     # -12t + 6
    d2h11_dt2 = muladd(T(6), t, T(-2))      # 6t - 2

    value_contrib = muladd(d2h00_dt2, T(yL), d2h01_dt2 * T(yR))
    deriv_contrib = muladd(d2h10_dt2, T(dyL), d2h11_dt2 * T(dyR)) * T(h)

    d2P_dt2 = value_contrib + deriv_contrib
    inv_h2 = T(inv_h) * T(inv_h)
    return d2P_dt2 * inv_h2
end

"""
    _hermite_kernel_1d(::EvalDeriv3, yL, yR, dyL, dyR, h, inv_h, dL)

Evaluate third derivative: d³P/dx³ = (d³P/dt³) / h³ (constant within interval)
"""
@inline function _hermite_kernel_1d(
    ::EvalDeriv3,
    yL::Tv, yR::Tv, dyL::Tv, dyR::Tv,
    h::Tg, inv_h::Tg, dL::Tq
) where {Tv, Tg, Tq}
    T = promote_type(Tv, Tq)
    # Third derivatives are constants: d³h00/dt³=12, d³h10/dt³=6, d³h01/dt³=-12, d³h11/dt³=6
    value_contrib = T(12) * (T(yL) - T(yR))
    deriv_contrib = T(6) * T(h) * (T(dyL) + T(dyR))

    d3P_dt3 = value_contrib + deriv_contrib
    inv_h3 = T(inv_h) * T(inv_h) * T(inv_h)
    return d3P_dt3 * inv_h3
end

# ========================================
# MOMENT-TO-DERIVATIVE CONVERSION (1D)
# ========================================
#
# Convert cubic spline moments (second derivatives) to first derivatives
# for use in Hermite interpolation construction.
#
# Mathematical Foundation:
# Cubic splines solve for moments m[i] = f''(x[i]) via tridiagonal systems.
# Given moments, we compute first derivatives dy[i] = f'(x[i]):
#   S'(x[i]) from right = (y[i+1] - y[i])/h - h*(2*m[i] + m[i+1])/6
#   S'(x[i+1]) from left = (y[i+1] - y[i])/h + h*(m[i] + 2*m[i+1])/6

"""
    _moments_to_derivatives_1d!(dydx, m, y, spacing)

Convert cubic spline moments (second derivatives) to first derivatives.
In-place version that modifies `dydx`.

# Arguments
- `dydx::AbstractVector{Tv}`: Output first derivatives (modified in-place)
- `m::AbstractVector{Tv}`: Input moments (second derivatives)
- `y::AbstractVector{Tv}`: Function values at grid points
- `spacing::AbstractGridSpacing{Tg}`: Grid spacing information
"""
function _moments_to_derivatives_1d!(
    dydx::AbstractVector{Tv},
    m::AbstractVector{Tv},
    y::AbstractVector{Tv},
    spacing::AbstractGridSpacing{Tg}
) where {Tv, Tg}
    n = length(y)
    @assert length(dydx) == n "dydx length mismatch"
    @assert length(m) == n "m length mismatch"
    @assert n >= 2 "Need at least 2 points"

    inv_6 = Tv(1) / Tv(6)

    # First point (using right derivative from first interval)
    @inbounds begin
        h1 = Tv(_get_h(spacing, 1))
        inv_h1 = Tv(_get_inv_h(spacing, 1))
        h_over_6 = h1 * inv_6
        linear_slope = (y[2] - y[1]) * inv_h1
        moment_sum = muladd(Tv(2), m[1], m[2])
        dydx[1] = muladd(-h_over_6, moment_sum, linear_slope)
    end

    # Interior and last points (using left derivative from each interval)
    @inbounds for i in 1:(n-1)
        h = Tv(_get_h(spacing, i))
        inv_h = Tv(_get_inv_h(spacing, i))
        h_over_6 = h * inv_6
        linear_slope = (y[i+1] - y[i]) * inv_h
        moment_sum = muladd(Tv(2), m[i+1], m[i])
        dydx[i+1] = muladd(h_over_6, moment_sum, linear_slope)
    end

    return dydx
end

"""
    _apply_derivative_bc!(dydx, bc, endpoints)

Apply boundary condition corrections to computed derivatives.
Called after _moments_to_derivatives_1d! to enforce specific BC values.
"""
function _apply_derivative_bc!(dydx::AbstractVector{Tv}, bc::BCPair, args...) where {Tv}
    if bc.left isa Deriv1
        @inbounds dydx[1] = Tv(bc.left.val)
    end
    if bc.right isa Deriv1
        @inbounds dydx[end] = Tv(bc.right.val)
    end
    return nothing
end

function _apply_derivative_bc!(dydx::AbstractVector{Tv}, bc::PeriodicData, args...) where {Tv}
    # Enforce periodic: dydx[1] == dydx[end]
    avg = (dydx[1] + dydx[end]) / Tv(2)
    @inbounds dydx[1] = avg
    @inbounds dydx[end] = avg
    return nothing
end

# Fallback (no-op for other BC types)
function _apply_derivative_bc!(dydx, bc, args...)
    return nothing
end
