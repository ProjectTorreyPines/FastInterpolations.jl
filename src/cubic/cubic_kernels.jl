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

"""
    _cubic_kernel(::EvalDeriv3, zL, zR, yL, yR, h, inv_h, dL, dR)

Third derivative of cubic spline (constant within each interval).

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
    zL::T, zR::T, ::T, ::T, ::T, inv_h::T, ::T, ::T
) where {T}
    return (zR - zL) * inv_h
end


# ========================================
# Endpoint Derivative Estimation Kernels
# ========================================
# Scalar kernels for computing endpoint derivatives from data using
# 4-point one-sided Lagrange polynomial differentiation.
#
# These are the SINGLE SOURCE OF TRUTH for Lagrange stencil coefficients.
# Used by LagrangeBC in 1D, batch, and ND paths. Any change to these
# coefficients should only be made here.
#
# Mathematical basis: The Lagrange interpolating polynomial through n points
# has derivative formula that is exact for polynomials of degree n-1 or less.
# For 4 points, this is 4th-order accurate for smooth functions.

"""
    _lagrange_d1_left_uniform(f1, f2, f3, f4, inv_h) -> T

Compute first derivative at LEFT endpoint using 4-point Lagrange formula.

For uniform grid with spacing h:
    f'(x₁) ≈ (-11f₁ + 18f₂ - 9f₃ + 2f₄) / (6h)

# Arguments
- `f1, f2, f3, f4::T`: Function values at first 4 grid points
- `inv_h::T`: Inverse of uniform grid spacing (1/h)

# Returns
- Estimated derivative f'(x₁)

# Mathematical Derivation
This formula comes from differentiating the unique cubic polynomial P(x)
passing through (x₁,f₁), (x₂,f₂), (x₃,f₃), (x₄,f₄) and evaluating P'(x₁).

For uniform grid x_i = x₁ + (i-1)h, the Lagrange derivative formula becomes:
    P'(x₁) = [(-11)f₁ + 18f₂ + (-9)f₃ + 2f₄] / (6h)

# Accuracy
- Exact for polynomials of degree ≤ 3 (cubic and below)
- 4th-order accurate O(h⁴) for smooth functions

# Operation Count
    0 fdiv + 2 fmul + 3 fmadd = 5 FP ops (excluding inv_h computation)
"""
@inline function _lagrange_d1_left_uniform(f1::T, f2::T, f3::T, f4::T, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-11, 18, -9, 2) / 6
    # Using muladd chain for consistent FMA behavior and optimal accuracy
    # Structured as: coeff * (-11*f1 + 18*f2 - 9*f3 + 2*f4)
    coeff = inv_h / 6
    numer = muladd(T(-11), f1, muladd(T(18), f2, muladd(T(-9), f3, T(2) * f4)))
    return coeff * numer
end

"""
    _lagrange_d1_right_uniform(fnm3, fnm2, fnm1, fn, inv_h) -> T

Compute first derivative at RIGHT endpoint using 4-point Lagrange formula.

For uniform grid with spacing h:
    f'(xₙ) ≈ (-2fₙ₋₃ + 9fₙ₋₂ - 18fₙ₋₁ + 11fₙ) / (6h)

# Arguments
- `fnm3, fnm2, fnm1, fn::T`: Function values at last 4 grid points
  (n-3, n-2, n-1, n in 1-based indexing)
- `inv_h::T`: Inverse of uniform grid spacing (1/h)

# Returns
- Estimated derivative f'(xₙ)

# Mathematical Derivation
Mirror of the left formula, obtained by differentiating the Lagrange
polynomial through the last 4 points and evaluating at xₙ.

For uniform grid:
    P'(xₙ) = [(-2)fₙ₋₃ + 9fₙ₋₂ + (-18)fₙ₋₁ + 11fₙ] / (6h)

Note: Coefficients are negated/reversed from left formula due to symmetry:
    Left:  (-11, 18, -9, 2)
    Right: (-2, 9, -18, 11)

# Accuracy
- Exact for polynomials of degree ≤ 3 (cubic and below)
- 4th-order accurate O(h⁴) for smooth functions

# Operation Count
    0 fdiv + 2 fmul + 3 fmadd = 5 FP ops (excluding inv_h computation)
"""
@inline function _lagrange_d1_right_uniform(fnm3::T, fnm2::T, fnm1::T, fn::T, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-2, 9, -18, 11) / 6
    coeff = inv_h / 6
    numer = muladd(T(-2), fnm3, muladd(T(9), fnm2, muladd(T(-18), fnm1, T(11) * fn)))
    return coeff * numer
end


# ========================================
# Unified Endpoint Derivative Estimation
# ========================================
# Higher-level API using Val{:left}/Val{:right} dispatch.
# Handles both uniform (Range) and non-uniform (Vector) grids.
#
# These functions are used by LagrangeBC in the cubic solver to automatically
# compute endpoint derivatives from data without user-specified values.

"""
    _estimate_endpoint_derivative(xs::AbstractRange, ys, ::Val{:left}) -> T

Estimate first derivative at LEFT endpoint for UNIFORM grid using 4-point Lagrange formula.

# Arguments
- `xs::AbstractRange{T}`: Uniform grid (Range type)
- `ys::AbstractVector{T}`: Function values (must have ≥4 elements)
- `::Val{:left}`: Dispatch tag for left endpoint

# Returns
- Estimated f'(x₁) using formula: (-11f₁ + 18f₂ - 9f₃ + 2f₄) / (6h)
"""
@inline function _estimate_endpoint_derivative(xs::AbstractRange{T}, ys::AbstractVector{T}, ::Val{:left}) where {T<:AbstractFloat}
    @inbounds begin
        f1, f2, f3, f4 = ys[1], ys[2], ys[3], ys[4]
        inv_h = inv(T(step(xs)))
        return _lagrange_d1_left_uniform(f1, f2, f3, f4, inv_h)
    end
end

"""
    _estimate_endpoint_derivative(xs::AbstractRange, ys, ::Val{:right}) -> T

Estimate first derivative at RIGHT endpoint for UNIFORM grid using 4-point Lagrange formula.

# Arguments
- `xs::AbstractRange{T}`: Uniform grid (Range type)
- `ys::AbstractVector{T}`: Function values (must have ≥4 elements)
- `::Val{:right}`: Dispatch tag for right endpoint

# Returns
- Estimated f'(xₙ) using formula: (-2fₙ₋₃ + 9fₙ₋₂ - 18fₙ₋₁ + 11fₙ) / (6h)
"""
@inline function _estimate_endpoint_derivative(xs::AbstractRange{T}, ys::AbstractVector{T}, ::Val{:right}) where {T<:AbstractFloat}
    n = length(ys)
    @inbounds begin
        f1, f2, f3, f4 = ys[n-3], ys[n-2], ys[n-1], ys[n]
        inv_h = inv(T(step(xs)))
        return _lagrange_d1_right_uniform(f1, f2, f3, f4, inv_h)
    end
end

"""
    _estimate_endpoint_derivative(xs::AbstractVector, ys, ::Val{:left}) -> T

Estimate first derivative at LEFT endpoint for NON-UNIFORM grid using full Lagrange formula.

# Mathematical Background
For non-uniform grids, we compute the derivative of the Lagrange interpolating polynomial
P(x) through the first 4 points and evaluate P'(x₁). The formula involves computing
the Lagrange basis function derivatives at x₁.

# Arguments
- `xs::AbstractVector{T}`: Non-uniform grid (Vector type)
- `ys::AbstractVector{T}`: Function values (must have ≥4 elements)
- `::Val{:left}`: Dispatch tag for left endpoint

# Returns
- Estimated f'(x₁) using full Lagrange derivative formula
"""
@inline function _estimate_endpoint_derivative(xs::AbstractVector{T}, ys::AbstractVector{T}, ::Val{:left}) where {T<:AbstractFloat}
    @inbounds begin
        x1, x2, x3, x4 = xs[1], xs[2], xs[3], xs[4]
        f1, f2, f3, f4 = ys[1], ys[2], ys[3], ys[4]

        # Differences between x coordinates
        d12, d13, d14 = x1 - x2, x1 - x3, x1 - x4
        d23, d24, d34 = x2 - x3, x2 - x4, x3 - x4

        # Lagrange basis derivative denominators at x1
        # L'_i(x1) = [product of (x1 - xj) for j≠i, j≠1] / [product of (xi - xj) for j≠i]
        L1_numer = muladd(d12, d13, muladd(d12, d14, d13 * d14))  # d(L1)/dx at x1
        L1_denom = d12 * d13 * d14
        L2_denom = (-d12) * d23 * d24
        L3_denom = d13 * d23 * d34
        L4_denom = -d14 * d24 * d34

        # P'(x1) = sum of fi * L'_i(x1)
        result = f1 * L1_numer / L1_denom
        result = muladd(f2, d13 * d14 / L2_denom, result)
        result = muladd(f3, d12 * d14 / L3_denom, result)
        result = muladd(f4, d12 * d13 / L4_denom, result)
        return result
    end
end

"""
    _estimate_endpoint_derivative(xs::AbstractVector, ys, ::Val{:right}) -> T

Estimate first derivative at RIGHT endpoint for NON-UNIFORM grid using full Lagrange formula.

# Arguments
- `xs::AbstractVector{T}`: Non-uniform grid (Vector type)
- `ys::AbstractVector{T}`: Function values (must have ≥4 elements)
- `::Val{:right}`: Dispatch tag for right endpoint

# Returns
- Estimated f'(xₙ) using full Lagrange derivative formula
"""
@inline function _estimate_endpoint_derivative(xs::AbstractVector{T}, ys::AbstractVector{T}, ::Val{:right}) where {T<:AbstractFloat}
    n = length(xs)
    @inbounds begin
        x1, x2, x3, x4 = xs[n-3], xs[n-2], xs[n-1], xs[n]
        f1, f2, f3, f4 = ys[n-3], ys[n-2], ys[n-1], ys[n]

        # Differences between x coordinates
        d12, d13, d14 = x1 - x2, x1 - x3, x1 - x4
        d23, d24, d34 = x2 - x3, x2 - x4, x3 - x4

        # Lagrange basis derivative denominators at x4 (rightmost point)
        L1_denom = d12 * d13 * d14
        L2_denom = (-d12) * d23 * d24
        L3_denom = d13 * d23 * d34
        L4_numer = muladd(d14, d24, muladd(d14, d34, d24 * d34))  # d(L4)/dx at x4
        L4_denom = -d14 * d24 * d34

        # P'(x4) = sum of fi * L'_i(x4)
        result = f1 * d24 * d34 / L1_denom
        result = muladd(f2, d14 * d34 / L2_denom, result)
        result = muladd(f3, d14 * d24 / L3_denom, result)
        result = muladd(f4, L4_numer / L4_denom, result)
        return result
    end
end
