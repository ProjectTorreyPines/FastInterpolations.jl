# ========================================
# Boundary Condition Kernels
# ========================================
# Pure mathematical kernel functions for BC-related computations.
# These kernels are method-agnostic and can be used by:
# - Cubic splines (cubic_solver.jl)
# - Quadratic splines (quadratic_solver.jl)
# - N-D Hermite interpolation (nd_hermite_eval.jl)
# - Any other interpolation method requiring endpoint derivative estimation
#
# Design Philosophy:
# - Single Source of Truth: All Lagrange derivative kernels live here
# - Two-Stage Pattern: Separate coefficient computation from application
#   Stage 1: Compute coefficients from grid (once per grid)
#   Stage 2: Apply coefficients to data (repeated for batch operations)


# ========================================
# Uniform Grid: Lagrange Endpoint Derivative Kernels
# ========================================
# 4-point one-sided Lagrange polynomial derivative formulas for uniform grids.
# These are the SINGLE SOURCE OF TRUTH for Lagrange stencil coefficients.

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

# Accuracy
- Exact for polynomials of degree ≤ 3 (cubic and below)
- 3rd-order accurate O(h³) for smooth functions

# Operation Count
    0 fdiv + 2 fmul + 3 fmadd = 5 FP ops (excluding inv_h computation)
"""
@inline function _lagrange_d1_left_uniform(f1::T, f2::T, f3::T, f4::T, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-11, 18, -9, 2) / 6
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

Note: Coefficients are negated/reversed from left formula due to symmetry:
    Left:  (-11, 18, -9, 2)
    Right: (-2, 9, -18, 11)

# Accuracy
- Exact for polynomials of degree ≤ 3 (cubic and below)
- 3rd-order accurate O(h³) for smooth functions

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
# Non-Uniform Grid: Precomputed Coefficient Kernels
# ========================================
# Two-stage kernel pattern for batch operations on non-uniform grids:
#   Stage 1: Compute coefficients from x coordinates (once per grid)
#   Stage 2: Apply coefficients to f values (repeated for each y vector)
#
# This mirrors the uniform grid pattern where inv_h is precomputed.

"""
    _lagrange_coeffs_left(x1, x2, x3, x4) -> (c1, c2, c3, c4)

Precompute Lagrange derivative coefficients for LEFT endpoint (x₁).

For a non-uniform grid, the derivative f'(x₁) can be expressed as:
    f'(x₁) = c₁f₁ + c₂f₂ + c₃f₃ + c₄f₄

where c₁, c₂, c₃, c₄ depend only on the x coordinates, not on f values.
This allows precomputation for batch operations.

# Arguments
- `x1, x2, x3, x4::T`: First 4 grid points

# Returns
- Tuple `(c1, c2, c3, c4)` of coefficients

# Operation Count
    4 fdiv + 12 fmul + 6 fadd = 22 FP ops (one-time cost per grid)
"""
@inline function _lagrange_coeffs_left(x1::T, x2::T, x3::T, x4::T) where {T<:AbstractFloat}
    d12, d13, d14 = x1 - x2, x1 - x3, x1 - x4
    d23, d24, d34 = x2 - x3, x2 - x4, x3 - x4

    # L'_1(x1) numerator: derivative of L1 basis at x1
    L1_numer = muladd(d12, d13, muladd(d12, d14, d13 * d14))

    # Compute coefficients c_i = L'_i(x1)
    c1 = L1_numer / (d12 * d13 * d14)
    c2 = (d13 * d14) / ((-d12) * d23 * d24)
    c3 = (d12 * d14) / (d13 * d23 * d34)
    c4 = (d12 * d13) / ((-d14) * d24 * d34)

    return (c1, c2, c3, c4)
end

"""
    _lagrange_coeffs_right(x1, x2, x3, x4) -> (c1, c2, c3, c4)

Precompute Lagrange derivative coefficients for RIGHT endpoint (x₄).

For a non-uniform grid, the derivative f'(x₄) can be expressed as:
    f'(x₄) = c₁f₁ + c₂f₂ + c₃f₃ + c₄f₄

where the x arguments are the LAST 4 grid points: x[n-3], x[n-2], x[n-1], x[n].

# Arguments
- `x1, x2, x3, x4::T`: Last 4 grid points (x[n-3], x[n-2], x[n-1], x[n])

# Returns
- Tuple `(c1, c2, c3, c4)` of coefficients

# Operation Count
    4 fdiv + 12 fmul + 6 fadd = 22 FP ops (one-time cost per grid)
"""
@inline function _lagrange_coeffs_right(x1::T, x2::T, x3::T, x4::T) where {T<:AbstractFloat}
    d12, d13, d14 = x1 - x2, x1 - x3, x1 - x4
    d23, d24, d34 = x2 - x3, x2 - x4, x3 - x4

    # L'_4(x4) numerator: derivative of L4 basis at x4
    L4_numer = muladd(d14, d24, muladd(d14, d34, d24 * d34))

    # Compute coefficients c_i = L'_i(x4)
    c1 = (d24 * d34) / (d12 * d13 * d14)
    c2 = (d14 * d34) / ((-d12) * d23 * d24)
    c3 = (d14 * d24) / (d13 * d23 * d34)
    c4 = L4_numer / ((-d14) * d24 * d34)

    return (c1, c2, c3, c4)
end

"""
    _lagrange_d1_nonuniform(c1, c2, c3, c4, f1, f2, f3, f4) -> T

Apply precomputed Lagrange coefficients to function values.

This is the "Stage 2" kernel for non-uniform grids:
    f'(x) = c₁f₁ + c₂f₂ + c₃f₃ + c₄f₄

Same pattern as uniform grid kernel, but coefficients are grid-dependent
rather than just `inv_h/6 * [-11, 18, -9, 2]`.

# Arguments
- `c1, c2, c3, c4::T`: Precomputed coefficients from `_lagrange_coeffs_left/right`
- `f1, f2, f3, f4::T`: Function values at the 4 grid points

# Returns
- Estimated derivative value

# Operation Count
    0 fdiv + 1 fmul + 3 fmadd = 4 FP ops (per evaluation)

# Comparison with Uniform Grid
    Uniform:     _lagrange_d1_left_uniform(f1,f2,f3,f4, inv_h)  → 5 FP ops
    Non-uniform: _lagrange_d1_nonuniform(c1,c2,c3,c4, f1,f2,f3,f4) → 4 FP ops
"""
@inline function _lagrange_d1_nonuniform(
    c1::T, c2::T, c3::T, c4::T,
    f1::T, f2::T, f3::T, f4::T
) where {T<:AbstractFloat}
    return muladd(c1, f1, muladd(c2, f2, muladd(c3, f3, c4 * f4)))
end


# ========================================
# Unified Endpoint Derivative Estimation API
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
        c1, c2, c3, c4 = _lagrange_coeffs_left(x1, x2, x3, x4)
        return _lagrange_d1_nonuniform(c1, c2, c3, c4, f1, f2, f3, f4)
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
        c1, c2, c3, c4 = _lagrange_coeffs_right(x1, x2, x3, x4)
        return _lagrange_d1_nonuniform(c1, c2, c3, c4, f1, f2, f3, f4)
    end
end
