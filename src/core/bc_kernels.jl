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
# Uniform Grid: Quadratic (3-point) Endpoint Derivative Kernels
# ========================================
# 3-point one-sided Lagrange polynomial derivative formulas for uniform grids.
# Used by ParabolaFit (PolyFit{2}) boundary conditions.

"""
    _quadratic_d1_left_uniform(f1, f2, f3, inv_h) -> T

Compute first derivative at LEFT endpoint using 3-point Lagrange formula.

For uniform grid with spacing h:
    f'(x₁) ≈ (-3f₁ + 4f₂ - f₃) / (2h)

# Arguments
- `f1, f2, f3::T`: Function values at first 3 grid points
- `inv_h::T`: Inverse of uniform grid spacing (1/h)

# Returns
- Estimated derivative f'(x₁)

# Mathematical Derivation
This formula comes from differentiating the unique quadratic polynomial P(x)
passing through (x₁,f₁), (x₂,f₂), (x₃,f₃) and evaluating P'(x₁).

# Accuracy
- Exact for polynomials of degree ≤ 2 (quadratic and below)
- 2nd-order accurate O(h²) for smooth functions

# Operation Count
    0 fdiv + 2 fmul + 2 fmadd = 4 FP ops (excluding inv_h computation)
"""
@inline function _quadratic_d1_left_uniform(f1::T, f2::T, f3::T, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-3, 4, -1) / 2
    coeff = inv_h / 2
    numer = muladd(T(-3), f1, muladd(T(4), f2, -f3))
    return coeff * numer
end

"""
    _quadratic_d1_right_uniform(fnm2, fnm1, fn, inv_h) -> T

Compute first derivative at RIGHT endpoint using 3-point Lagrange formula.

For uniform grid with spacing h:
    f'(xₙ) ≈ (fₙ₋₂ - 4fₙ₋₁ + 3fₙ) / (2h)

# Arguments
- `fnm2, fnm1, fn::T`: Function values at last 3 grid points
  (n-2, n-1, n in 1-based indexing)
- `inv_h::T`: Inverse of uniform grid spacing (1/h)

# Returns
- Estimated derivative f'(xₙ)

# Mathematical Derivation
Mirror of the left formula, obtained by differentiating the Lagrange
polynomial through the last 3 points and evaluating at xₙ.

Note: Coefficients are negated/reversed from left formula due to symmetry:
    Left:  (-3, 4, -1)
    Right: (1, -4, 3)

# Accuracy
- Exact for polynomials of degree ≤ 2 (quadratic and below)
- 2nd-order accurate O(h²) for smooth functions

# Operation Count
    0 fdiv + 2 fmul + 2 fmadd = 4 FP ops (excluding inv_h computation)
"""
@inline function _quadratic_d1_right_uniform(fnm2::T, fnm1::T, fn::T, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (1, -4, 3) / 2
    coeff = inv_h / 2
    numer = muladd(T(1), fnm2, muladd(T(-4), fnm1, T(3) * fn))
    return coeff * numer
end


# ========================================
# Uniform Grid: Linear (2-point) Endpoint Derivative Kernels
# ========================================
# 2-point one-sided finite difference formulas for uniform grids.
# Used by LinearFit (PolyFit{1}) boundary conditions.

"""
    _linear_d1_left_uniform(f1, f2, inv_h) -> T

Compute first derivative at LEFT endpoint using 2-point forward difference.

For uniform grid with spacing h:
    f'(x₁) ≈ (f₂ - f₁) / h

# Arguments
- `f1, f2::T`: Function values at first 2 grid points
- `inv_h::T`: Inverse of uniform grid spacing (1/h)

# Returns
- Estimated derivative f'(x₁)

# Mathematical Derivation
This is the simplest forward difference formula, exact for linear polynomials.

# Accuracy
- Exact for polynomials of degree ≤ 1 (linear and constant)
- 1st-order accurate O(h) for smooth functions

# Operation Count
    0 fdiv + 1 fmul + 1 fsub = 2 FP ops (excluding inv_h computation)
"""
@inline function _linear_d1_left_uniform(f1::T, f2::T, inv_h::T) where {T<:AbstractFloat}
    return (f2 - f1) * inv_h
end

"""
    _linear_d1_right_uniform(fnm1, fn, inv_h) -> T

Compute first derivative at RIGHT endpoint using 2-point backward difference.

For uniform grid with spacing h:
    f'(xₙ) ≈ (fₙ - fₙ₋₁) / h

# Arguments
- `fnm1, fn::T`: Function values at last 2 grid points
  (n-1, n in 1-based indexing)
- `inv_h::T`: Inverse of uniform grid spacing (1/h)

# Returns
- Estimated derivative f'(xₙ)

# Mathematical Derivation
Mirror of the left formula (backward difference).

Note: Both forward and backward differences use the same formula structure
since linear interpolation is symmetric.

# Accuracy
- Exact for polynomials of degree ≤ 1 (linear and constant)
- 1st-order accurate O(h) for smooth functions

# Operation Count
    0 fdiv + 1 fmul + 1 fsub = 2 FP ops (excluding inv_h computation)
"""
@inline function _linear_d1_right_uniform(fnm1::T, fn::T, inv_h::T) where {T<:AbstractFloat}
    return (fn - fnm1) * inv_h
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
# Non-Uniform Grid: 3-point (Quadratic) Coefficient Kernels
# ========================================
# Two-stage kernel pattern for 3-point (ParabolaFit) operations on non-uniform grids.

"""
    _quadratic_coeffs_left(x1, x2, x3) -> (c1, c2, c3)

Precompute 3-point Lagrange derivative coefficients for LEFT endpoint (x₁).

For a non-uniform grid, the derivative f'(x₁) can be expressed as:
    f'(x₁) = c₁f₁ + c₂f₂ + c₃f₃

where c₁, c₂, c₃ depend only on the x coordinates, not on f values.

# Arguments
- `x1, x2, x3::T`: First 3 grid points

# Returns
- Tuple `(c1, c2, c3)` of coefficients

# Mathematical Derivation
For the Lagrange basis L₁(x) = (x-x₂)(x-x₃)/[(x₁-x₂)(x₁-x₃)]:
    L'₁(x₁) = [(x₁-x₂) + (x₁-x₃)] / [(x₁-x₂)(x₁-x₃)]

# Operation Count
    3 fdiv + 4 fmul + 3 fadd = 10 FP ops (one-time cost per grid)
"""
@inline function _quadratic_coeffs_left(x1::T, x2::T, x3::T) where {T<:AbstractFloat}
    d12, d13 = x1 - x2, x1 - x3
    d23 = x2 - x3

    # c1 = L'_1(x1) = (d12 + d13) / (d12 * d13)
    c1 = (d12 + d13) / (d12 * d13)
    # c2 = L'_2(x1) = d13 / [(-d12) * d23]
    c2 = d13 / ((-d12) * d23)
    # c3 = L'_3(x1) = d12 / [d13 * d23]
    c3 = d12 / (d13 * d23)

    return (c1, c2, c3)
end

"""
    _quadratic_coeffs_right(x1, x2, x3) -> (c1, c2, c3)

Precompute 3-point Lagrange derivative coefficients for RIGHT endpoint (x₃).

For a non-uniform grid, the derivative f'(x₃) can be expressed as:
    f'(x₃) = c₁f₁ + c₂f₂ + c₃f₃

where the x arguments are the LAST 3 grid points: x[n-2], x[n-1], x[n].

# Arguments
- `x1, x2, x3::T`: Last 3 grid points (x[n-2], x[n-1], x[n])

# Returns
- Tuple `(c1, c2, c3)` of coefficients

# Mathematical Derivation
For the Lagrange basis L₃(x) = (x-x₁)(x-x₂)/[(x₃-x₁)(x₃-x₂)]:
    L'₃(x₃) = [(x₃-x₁) + (x₃-x₂)] / [(x₃-x₁)(x₃-x₂)]

# Operation Count
    3 fdiv + 4 fmul + 3 fadd = 10 FP ops (one-time cost per grid)
"""
@inline function _quadratic_coeffs_right(x1::T, x2::T, x3::T) where {T<:AbstractFloat}
    d12, d13, d23 = x1 - x2, x1 - x3, x2 - x3

    # At x = x3, the Lagrange basis derivatives are:
    # L'_1(x3) = (x3 - x2) / [(x1-x2)(x1-x3)] = -d23 / [d12 * d13]
    c1 = (-d23) / (d12 * d13)
    # L'_2(x3) = (x3 - x1) / [(x2-x1)(x2-x3)] = -d13 / [(-d12)(d23)] = d13 / [d12 * d23]
    c2 = d13 / (d12 * d23)
    # L'_3(x3) = [(x3-x2)+(x3-x1)] / [(x3-x1)(x3-x2)] = -(d13+d23) / [d13 * d23]
    c3 = (-(d13 + d23)) / (d13 * d23)

    return (c1, c2, c3)
end

"""
    _quadratic_d1_nonuniform(c1, c2, c3, f1, f2, f3) -> T

Apply precomputed 3-point Lagrange coefficients to function values.

This is the "Stage 2" kernel for 3-point non-uniform grids:
    f'(x) = c₁f₁ + c₂f₂ + c₃f₃

# Arguments
- `c1, c2, c3::T`: Precomputed coefficients from `_quadratic_coeffs_left/right`
- `f1, f2, f3::T`: Function values at the 3 grid points

# Returns
- Estimated derivative value

# Operation Count
    0 fdiv + 1 fmul + 2 fmadd = 3 FP ops (per evaluation)
"""
@inline function _quadratic_d1_nonuniform(
    c1::T, c2::T, c3::T,
    f1::T, f2::T, f3::T
) where {T<:AbstractFloat}
    return muladd(c1, f1, muladd(c2, f2, c3 * f3))
end


# ========================================
# Non-Uniform Grid: 2-point (Linear) Coefficient Kernels
# ========================================
# Two-stage kernel pattern for 2-point (LinearFit) operations on non-uniform grids.

"""
    _linear_coeffs_left(x1, x2) -> (c1, c2)

Precompute 2-point finite difference coefficients for LEFT endpoint (x₁).

For a non-uniform grid, the derivative f'(x₁) can be expressed as:
    f'(x₁) = c₁f₁ + c₂f₂

where c₁, c₂ depend only on the x coordinates, not on f values.

# Arguments
- `x1, x2::T`: First 2 grid points

# Returns
- Tuple `(c1, c2)` of coefficients: `(-1/Δx, 1/Δx)` where `Δx = x₂ - x₁`

# Mathematical Derivation
For the 2-point linear interpolant through (x₁,f₁) and (x₂,f₂):
    f'(x) = (f₂ - f₁) / (x₂ - x₁) = -f₁/Δx + f₂/Δx

# Operation Count
    1 fdiv + 1 fneg = 2 FP ops (one-time cost per grid)
"""
@inline function _linear_coeffs_left(x1::T, x2::T) where {T<:AbstractFloat}
    inv_dx = inv(x2 - x1)
    return (-inv_dx, inv_dx)
end

"""
    _linear_coeffs_right(x1, x2) -> (c1, c2)

Precompute 2-point finite difference coefficients for RIGHT endpoint (x₂).

For a non-uniform grid, the derivative f'(x₂) can be expressed as:
    f'(x₂) = c₁f₁ + c₂f₂

where the x arguments are the LAST 2 grid points: x[n-1], x[n].

# Arguments
- `x1, x2::T`: Last 2 grid points (x[n-1], x[n])

# Returns
- Tuple `(c1, c2)` of coefficients: `(-1/Δx, 1/Δx)` where `Δx = x₂ - x₁`

# Note
For 2-point linear interpolation, left and right coefficients are identical
since the derivative of a line is constant everywhere.

# Operation Count
    1 fdiv + 1 fneg = 2 FP ops (one-time cost per grid)
"""
@inline function _linear_coeffs_right(x1::T, x2::T) where {T<:AbstractFloat}
    inv_dx = inv(x2 - x1)
    return (-inv_dx, inv_dx)
end

"""
    _linear_d1_nonuniform(c1, c2, f1, f2) -> T

Apply precomputed 2-point finite difference coefficients to function values.

This is the "Stage 2" kernel for 2-point non-uniform grids:
    f'(x) = c₁f₁ + c₂f₂

# Arguments
- `c1, c2::T`: Precomputed coefficients from `_linear_coeffs_left/right`
- `f1, f2::T`: Function values at the 2 grid points

# Returns
- Estimated derivative value

# Operation Count
    0 fdiv + 1 fmul + 1 fmadd = 2 FP ops (per evaluation)
"""
@inline function _linear_d1_nonuniform(
    c1::T, c2::T,
    f1::T, f2::T
) where {T<:AbstractFloat}
    return muladd(c1, f1, c2 * f2)
end


# ========================================
# Unified Endpoint Derivative Estimation API
# ========================================
# Higher-level API using Val{:left}/Val{:right} and PolyFit{D} dispatch.
# Handles both uniform (Range) and non-uniform (Vector) grids.
#
# API signature: _estimate_endpoint_derivative(xs, ys, Val(:left/:right), PolyFit{D}())
#
# Supported polynomial degrees:
#   - PolyFit{1} (LinearFit):   2-point, O(h)  accuracy
#   - PolyFit{2} (ParabolaFit): 3-point, O(h²) accuracy
#   - PolyFit{3} (CubicFit):    4-point, O(h³) accuracy


# ----------------------------------------
# CubicFit (PolyFit{3}) - 4-point formulas
# ----------------------------------------

"""
    _estimate_endpoint_derivative(xs, ys, ::Val{:left}, ::PolyFit{3}) -> T

Estimate first derivative at LEFT endpoint using 4-point CubicFit formula.

# Arguments
- `xs`: Grid coordinates (AbstractRange for uniform, AbstractVector for non-uniform)
- `ys::AbstractVector{T}`: Function values (must have ≥4 elements)
- `::Val{:left}`: Dispatch tag for left endpoint
- `::PolyFit{3}`: CubicFit boundary condition (4-point, O(h³))

# Returns
- Estimated f'(x₁) using formula: (-11f₁ + 18f₂ - 9f₃ + 2f₄) / (6h) for uniform grids
"""
@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, ::Val{:left}, ::PolyFit{3}
) where {T<:AbstractFloat}
    @inbounds begin
        f1, f2, f3, f4 = ys[1], ys[2], ys[3], ys[4]
        inv_h = inv(T(step(xs)))
        return _lagrange_d1_left_uniform(f1, f2, f3, f4, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, ::Val{:right}, ::PolyFit{3}
) where {T<:AbstractFloat}
    n = length(ys)
    @inbounds begin
        f1, f2, f3, f4 = ys[n-3], ys[n-2], ys[n-1], ys[n]
        inv_h = inv(T(step(xs)))
        return _lagrange_d1_right_uniform(f1, f2, f3, f4, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, ::Val{:left}, ::PolyFit{3}
) where {T<:AbstractFloat}
    @inbounds begin
        x1, x2, x3, x4 = xs[1], xs[2], xs[3], xs[4]
        f1, f2, f3, f4 = ys[1], ys[2], ys[3], ys[4]
        c1, c2, c3, c4 = _lagrange_coeffs_left(x1, x2, x3, x4)
        return _lagrange_d1_nonuniform(c1, c2, c3, c4, f1, f2, f3, f4)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, ::Val{:right}, ::PolyFit{3}
) where {T<:AbstractFloat}
    n = length(xs)
    @inbounds begin
        x1, x2, x3, x4 = xs[n-3], xs[n-2], xs[n-1], xs[n]
        f1, f2, f3, f4 = ys[n-3], ys[n-2], ys[n-1], ys[n]
        c1, c2, c3, c4 = _lagrange_coeffs_right(x1, x2, x3, x4)
        return _lagrange_d1_nonuniform(c1, c2, c3, c4, f1, f2, f3, f4)
    end
end


# ----------------------------------------
# ParabolaFit (PolyFit{2}) - 3-point formulas
# ----------------------------------------

"""
    _estimate_endpoint_derivative(xs, ys, ::Val{:left}, ::PolyFit{2}) -> T

Estimate first derivative at LEFT endpoint using 3-point ParabolaFit formula.

# Arguments
- `xs`: Grid coordinates (AbstractRange for uniform, AbstractVector for non-uniform)
- `ys::AbstractVector{T}`: Function values (must have ≥3 elements)
- `::Val{:left}`: Dispatch tag for left endpoint
- `::PolyFit{2}`: ParabolaFit boundary condition (3-point, O(h²))

# Returns
- Estimated f'(x₁) using formula: (-3f₁ + 4f₂ - f₃) / (2h) for uniform grids
"""
@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, ::Val{:left}, ::PolyFit{2}
) where {T<:AbstractFloat}
    @inbounds begin
        f1, f2, f3 = ys[1], ys[2], ys[3]
        inv_h = inv(T(step(xs)))
        return _quadratic_d1_left_uniform(f1, f2, f3, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, ::Val{:right}, ::PolyFit{2}
) where {T<:AbstractFloat}
    n = length(ys)
    @inbounds begin
        f1, f2, f3 = ys[n-2], ys[n-1], ys[n]
        inv_h = inv(T(step(xs)))
        return _quadratic_d1_right_uniform(f1, f2, f3, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, ::Val{:left}, ::PolyFit{2}
) where {T<:AbstractFloat}
    @inbounds begin
        x1, x2, x3 = xs[1], xs[2], xs[3]
        f1, f2, f3 = ys[1], ys[2], ys[3]
        c1, c2, c3 = _quadratic_coeffs_left(x1, x2, x3)
        return _quadratic_d1_nonuniform(c1, c2, c3, f1, f2, f3)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, ::Val{:right}, ::PolyFit{2}
) where {T<:AbstractFloat}
    n = length(xs)
    @inbounds begin
        x1, x2, x3 = xs[n-2], xs[n-1], xs[n]
        f1, f2, f3 = ys[n-2], ys[n-1], ys[n]
        c1, c2, c3 = _quadratic_coeffs_right(x1, x2, x3)
        return _quadratic_d1_nonuniform(c1, c2, c3, f1, f2, f3)
    end
end


# ----------------------------------------
# LinearFit (PolyFit{1}) - 2-point formulas
# ----------------------------------------

"""
    _estimate_endpoint_derivative(xs, ys, ::Val{:left}, ::PolyFit{1}) -> T

Estimate first derivative at LEFT endpoint using 2-point LinearFit (forward difference).

# Arguments
- `xs`: Grid coordinates (AbstractRange for uniform, AbstractVector for non-uniform)
- `ys::AbstractVector{T}`: Function values (must have ≥2 elements)
- `::Val{:left}`: Dispatch tag for left endpoint
- `::PolyFit{1}`: LinearFit boundary condition (2-point, O(h))

# Returns
- Estimated f'(x₁) using formula: (f₂ - f₁) / h for uniform grids
"""
@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, ::Val{:left}, ::PolyFit{1}
) where {T<:AbstractFloat}
    @inbounds begin
        f1, f2 = ys[1], ys[2]
        inv_h = inv(T(step(xs)))
        return _linear_d1_left_uniform(f1, f2, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, ::Val{:right}, ::PolyFit{1}
) where {T<:AbstractFloat}
    n = length(ys)
    @inbounds begin
        f1, f2 = ys[n-1], ys[n]
        inv_h = inv(T(step(xs)))
        return _linear_d1_right_uniform(f1, f2, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, ::Val{:left}, ::PolyFit{1}
) where {T<:AbstractFloat}
    @inbounds begin
        x1, x2 = xs[1], xs[2]
        f1, f2 = ys[1], ys[2]
        c1, c2 = _linear_coeffs_left(x1, x2)
        return _linear_d1_nonuniform(c1, c2, f1, f2)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, ::Val{:right}, ::PolyFit{1}
) where {T<:AbstractFloat}
    n = length(xs)
    @inbounds begin
        x1, x2 = xs[n-1], xs[n]
        f1, f2 = ys[n-1], ys[n]
        c1, c2 = _linear_coeffs_right(x1, x2)
        return _linear_d1_nonuniform(c1, c2, f1, f2)
    end
end
