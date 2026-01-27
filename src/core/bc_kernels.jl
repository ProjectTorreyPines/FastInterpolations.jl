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
# - Type-Dispatched API: PolyFit{D} for degree, Val{:left/:right} for endpoint
# - Two-Stage Pattern: Separate coefficient computation from application
#   Stage 1: Compute coefficients from grid (once per grid)
#   Stage 2: Apply coefficients to data (repeated for batch operations)


# ========================================
# Type-Dispatched Kernel API
# ========================================
# Unified kernel interface using PolyFit{D} and Val{:left/:right} dispatch.
# These are the primary internal kernels - named for what they DO, not WHERE.
#
# API:
#   _compute_deriv1(PolyFit{D}, Val{Side}, f::NTuple, inv_h)  -- uniform grid
#   _compute_deriv1_coeffs(PolyFit{D}, Val{Side}, x::NTuple)  -- non-uniform coeffs
#   _weighted_sum(c::NTuple, f::NTuple)                        -- apply coefficients


# ----------------------------------------
# _weighted_sum: Generic coefficient application
# ----------------------------------------
# Computes Σ cᵢfᵢ using muladd chains for accuracy.

"""
    _weighted_sum(c::NTuple{N,T}, f::NTuple{N,T}) -> T

Compute weighted sum Σ cᵢfᵢ using muladd for numerical stability.

This is the "Stage 2" kernel for non-uniform grids: applies precomputed
coefficients to function values.

# Operation Count
    N-1 fmadd + 1 fmul = N FP ops
"""
@inline function _weighted_sum(c::NTuple{2,T}, f::NTuple{2,T}) where {T<:AbstractFloat}
    muladd(c[1], f[1], c[2] * f[2])
end

@inline function _weighted_sum(c::NTuple{3,T}, f::NTuple{3,T}) where {T<:AbstractFloat}
    muladd(c[1], f[1], muladd(c[2], f[2], c[3] * f[3]))
end

@inline function _weighted_sum(c::NTuple{4,T}, f::NTuple{4,T}) where {T<:AbstractFloat}
    muladd(c[1], f[1], muladd(c[2], f[2], muladd(c[3], f[3], c[4] * f[4])))
end


# ----------------------------------------
# _compute_deriv1: Uniform grid direct computation
# ----------------------------------------
# Computes first derivative directly using known stencil coefficients.
# Dispatches on PolyFit{D} (degree) and Val{:left/:right} (endpoint).

"""
    _compute_deriv1(::PolyFit{D}, ::Val{Side}, f::NTuple, inv_h) -> T

Compute first derivative on uniform grid using D+1 point stencil.

# Arguments
- `::PolyFit{D}`: Polynomial degree (D=1,2,3 supported)
- `::Val{:left}` or `::Val{:right}`: Which endpoint
- `f::NTuple{D+1,T}`: Function values at stencil points
- `inv_h::T`: Inverse grid spacing (1/h)

# Supported Degrees
- PolyFit{1} (LinearFit): 2 points, O(h) accuracy
- PolyFit{2} (ParabolaFit): 3 points, O(h²) accuracy
- PolyFit{3} (CubicFit): 4 points, O(h³) accuracy
"""
# PolyFit{1} (LinearFit) - 2 points, O(h)
@inline function _compute_deriv1(::PolyFit{1}, ::Val{:left}, f::NTuple{2,T}, inv_h::T) where {T<:AbstractFloat}
    (f[2] - f[1]) * inv_h
end

@inline function _compute_deriv1(::PolyFit{1}, ::Val{:right}, f::NTuple{2,T}, inv_h::T) where {T<:AbstractFloat}
    (f[2] - f[1]) * inv_h  # Same as left for linear
end

# PolyFit{2} (ParabolaFit) - 3 points, O(h²)
@inline function _compute_deriv1(::PolyFit{2}, ::Val{:left}, f::NTuple{3,T}, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-3, 4, -1) / 2
    coeff = inv_h / 2
    muladd(T(-3), f[1], muladd(T(4), f[2], -f[3])) * coeff
end

@inline function _compute_deriv1(::PolyFit{2}, ::Val{:right}, f::NTuple{3,T}, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (1, -4, 3) / 2
    coeff = inv_h / 2
    muladd(T(1), f[1], muladd(T(-4), f[2], T(3) * f[3])) * coeff
end

# PolyFit{3} (CubicFit) - 4 points, O(h³)
@inline function _compute_deriv1(::PolyFit{3}, ::Val{:left}, f::NTuple{4,T}, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-11, 18, -9, 2) / 6
    coeff = inv_h / 6
    muladd(T(-11), f[1], muladd(T(18), f[2], muladd(T(-9), f[3], T(2) * f[4]))) * coeff
end

@inline function _compute_deriv1(::PolyFit{3}, ::Val{:right}, f::NTuple{4,T}, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-2, 9, -18, 11) / 6
    coeff = inv_h / 6
    muladd(T(-2), f[1], muladd(T(9), f[2], muladd(T(-18), f[3], T(11) * f[4]))) * coeff
end


# ----------------------------------------
# _compute_deriv1_coeffs: Non-uniform grid coefficients
# ----------------------------------------
# Precomputes coefficients from x coordinates for non-uniform grids.
# These can be cached and reused for batch operations.

"""
    _compute_deriv1_coeffs(::PolyFit{D}, ::Val{Side}, x::NTuple) -> NTuple

Precompute derivative coefficients for non-uniform grid.

Returns coefficients (c₁, ..., cₙ) such that f'(endpoint) ≈ Σ cᵢfᵢ.
Use with `_weighted_sum(coeffs, f_values)` to compute the derivative.

# Arguments
- `::PolyFit{D}`: Polynomial degree
- `::Val{:left}` or `::Val{:right}`: Which endpoint
- `x::NTuple{D+1,T}`: Grid coordinates at stencil points
"""
# PolyFit{1} (LinearFit) - 2 points
@inline function _compute_deriv1_coeffs(::PolyFit{1}, ::Val{:left}, x::NTuple{2,T}) where {T<:AbstractFloat}
    inv_dx = inv(x[2] - x[1])
    (-inv_dx, inv_dx)
end

@inline function _compute_deriv1_coeffs(::PolyFit{1}, ::Val{:right}, x::NTuple{2,T}) where {T<:AbstractFloat}
    inv_dx = inv(x[2] - x[1])
    (-inv_dx, inv_dx)  # Same as left for linear
end

# PolyFit{2} (ParabolaFit) - 3 points
@inline function _compute_deriv1_coeffs(::PolyFit{2}, ::Val{:left}, x::NTuple{3,T}) where {T<:AbstractFloat}
    x1, x2, x3 = x
    d12, d13 = x1 - x2, x1 - x3
    d23 = x2 - x3
    c1 = (d12 + d13) / (d12 * d13)
    c2 = d13 / ((-d12) * d23)
    c3 = d12 / (d13 * d23)
    (c1, c2, c3)
end

@inline function _compute_deriv1_coeffs(::PolyFit{2}, ::Val{:right}, x::NTuple{3,T}) where {T<:AbstractFloat}
    x1, x2, x3 = x
    d12, d13, d23 = x1 - x2, x1 - x3, x2 - x3
    c1 = (-d23) / (d12 * d13)
    c2 = d13 / (d12 * d23)
    c3 = (-(d13 + d23)) / (d13 * d23)
    (c1, c2, c3)
end

# PolyFit{3} (CubicFit) - 4 points
@inline function _compute_deriv1_coeffs(::PolyFit{3}, ::Val{:left}, x::NTuple{4,T}) where {T<:AbstractFloat}
    x1, x2, x3, x4 = x
    d12, d13, d14 = x1 - x2, x1 - x3, x1 - x4
    d23, d24, d34 = x2 - x3, x2 - x4, x3 - x4
    L1_numer = muladd(d12, d13, muladd(d12, d14, d13 * d14))
    c1 = L1_numer / (d12 * d13 * d14)
    c2 = (d13 * d14) / ((-d12) * d23 * d24)
    c3 = (d12 * d14) / (d13 * d23 * d34)
    c4 = (d12 * d13) / ((-d14) * d24 * d34)
    (c1, c2, c3, c4)
end

@inline function _compute_deriv1_coeffs(::PolyFit{3}, ::Val{:right}, x::NTuple{4,T}) where {T<:AbstractFloat}
    x1, x2, x3, x4 = x
    d12, d13, d14 = x1 - x2, x1 - x3, x1 - x4
    d23, d24, d34 = x2 - x3, x2 - x4, x3 - x4
    L4_numer = muladd(d14, d24, muladd(d14, d34, d24 * d34))
    c1 = (d24 * d34) / (d12 * d13 * d14)
    c2 = (d14 * d34) / ((-d12) * d23 * d24)
    c3 = (d14 * d24) / (d13 * d23 * d34)
    c4 = L4_numer / ((-d14) * d24 * d34)
    (c1, c2, c3, c4)
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
# Helper: Extract stencil values as NTuple
# ----------------------------------------

"""
    _extract_stencil_values(v::AbstractVector, ::Val{:left}, ::Val{N}) -> NTuple{N}
    _extract_stencil_values(v::AbstractVector, ::Val{:right}, ::Val{N}) -> NTuple{N}

Extract N values from left or right endpoint as NTuple for kernel dispatch.
"""
@inline function _extract_stencil_values(v::AbstractVector{T}, ::Val{:left}, ::Val{N}) where {T, N}
    @inbounds ntuple(i -> v[i], Val(N))
end

@inline function _extract_stencil_values(v::AbstractVector{T}, ::Val{:right}, ::Val{N}) where {T, N}
    n = length(v)
    @inbounds ntuple(i -> v[n - N + i], Val(N))
end


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
    xs::AbstractRange{T}, ys::AbstractVector{T}, side::Val{S}, ::PolyFit{3}
) where {T<:AbstractFloat, S}
    @inbounds begin
        f = _extract_stencil_values(ys, side, Val(4))
        inv_h = inv(T(step(xs)))
        return _compute_deriv1(PolyFit{3}(), side, f, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, side::Val{S}, ::PolyFit{3}
) where {T<:AbstractFloat, S}
    @inbounds begin
        x = _extract_stencil_values(xs, side, Val(4))
        f = _extract_stencil_values(ys, side, Val(4))
        c = _compute_deriv1_coeffs(PolyFit{3}(), side, x)
        return _weighted_sum(c, f)
    end
end


# ----------------------------------------
# ParabolaFit (PolyFit{2}) - 3-point formulas
# ----------------------------------------

"""
    _estimate_endpoint_derivative(xs, ys, side, ::PolyFit{2}) -> T

Estimate first derivative at endpoint using 3-point ParabolaFit formula.

# Arguments
- `xs`: Grid coordinates (AbstractRange for uniform, AbstractVector for non-uniform)
- `ys::AbstractVector{T}`: Function values (must have ≥3 elements)
- `side`: `Val(:left)` or `Val(:right)` for endpoint selection
- `::PolyFit{2}`: ParabolaFit boundary condition (3-point, O(h²))

# Returns
- Estimated derivative using formula: (-3f₁ + 4f₂ - f₃) / (2h) for uniform left
"""
@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, side::Val{S}, ::PolyFit{2}
) where {T<:AbstractFloat, S}
    @inbounds begin
        f = _extract_stencil_values(ys, side, Val(3))
        inv_h = inv(T(step(xs)))
        return _compute_deriv1(PolyFit{2}(), side, f, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, side::Val{S}, ::PolyFit{2}
) where {T<:AbstractFloat, S}
    @inbounds begin
        x = _extract_stencil_values(xs, side, Val(3))
        f = _extract_stencil_values(ys, side, Val(3))
        c = _compute_deriv1_coeffs(PolyFit{2}(), side, x)
        return _weighted_sum(c, f)
    end
end


# ----------------------------------------
# LinearFit (PolyFit{1}) - 2-point formulas
# ----------------------------------------

"""
    _estimate_endpoint_derivative(xs, ys, side, ::PolyFit{1}) -> T

Estimate first derivative at endpoint using 2-point LinearFit (finite difference).

# Arguments
- `xs`: Grid coordinates (AbstractRange for uniform, AbstractVector for non-uniform)
- `ys::AbstractVector{T}`: Function values (must have ≥2 elements)
- `side`: `Val(:left)` or `Val(:right)` for endpoint selection
- `::PolyFit{1}`: LinearFit boundary condition (2-point, O(h))

# Returns
- Estimated derivative using formula: (f₂ - f₁) / h for uniform grids
"""
@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, side::Val{S}, ::PolyFit{1}
) where {T<:AbstractFloat, S}
    @inbounds begin
        f = _extract_stencil_values(ys, side, Val(2))
        inv_h = inv(T(step(xs)))
        return _compute_deriv1(PolyFit{1}(), side, f, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, side::Val{S}, ::PolyFit{1}
) where {T<:AbstractFloat, S}
    @inbounds begin
        x = _extract_stencil_values(xs, side, Val(2))
        f = _extract_stencil_values(ys, side, Val(2))
        c = _compute_deriv1_coeffs(PolyFit{1}(), side, x)
        return _weighted_sum(c, f)
    end
end
