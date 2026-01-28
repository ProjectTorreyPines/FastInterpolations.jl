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
    return muladd(c[1], f[1], c[2] * f[2])
end

@inline function _weighted_sum(c::NTuple{3,T}, f::NTuple{3,T}) where {T<:AbstractFloat}
    return muladd(c[1], f[1], muladd(c[2], f[2], c[3] * f[3]))
end

@inline function _weighted_sum(c::NTuple{4,T}, f::NTuple{4,T}) where {T<:AbstractFloat}
    return muladd(c[1], f[1], muladd(c[2], f[2], muladd(c[3], f[3], c[4] * f[4])))
end

# Generic fallback for N > 4 (PolyFit{D} where D > 3)
# Uses @generated to produce unrolled muladd chain at compile time
@generated function _weighted_sum(c::NTuple{N,T}, f::NTuple{N,T}) where {N,T<:AbstractFloat}
    # Build muladd chain: muladd(c[1], f[1], muladd(c[2], f[2], ...c[N]*f[N]...))
    expr = :(c[$N] * f[$N])
    for i in (N-1):-1:1
        expr = :(muladd(c[$i], f[$i], $expr))
    end
    return expr
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
    return (f[2] - f[1]) * inv_h
end

@inline function _compute_deriv1(::PolyFit{1}, ::Val{:right}, f::NTuple{2,T}, inv_h::T) where {T<:AbstractFloat}
    return (f[2] - f[1]) * inv_h  # Same as left for linear
end

# PolyFit{2} (ParabolaFit) - 3 points, O(h²)
@inline function _compute_deriv1(::PolyFit{2}, ::Val{:left}, f::NTuple{3,T}, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-3, 4, -1) / 2
    coeff = inv_h / 2
    return muladd(T(-3), f[1], muladd(T(4), f[2], -f[3])) * coeff
end

@inline function _compute_deriv1(::PolyFit{2}, ::Val{:right}, f::NTuple{3,T}, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (1, -4, 3) / 2
    coeff = inv_h / 2
    return muladd(T(1), f[1], muladd(T(-4), f[2], T(3) * f[3])) * coeff
end

# PolyFit{3} (CubicFit) - 4 points, O(h³)
@inline function _compute_deriv1(::PolyFit{3}, ::Val{:left}, f::NTuple{4,T}, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-11, 18, -9, 2) / 6
    coeff = inv_h / 6
    return muladd(T(-11), f[1], muladd(T(18), f[2], muladd(T(-9), f[3], T(2) * f[4]))) * coeff
end

@inline function _compute_deriv1(::PolyFit{3}, ::Val{:right}, f::NTuple{4,T}, inv_h::T) where {T<:AbstractFloat}
    # Coefficients: (-2, 9, -18, 11) / 6
    coeff = inv_h / 6
    return muladd(T(-2), f[1], muladd(T(9), f[2], muladd(T(-18), f[3], T(11) * f[4]))) * coeff
end

# Generic fallback for D > 3 (uses barycentric differentiation)
# Julia dispatch ensures D=1,2,3 use the specialized methods above.
@inline @with_pool pool function _compute_deriv1(
    pf::PolyFit{D}, side::Val{S}, f::NTuple{N,T}, inv_h::T
) where {D, S, N, T<:AbstractFloat}
    # Compute coefficients on reference grid t = 0, 1, ..., D
    coeffs = acquire!(pool, T, N)
    β = acquire!(pool, T, N)
    # Reference grid as NTuple (allocation-free)
    t = ntuple(i -> T(i - 1), Val(N))
    _compute_deriv1_coeffs!(coeffs, β, pf, side, t)
    # Scale by inv_h and compute weighted sum
    s = zero(T)
    @inbounds for i in 1:N
        s = muladd(coeffs[i] * inv_h, f[i], s)
    end
    return s
end


# ----------------------------------------
# _compute_deriv1_coeffs: Non-uniform grid coefficients
# ----------------------------------------
# Precomputes coefficients from x coordinates for non-uniform grids.
# These can be cached and reused for batch operations.

"""
    _compute_deriv1_coeffs(::PolyFit{D}, ::Val{Side}, x::NTuple) -> NTuple{D+1,T}

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
    return (-inv_dx, inv_dx)
end

@inline function _compute_deriv1_coeffs(::PolyFit{1}, ::Val{:right}, x::NTuple{2,T}) where {T<:AbstractFloat}
    inv_dx = inv(x[2] - x[1])
    return (-inv_dx, inv_dx)  # Same as left for linear
end

# PolyFit{2} (ParabolaFit) - 3 points
@inline function _compute_deriv1_coeffs(::PolyFit{2}, ::Val{:left}, x::NTuple{3,T}) where {T<:AbstractFloat}
    x1, x2, x3 = x
    d12, d13 = x1 - x2, x1 - x3
    d23 = x2 - x3
    c1 = (d12 + d13) / (d12 * d13)
    c2 = d13 / ((-d12) * d23)
    c3 = d12 / (d13 * d23)
    return (c1, c2, c3)
end

@inline function _compute_deriv1_coeffs(::PolyFit{2}, ::Val{:right}, x::NTuple{3,T}) where {T<:AbstractFloat}
    x1, x2, x3 = x
    d12, d13, d23 = x1 - x2, x1 - x3, x2 - x3
    c1 = (-d23) / (d12 * d13)
    c2 = d13 / (d12 * d23)
    c3 = (-(d13 + d23)) / (d13 * d23)
    return (c1, c2, c3)
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
    return (c1, c2, c3, c4)
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
    return (c1, c2, c3, c4)
end

# Generic fallback for D > 3 (uses barycentric differentiation)
# Returns NTuple for compatibility with _weighted_sum.
# Julia dispatch ensures D=1,2,3 use the specialized methods above.
@inline @with_pool pool function _compute_deriv1_coeffs(
    pf::PolyFit{D}, side::Val{S}, x::NTuple{N,T}
) where {D, S, N, T<:AbstractFloat}
    c = acquire!(pool, T, N)
    β = acquire!(pool, T, N)
    _compute_deriv1_coeffs!(c, β, pf, side, x)
    return ntuple(i -> @inbounds(c[i]), Val(N))
end


# ----------------------------------------
# Generic Barycentric Differentiation (D > 3)
# ----------------------------------------
# For polynomial degrees D > 3, we use the barycentric differentiation
# formula which works for any number of points on any grid spacing.
#
# Math: Given N points, the derivative at node x_k is:
#   f'(x_k) = Σ_i c_i * f_i
# where:
#   β_i = 1 / Π_{j≠i} (x_i - x_j)          (barycentric weights)
#   c_i = β_i / (β_k * (x_k - x_i))        for i ≠ k
#   c_k = -Σ_{i≠k} c_i                     (diagonal element)

"""
    _barycentric_weights!(β::AbstractVector{T}, x::NTuple{N,T}, ::Val{N}) -> β

In-place computation of barycentric weights β_i = 1 / Π_{j≠i} (x_i - x_j).

# Arguments
- `β::AbstractVector{T}`: Output buffer of length N (mutated)
- `x::NTuple{N,T}`: Grid coordinates
- `::Val{N}`: Number of points to use

# Returns
- `β`: The same buffer, now containing the barycentric weights
"""
@inline function _barycentric_weights!(
    β::AbstractVector{T}, x::NTuple{N,T}, ::Val{N}
) where {N,T<:AbstractFloat}
    @assert length(β) >= N "Buffer β must have length ≥ $N"
    @inbounds for i in 1:N
        xi = x[i]
        wi = one(T)
        for j in 1:N
            j == i && continue
            wi *= inv(xi - x[j])
        end
        β[i] = wi
    end
    return β
end

"""
    _d1_coeffs_at_node!(c, β, x::NTuple{N,T}, k::Int, ::Val{N}) -> c

In-place computation of first-derivative coefficients at node x_k.

Uses barycentric formula:
- c_i = β_i / (β_k * (x_k - x_i)) for i ≠ k
- c_k = -Σ_{i≠k} c_i

# Arguments
- `coeffs::AbstractVector{T}`: Output buffer of length N for coefficients (mutated)
- `β::AbstractVector{T}`: Workspace buffer of length N for barycentric weights (mutated)
- `x::NTuple{N,T}`: Grid coordinates
- `k::Int`: Node index (1 for left endpoint, N for right endpoint)
- `::Val{N}`: Number of points to use

# Returns
- `c`: The same buffer, now containing the derivative coefficients
"""
@inline function _d1_coeffs_at_node!(
    coeffs::AbstractVector{T}, β::AbstractVector{T}, x::NTuple{N,T}, k::Int, ::Val{N}
) where {N,T<:AbstractFloat}
    @assert length(coeffs) >= N "Buffer coeffs must have length ≥ $N"
    @assert length(β) >= N "Buffer β must have length ≥ $N"
    @assert 1 <= k <= N "Node index k=$k must be in 1:$N"

    # Compute barycentric weights in-place
    _barycentric_weights!(β, x, Val(N))

    @inbounds xk = x[k]
    @inbounds βk = β[k]

    # Compute coefficients and accumulate sum for diagonal
    s = zero(T)
    @inbounds for i in 1:N
        if i == k
            coeffs[i] = zero(T)  # placeholder, will be overwritten
        else
            coeffs[i] = β[i] / (βk * (xk - x[i]))
            s += coeffs[i]
        end
    end

    # Diagonal element
    @inbounds coeffs[k] = -s

    return coeffs
end


# ----------------------------------------
# Generic PolyFit{D} Coefficient Computation (D > 3)
# ----------------------------------------
# Julia dispatch prefers the more specific D=1,2,3 methods above.

"""
    _compute_deriv1_coeffs!(coeffs, β, ::PolyFit{D}, side::Val{S}, x::NTuple{N,T}) -> coeffs          

In-place computation of derivative coefficients for generic polynomial degree D.

This is the generic fallback for D > 3. For D = 1, 2, 3, the specialized
`_compute_deriv1_coeffs` methods (which return NTuple) are used instead.

# Arguments
- `coeffs::AbstractVector{T}`: Output buffer for coefficients (length ≥ D+1)
- `β::AbstractVector{T}`: Workspace buffer for barycentric weights (length ≥ D+1)
- `::PolyFit{D}`: Polynomial degree
- `side::Val{:left}` or `Val{:right}`: Which endpoint
- `x::NTuple{N,T}`: Grid coordinates (length ≥ D+1)

# Returns
- `coeffs::AbstractVector{T}`: The coefficient buffer, now containing the derivative coefficients
"""
@inline function _compute_deriv1_coeffs!(
    coeffs::AbstractVector{T}, β::AbstractVector{T},
    ::PolyFit{D}, side::Val{S}, x::NTuple{N,T}
) where {D,S,N,T<:AbstractFloat}
    @assert S === :left || S === :right "Invalid side: Val(:$S). Must be Val(:left) or Val(:right)."
    k = (S === :left) ? 1 : N
    _d1_coeffs_at_node!(coeffs, β, x, k, Val(N))
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
# Helper: Extract stencil values
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
# Unified Endpoint Derivative Estimation
# ----------------------------------------
# Simple unified API for all polynomial degrees.
# Kernel functions (_compute_deriv1, _compute_deriv1_coeffs) handle
# D-specific dispatch internally via Julia's type dispatch system.

"""
Maximum recommended polynomial degree for endpoint derivative estimation.
Higher degrees (D > 6) may amplify numerical noise in the data.
"""
const MAX_RECOMMENDED_POLYFIT_DEGREE = 6

# Thread-local warning tracker (avoids global state issues)
const _polyfit_warning_issued = Ref(false)

"""
    _warn_high_degree(D::Int)

Issue a one-time warning if polynomial degree exceeds recommendation.
Called by `_check_polyfit_requirements` when D > MAX_RECOMMENDED_POLYFIT_DEGREE.
"""
@noinline function _warn_high_degree(D::Int)
    if !_polyfit_warning_issued[]
        _polyfit_warning_issued[] = true
        @warn "PolyFit{$D} uses $(D+1) points. For D > $MAX_RECOMMENDED_POLYFIT_DEGREE, " *
              "numerical noise amplification may degrade accuracy. Consider D ≤ 6."
    end
    nothing
end

"""
    _check_polyfit_requirements(D::Int, n::Int)

Validate PolyFit{D} requirements:
1. Bounds check: `n ≥ D + 1` (can be elided with `@inbounds`)
2. Degree warning: one-time warning if D > MAX_RECOMMENDED_POLYFIT_DEGREE

# Arguments
- `D::Int`: Polynomial degree
- `n::Int`: Number of available data points (length of xs or ys)

# Throws
- `ArgumentError` if `n < D + 1` (unless bounds checking is disabled)
"""
@inline function _check_polyfit_requirements(D::Int, n::Int)
    @boundscheck if n < D + 1
        throw(ArgumentError(
            "PolyFit{$D} requires at least $(D+1) points, got $n"
        ))
    end
    D > MAX_RECOMMENDED_POLYFIT_DEGREE && _warn_high_degree(D)
    nothing
end


"""
    _estimate_endpoint_derivative(xs, ys, side, ::PolyFit{D}) -> T

Estimate first derivative at endpoint using D+1 point polynomial fit.

# Arguments
- `xs`: Grid coordinates (AbstractRange for uniform, AbstractVector for non-uniform)
- `ys::AbstractVector{T}`: Function values (must have ≥ D+1 elements)
- `side`: `Val(:left)` or `Val(:right)` for endpoint selection
- `::PolyFit{D}`: Polynomial degree

# Supported Degrees
- PolyFit{1} (LinearFit):   2 points, O(h) accuracy
- PolyFit{2} (ParabolaFit): 3 points, O(h²) accuracy
- PolyFit{3} (CubicFit):    4 points, O(h³) accuracy
- PolyFit{D} (D > 3):       D+1 points, O(h^D) accuracy (barycentric method)

# Implementation
- D ≤ 3: Dispatches to specialized NTuple-based kernels (allocation-free)
- D > 3: Dispatches to generic barycentric kernels (Vector-based)
"""
@inline function _estimate_endpoint_derivative(
    xs::AbstractRange{T}, ys::AbstractVector{T}, side::Val{S}, pf::PolyFit{D}
) where {T<:AbstractFloat, S, D}
    _check_polyfit_requirements(D, length(ys))
    @inbounds begin
        f = _extract_stencil_values(ys, side, Val(D + 1))
        inv_h = inv(T(step(xs)))
        return _compute_deriv1(pf, side, f, inv_h)
    end
end

@inline function _estimate_endpoint_derivative(
    xs::AbstractVector{T}, ys::AbstractVector{T}, side::Val{S}, pf::PolyFit{D}
) where {T<:AbstractFloat, S, D}
    @assert length(xs) == length(ys) "xs and ys must have same length"
    _check_polyfit_requirements(D, length(ys))
    @inbounds begin
        x = _extract_stencil_values(xs, side, Val(D + 1))
        f = _extract_stencil_values(ys, side, Val(D + 1))
        coeffs = _compute_deriv1_coeffs(pf, side, x)
        return _weighted_sum(coeffs, f)
    end
end
