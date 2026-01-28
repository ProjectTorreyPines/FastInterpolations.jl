# ========================================
# Quadratic Spline Coefficient Computation
# ========================================
# Functions to compute spline coefficients (s, d, a) from grid and values.
#
# Mathematical model:
#   S_i(x) = a_i*(x - x_i)² + d_i*(x - x_i) + y_i
#
# Coefficient computation:
#   1. s[i] = (y[i+1] - y[i]) / h[i]  (secant slopes)
#   2. Fill d[] via BC-dependent recurrence:
#      - Left BC:  d[1] from BC, forward  recurrence d[i+1] = 2*s[i] - d[i]
#      - Right BC: d[n] from BC, backward recurrence d[i] = 2*s[i] - d[i+1]
#   3. a[i] = (s[i] - d[i]) / h[i]  (quadratic coefficients)

# ========================================
# Type Alias for Quadratic BC
# ========================================

"""
Supported boundary conditions for quadratic spline interpolation.
- `Left{T}`: BC at left endpoint (forward recurrence)
- `Right{T}`: BC at right endpoint (backward recurrence)
- `MinCurvFit{T}`: Global curvature minimization
"""
const QuadraticBC{T} = Union{Left{T}, Right{T}, MinCurvFit{T}}

# ========================================
# Secant Computation
# ========================================

"""
    _compute_quadratic_secants!(s, y, inv_h)

Compute secant slopes: s[i] = (y[i+1] - y[i]) * inv_h[i]

# Arguments
- `s::Vector{T}`: Output vector (length n-1)
- `y::AbstractVector{T}`: Values at grid points (length n)
- `inv_h::Vector{T}`: Inverse grid spacing (length n-1)
"""
@inline function _compute_quadratic_secants!(s::AbstractVector{T}, y::AbstractVector{T}, inv_h::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y) - 1
    @inbounds for i in 1:n
        s[i] = (y[i+1] - y[i]) * inv_h[i]
    end
    return s
end

# ========================================
# Recurrence Functions
# ========================================

"""
    _forward_recurrence!(d, s, d1)

Fill slope array using forward recurrence from d[1].
d[i+1] = 2*s[i] - d[i]

# Arguments
- `d::Vector{T}`: Output slope array (length n)
- `s::Vector{T}`: Secant slopes (length n-1)
- `d1::T`: Initial slope d[1]
"""
@inline function _forward_recurrence!(d::AbstractVector{T}, s::AbstractVector{T}, d1::T) where {T<:AbstractFloat}
    d[1] = d1
    n = length(d)
    @inbounds for i in 1:(n-1)
        d[i+1] = 2*s[i] - d[i]
    end
    return d
end

"""
    _backward_recurrence!(d, s, dn)

Fill slope array using backward recurrence from d[n].
d[i] = 2*s[i] - d[i+1]

# Arguments
- `d::Vector{T}`: Output slope array (length n)
- `s::Vector{T}`: Secant slopes (length n-1)
- `dn::T`: Final slope d[n]
"""
@inline function _backward_recurrence!(d::AbstractVector{T}, s::AbstractVector{T}, dn::T) where {T<:AbstractFloat}
    n = length(d)
    d[n] = dn
    @inbounds for i in (n-1):-1:1
        d[i] = 2*s[i] - d[i+1]
    end
    return d
end

# ========================================
# Slope Filling (BC-Dispatched)
# ========================================

"""
    _fill_slopes!(d, s, h, bc, x, y)

Fill slope array d[] based on boundary condition type.
Dispatches at compile time to use optimal recurrence direction:
- Left BC:  compute d[1], forward recurrence  → O(n)
- Right BC: compute d[n], backward recurrence → O(n)

The `x` and `y` parameters are needed for PolyFit{D} BCs which estimate
derivatives from data. For other BC types, they are ignored.
"""
# Left(Deriv1): d[1] given directly, forward recurrence
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T},
                               bc::Left{T, Deriv1{T}}, ::AbstractVector{T}, ::AbstractVector{T}) where {T<:AbstractFloat}
    d1 = bc.bc.val
    _forward_recurrence!(d, s, d1)
end

# Left(Deriv2): d[1] = s[1] - (κ/2)*h[1], forward recurrence
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T},
                               bc::Left{T, Deriv2{T}}, ::AbstractVector{T}, ::AbstractVector{T}) where {T<:AbstractFloat}
    κ = bc.bc.val
    d1 = s[1] - (κ / 2) * h[1]
    _forward_recurrence!(d, s, d1)
end

# Right(Deriv1): d[n] given directly, backward recurrence
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T},
                               bc::Right{T, Deriv1{T}}, ::AbstractVector{T}, ::AbstractVector{T}) where {T<:AbstractFloat}
    dn = bc.bc.val
    _backward_recurrence!(d, s, dn)
end

# Right(Deriv2): compute d[n] from curvature, backward recurrence
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T},
                               bc::Right{T, Deriv2{T}}, ::AbstractVector{T}, ::AbstractVector{T}) where {T<:AbstractFloat}
    κ = bc.bc.val
    # a[n-1] = κ/2
    # d[n-1] = s[n-1] - a[n-1]*h[n-1]
    # d[n] = 2*a[n-1]*h[n-1] + d[n-1] = s[n-1] + (κ/2)*h[n-1]
    dn = s[end] + (κ / 2) * h[end]
    _backward_recurrence!(d, s, dn)
end

# MinCurvFit: minimize total curvature via closed-form optimization
"""
    _fill_slopes!(d, s, h, ::MinCurvFit, x, y)

Fill slope array using global curvature minimization.

Minimizes total curvature: ∫(S'')² dx = Σ 4*a[i]²*h[i] = Σ (s[i] - d[i])²/h[i]

# Mathematical Derivation
The slope d[i] depends on d[1] via forward recurrence:
- d[i] = α[i] * d[1] + β[i]  where α[i] = (-1)^(i+1) (alternating sign)
- β[1] = 0, β[i+1] = 2*s[i] - β[i]

Setting df/d(d[1]) = 0 gives the closed-form solution:
- d[1]_optimal = [Σ α[i]*(s[i] - β[i])/h[i]] / [Σ 1/h[i]]

# Complexity
O(n) time, O(1) extra space (on-the-fly β computation).
"""
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T},
                               ::MinCurvFit{T}, ::AbstractVector{T}, ::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(d)
    n_intervals = n - 1  # = length(s) = length(h)

    # Edge case: single segment (n=2)
    # For single segment, minimize a² = (s-d[1])²/h
    # This means d[1] = s[1] (making a = 0, zero curvature)
    if n == 2
        d1 = @inbounds s[1]
        _forward_recurrence!(d, s, d1)
        return d
    end

    # Compute optimal d[1] using closed-form solution
    # d[i] = α[i] * d[1] + β[i]
    # α[i] = (-1)^(i+1): +1, -1, +1, -1, ...
    # β[i+1] = 2*s[i] - β[i], β[1] = 0

    # Objective: minimize Σ (s[i] - d[i])²/h[i]
    # = Σ (s[i] - α[i]*d[1] - β[i])²/h[i]
    # df/d(d[1]) = -2 * Σ α[i]*(s[i] - α[i]*d[1] - β[i])/h[i] = 0
    # Note: α[i]² = 1

    # Rearranging:
    # Σ α[i]*(s[i] - β[i])/h[i] = d[1] * Σ 1/h[i]
    # d[1] = [Σ α[i]*(s[i] - β[i])/h[i]] / [Σ 1/h[i]]

    inv_h_sum = zero(T)
    numerator = zero(T)
    β = zero(T)
    sign = one(T)  # α[1] = (-1)^(1+1) = +1

    @inbounds for i in 1:n_intervals
        inv_h_i = inv(h[i])
        inv_h_sum += inv_h_i
        numerator += sign * (s[i] - β) * inv_h_i
        β = 2*s[i] - β
        sign = -sign  # alternate: +1, -1, +1, ...
    end

    d1_optimal = numerator / inv_h_sum
    _forward_recurrence!(d, s, d1_optimal)
end

# ========================================
# Generic PolyFit{D}: Materialize to Deriv1
# ========================================

"""
    _fill_slopes!(d, s, h, bc::Left{T, PolyFit{D,T}}, x, y)

Fill slope array using generic polynomial fit at left endpoint.

Materializes PolyFit{D} to Deriv1 using `materialize_bc`, then delegates to
the Deriv1 code path. Supports all polynomial degrees: LinearFit (D=1),
QuadraticFit (D=2), CubicFit (D=3), etc.
"""
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T},
                               bc::Left{T, PolyFit{D, T}}, x::AbstractVector{T}, y::AbstractVector{T}) where {D, T<:AbstractFloat}
    # Materialize PolyFit{D} → Deriv1 using estimated derivative
    concrete_bc = materialize_bc(bc.bc, x, y, Val(:left))
    # Delegate to Deriv1 code path
    _fill_slopes!(d, s, h, Left(concrete_bc), x, y)
end

"""
    _fill_slopes!(d, s, h, bc::Right{T, PolyFit{D,T}}, x, y)

Fill slope array using generic polynomial fit at right endpoint.

Materializes PolyFit{D} to Deriv1 using `materialize_bc`, then delegates to
the Deriv1 code path.
"""
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T},
                               bc::Right{T, PolyFit{D, T}}, x::AbstractVector{T}, y::AbstractVector{T}) where {D, T<:AbstractFloat}
    # Materialize PolyFit{D} → Deriv1 using estimated derivative
    concrete_bc = materialize_bc(bc.bc, x, y, Val(:right))
    # Delegate to Deriv1 code path
    _fill_slopes!(d, s, h, Right(concrete_bc), x, y)
end

# ========================================
# Quadratic Coefficient Computation
# ========================================

"""
    _compute_quadratic_coefficients!(a, d, s, inv_h)

Compute quadratic coefficients: a[i] = (s[i] - d[i]) * inv_h[i]

# Arguments
- `a::Vector{T}`: Output coefficient array (length n-1)
- `d::Vector{T}`: Slope array (length n)
- `s::Vector{T}`: Secant slopes (length n-1)
- `inv_h::Vector{T}`: Inverse grid spacing (length n-1)
"""
@inline function _compute_quadratic_coefficients!(a::AbstractVector{T}, d::AbstractVector{T}, s::AbstractVector{T}, inv_h::AbstractVector{T}) where {T<:AbstractFloat}
    @inbounds for i in eachindex(a)
        a[i] = (s[i] - d[i]) * inv_h[i]
    end
    return a
end

# ========================================
# Grid Spacing Computation
# ========================================

"""
    _compute_grid_spacing!(h, inv_h, x)

Fill pre-allocated h and inv_h arrays with grid spacing and inverse.

# Arguments
- `h::AbstractVector{T}`: Output grid spacing (length n-1)
- `inv_h::AbstractVector{T}`: Output inverse grid spacing (length n-1)
- `x::AbstractVector{T}`: x-coordinates (length n)
"""
@inline function _compute_grid_spacing!(
    h::AbstractVector{T},
    inv_h::AbstractVector{T},
    x::AbstractVector{T}
) where {T<:AbstractFloat}
    @inbounds for i in eachindex(h, inv_h)
        h[i] = x[i+1] - x[i]
        inv_h[i] = inv(h[i])
    end
    return nothing
end

# ========================================
# Coefficient Computation (In-Place)
# ========================================

"""
    _compute_quadratic_coeffs!(h, d, a, x, y, bc)

Fill pre-allocated coefficient arrays for quadratic spline.
Uses AdaptiveArrayPools internally for temporary arrays (`inv_h`, `secant`).

# Arguments (outputs first, then inputs)
- `h::AbstractVector{FT}`: Grid spacing (length n-1)
- `d::AbstractVector{FT}`: Slope coefficients (length n)
- `a::AbstractVector{FT}`: Quadratic coefficients (length n-1)
- `x::AbstractVector{FT}`: x-coordinates (length n)
- `y::AbstractVector{FT}`: y-values (length n)
- `bc::QuadraticBC{FT}`: Boundary condition (Left, Right, or MinCurvFit)

# Note
Intermediate arrays (`inv_h`, `secant`) are acquired from thread-local pool
and automatically released when the function returns.
"""
@with_pool pool function _compute_quadratic_coeffs!(
    h::AbstractVector{FT},
    d::AbstractVector{FT},
    a::AbstractVector{FT},
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    bc::QuadraticBC{FT}
) where {FT<:AbstractFloat}
    nx = length(x)

    inv_h = acquire!(pool, FT, nx-1) # Inverse grid spacing
    secant = acquire!(pool, FT, nx-1) # secant slopes

    # 1. Compute grid spacing
    _compute_grid_spacing!(h, inv_h, x)

    # 2. Compute secants
    _compute_quadratic_secants!(secant, y, inv_h)

    # 3. Fill slopes d[] (BC-dispatched: Left→forward, Right→backward)
    _fill_slopes!(d, secant, h, bc, x, y)

    # 4. Compute quadratic coefficients a[]
    _compute_quadratic_coefficients!(a, d, secant, inv_h)

    return nothing
end

# ========================================
# Coefficient Computation (Allocating)
# ========================================

"""
    _compute_quadratic_coeffs(x, y, bc) -> (h, d, a)

Compute quadratic spline coefficients (allocating version).
Returns only the arrays needed for evaluation: `h`, `d`, `a`.

Intermediate arrays (`inv_h`, `secant`) are handled internally via
AdaptiveArrayPools and not returned.

For repeated interpolation on the same grid, use `QuadraticInterpolant`
which stores precomputed coefficients.
"""
function _compute_quadratic_coeffs(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    bc::QuadraticBC{FT}
) where {FT<:AbstractFloat}
    nx = length(x)

    # Allocate all arrays
    h = Vector{FT}(undef, nx-1)
    d = Vector{FT}(undef, nx)
    a = Vector{FT}(undef, nx-1)

    # Fill using in-place version
    _compute_quadratic_coeffs!(h, d, a, x, y, bc)

    return h, d, a
end
