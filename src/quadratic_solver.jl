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
#   2. d[1] from BC (see _compute_d1_from_bc)
#   3. d[i+1] = 2*s[i] - d[i]  (forward recurrence from d[1])
#   4. a[i] = (s[i] - d[i]) / h[i]  (quadratic coefficients)

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
    n = length(y)
    @inbounds for i in 1:(n-1)
        s[i] = (y[i+1] - y[i]) * inv_h[i]
    end
    return s
end

# ========================================
# Boundary Condition → d[1] Mapping
# ========================================

"""
    _compute_d1_from_bc(bc, s, h, n) -> d1::T

Compute d[1] (slope at first grid point) from boundary condition.

# BC Types and Formulas
- `Left(Deriv1(v))`: d[1] = v (given directly)
- `Left(Deriv2(κ))`: a[1] = κ/2, d[1] = s[1] - a[1]*h[1]
- `Right(Deriv1(v))`: d[n] = v, backward recurrence to d[1]
- `Right(Deriv2(κ))`: a[n-1] = κ/2, compute d[n], then backward recurrence
"""
# Left(Deriv1): slope at left endpoint given directly
@inline function _compute_d1_from_bc(bc::Left{T, Deriv1{T}}, s::AbstractVector{T}, h::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    return bc.bc.val
end

# Left(Deriv2): curvature at left endpoint
# a[1] = κ/2, d[1] = s[1] - a[1]*h[1]
@inline function _compute_d1_from_bc(bc::Left{T, Deriv2{T}}, s::AbstractVector{T}, h::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    κ = bc.bc.val
    a1 = κ / 2
    return s[1] - a1 * h[1]
end

# Right(Deriv1): slope at right endpoint, then backward recurrence
@inline function _compute_d1_from_bc(bc::Right{T, Deriv1{T}}, s::AbstractVector{T}, h::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    dn = bc.bc.val
    return _backward_recurrence_to_d1(s, dn, n)
end

# Right(Deriv2): curvature at right endpoint
# a[n-1] = κ/2, d[n-1] = s[n-1] - a[n-1]*h[n-1]
# d[n] = 2*a[n-1]*h[n-1] + d[n-1], then backward recurrence
@inline function _compute_d1_from_bc(bc::Right{T, Deriv2{T}}, s::AbstractVector{T}, h::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    κ = bc.bc.val
    # a[n-1] = κ/2
    a_nm1 = κ / 2
    # d[n-1] = s[n-1] - a[n-1]*h[n-1]
    d_nm1 = s[n-1] - a_nm1 * h[n-1]
    # d[n] = 2*a[n-1]*h[n-1] + d[n-1]
    dn = 2 * a_nm1 * h[n-1] + d_nm1
    return _backward_recurrence_to_d1(s, dn, n)
end

# ========================================
# Backward Recurrence Helper
# ========================================

"""
    _backward_recurrence_to_d1(s, dn, n) -> d1

Compute d[1] from d[n] using backward recurrence.
d[i] = 2*s[i] - d[i+1]
"""
@inline function _backward_recurrence_to_d1(s::AbstractVector{T}, dn::T, n::Int) where {T<:AbstractFloat}
    d = dn
    @inbounds for i in (n-1):-1:1
        d = 2*s[i] - d
    end
    return d
end

# ========================================
# Forward Recurrence
# ========================================

"""
    _forward_recurrence!(d, s, d1)

Fill slope array using forward recurrence.
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
