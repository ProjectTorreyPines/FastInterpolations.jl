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
    _fill_slopes!(d, s, h, bc)

Fill slope array d[] based on boundary condition type.
Dispatches at compile time to use optimal recurrence direction:
- Left BC:  compute d[1], forward recurrence  → O(n)
- Right BC: compute d[n], backward recurrence → O(n)
"""
# Left(Deriv1): d[1] given directly, forward recurrence
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T}, bc::Left{T, Deriv1{T}}) where {T<:AbstractFloat}
    d1 = bc.bc.val
    _forward_recurrence!(d, s, d1)
end

# Left(Deriv2): d[1] = s[1] - (κ/2)*h[1], forward recurrence
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T}, bc::Left{T, Deriv2{T}}) where {T<:AbstractFloat}
    κ = bc.bc.val
    d1 = s[1] - (κ / 2) * h[1]
    _forward_recurrence!(d, s, d1)
end

# Right(Deriv1): d[n] given directly, backward recurrence
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T}, bc::Right{T, Deriv1{T}}) where {T<:AbstractFloat}
    dn = bc.bc.val
    _backward_recurrence!(d, s, dn)
end

# Right(Deriv2): compute d[n] from curvature, backward recurrence
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T}, h::AbstractVector{T}, bc::Right{T, Deriv2{T}}) where {T<:AbstractFloat}
    κ = bc.bc.val
    # a[n-1] = κ/2
    # d[n-1] = s[n-1] - a[n-1]*h[n-1]
    # d[n] = 2*a[n-1]*h[n-1] + d[n-1] = s[n-1] + (κ/2)*h[n-1]
    dn = s[end] + (κ / 2) * h[end]
    _backward_recurrence!(d, s, dn)
end

# ========================================
# ParabolaFit: 3-point derivative formula
# ========================================

"""
    _fill_slopes!(d, s, h, bc::Left{T, ParabolaFit{T}})

Fill slope array using parabola fit at left endpoint.

Uses the 3-point derivative formula to compute d[1] from the first 3 points,
then applies forward recurrence. This exactly reproduces any polynomial ≤ degree 2.

# Mathematical Derivation
For the first 3 points, the Lagrange interpolant derivative at x₀ is:
- d[1] = [s[1]·(2h[1]+h[2]) − s[2]·h[1]] / (h[1]+h[2])

For uniform grids (h[1] = h[2] = h), this simplifies to:
- d[1] = (3·s[1] − s[2]) / 2

# Edge Case
For n=2 (single segment), falls back to linear: d[1] = s[1].
"""
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T},
                               h::AbstractVector{T}, ::Left{T, ParabolaFit{T}}) where {T<:AbstractFloat}
    n = length(d)

    # Edge case: single segment (n=2) - fallback to linear
    if n == 2
        d1 = @inbounds s[1]
        _forward_recurrence!(d, s, d1)
        return d
    end

    # 3-point derivative formula for d[1]
    # d[1] = [s[1]·(2h[1]+h[2]) − s[2]·h[1]] / (h[1]+h[2])
    @inbounds begin
        h1, h2 = h[1], h[2]
        s1, s2 = s[1], s[2]
        d1 = (s1 * (2*h1 + h2) - s2 * h1) / (h1 + h2)
    end
    _forward_recurrence!(d, s, d1)
end

"""
    _fill_slopes!(d, s, h, bc::Right{T, ParabolaFit{T}})

Fill slope array using parabola fit at right endpoint.

Uses the 3-point derivative formula to compute d[n] from the last 3 points,
then applies backward recurrence. This exactly reproduces any polynomial ≤ degree 2.

# Mathematical Derivation
For the last 3 points, the Lagrange interpolant derivative at x_n is:
- d[n] = [s[n-1]·(h[n-2]+2h[n-1]) − s[n-2]·h[n-1]] / (h[n-2]+h[n-1])

For uniform grids, this simplifies to:
- d[n] = (3·s[n-1] − s[n-2]) / 2

# Edge Case
For n=2 (single segment), falls back to linear: d[n] = s[1].
"""
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T},
                               h::AbstractVector{T}, ::Right{T, ParabolaFit{T}}) where {T<:AbstractFloat}
    n = length(d)

    # Edge case: single segment (n=2) - fallback to linear
    if n == 2
        dn = @inbounds s[1]
        _backward_recurrence!(d, s, dn)
        return d
    end

    # 3-point derivative formula for d[n]
    # Using last 3 points: indices n-2, n-1, n (so secants at n-2 and n-1)
    # d[n] = [s[n-1]·(h[n-2]+2h[n-1]) − s[n-2]·h[n-1]] / (h[n-2]+h[n-1])
    @inbounds begin
        n_intervals = length(s)
        h_nm2 = h[n_intervals-1]  # h[n-2] in 1-based indexing
        h_nm1 = h[n_intervals]    # h[n-1]
        s_nm2 = s[n_intervals-1]  # s[n-2]
        s_nm1 = s[n_intervals]    # s[n-1]
        dn = (s_nm1 * (h_nm2 + 2*h_nm1) - s_nm2 * h_nm1) / (h_nm2 + h_nm1)
    end
    _backward_recurrence!(d, s, dn)
end


# MinCurvFit: minimize total curvature via closed-form optimization
"""
    _fill_slopes!(d, s, h, ::MinCurvFit)

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
@inline function _fill_slopes!(d::AbstractVector{T}, s::AbstractVector{T},
                               h::AbstractVector{T}, ::MinCurvFit{T}) where {T<:AbstractFloat}
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
