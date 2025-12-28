# ========================================
# Cubic Spline System Builders and Solvers
# ========================================
# Internal functions for building cache and solving tridiagonal systems.
# Include order: bc_types.jl → cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

# BC types and normalization functions are defined in bc_types.jl

# ========================================
# Row Builders for Generic BC (Type Dispatch)
# ========================================

# First row - Deriv2 (second derivative specified): z[1] = bc.val
@inline function _set_first_row!(
    d_diag::AbstractVector{T}, du::AbstractVector{T}, ::Deriv2{T}, ::AbstractVector{T}
) where {T<:AbstractFloat}
    d_diag[1] = one(T)
    du[1] = zero(T)
    return nothing
end

# First row - Deriv1 (first derivative specified): 2h₁z₁ + h₁z₂ = 6[(y₂-y₁)/h₁ - S'(x₁)]
@inline function _set_first_row!(
    d_diag::AbstractVector{T}, du::AbstractVector{T}, ::Deriv1{T}, h::AbstractVector{T}
) where {T<:AbstractFloat}
    d_diag[1] = 2 * h[2]
    du[1] = h[2]
    return nothing
end

# Last row - Deriv2 (second derivative specified): z[n+1] = bc.val
@inline function _set_last_row!(
    dl::AbstractVector{T}, d_diag::AbstractVector{T}, ::Deriv2{T}, ::AbstractVector{T}, n::Int
) where {T<:AbstractFloat}
    dl[n] = zero(T)
    d_diag[n+1] = one(T)
    return nothing
end

# Last row - Deriv1 (first derivative specified): hₙzₙ + 2hₙzₙ₊₁ = 6[S'(xₙ₊₁) - (yₙ₊₁-yₙ)/hₙ]
@inline function _set_last_row!(
    dl::AbstractVector{T}, d_diag::AbstractVector{T}, ::Deriv1{T}, h::AbstractVector{T}, n::Int
) where {T<:AbstractFloat}
    dl[n] = h[n+1]
    d_diag[n+1] = 2 * h[n+1]
    return nothing
end

# ========================================
# Cache Builders
# ========================================

"Build cache for periodic cubic spline using Sherman-Morrison formula."
function _build_periodic_cache(x::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(x)
    n = N - 1  # Number of intervals

    length(x) >= 4 || throw(ArgumentError("Periodic spline requires at least 4 points"))

    period = x[N] - x[1]

    # Compute grid spacing h
    h = Vector{T}(undef, n + 2)
    h[1] = zero(T)
    h[end] = zero(T)
    @inbounds for i in 1:n
        h[i+1] = x[i+1] - x[i]
    end

    # Build modified tridiagonal matrix A' for Sherman-Morrison
    α = h[n+1]

    dl = Vector{T}(undef, n - 1)
    d_diag = Vector{T}(undef, n)
    du = Vector{T}(undef, n - 1)

    d_diag[1] = h[n+1] + 2 * h[2]

    @inbounds for i in 2:n-1
        dl[i-1] = h[i]
        d_diag[i] = 2 * (h[i] + h[i+1])
        du[i-1] = h[i+1]
    end

    dl[n-1] = h[n]
    d_diag[n] = 2 * h[n] + h[n+1]

    if n > 1
        du[n-1] = h[n]
    end

    tA_prime = Tridiagonal(dl, d_diag, du)
    lu_factor = lu(tA_prime)

    # Pre-compute q = A'^{-1} * u
    u = zeros(T, n)
    u[1] = one(T)
    u[n] = one(T)
    q = lu_factor \ u

    d_workspace = Vector{T}(undef, n)
    z_workspace = Vector{T}(undef, n + 1)
    y_temp_workspace = Vector{T}(undef, n)

    bc_data = PeriodicData(q, y_temp_workspace, period)

    return CubicSplineCache(x, h[1:n+1], lu_factor, d_workspace, z_workspace, bc_data)
end

"""
Build cache for generic derivative BC (Deriv1/Deriv2 combinations).
Uses type dispatch for zero-overhead specialization.
"""
function _build_derivative_bc_cache(
    x::AbstractVector{T},
    left_bc::L,
    right_bc::R
) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    n = length(x) - 1

    # Compute grid spacing h[i] = x[i+1] - x[i]
    h = Vector{T}(undef, n + 2)
    h[1] = zero(T)
    h[end] = zero(T)
    @inbounds for i in 1:n
        h[i+1] = x[i+1] - x[i]
    end

    # Build tridiagonal matrix A
    dl = Vector{T}(undef, n)       # Lower diagonal
    d_diag = Vector{T}(undef, n+1) # Main diagonal
    du = Vector{T}(undef, n)       # Upper diagonal

    # First and last rows depend on BC type (type dispatch)
    _set_first_row!(d_diag, du, left_bc, h)
    _set_last_row!(dl, d_diag, right_bc, h, n)

    # Interior rows (same for all BC types)
    @inbounds for i in 2:n
        dl[i-1] = h[i]
        d_diag[i] = 2 * (h[i] + h[i+1])
        du[i] = h[i+1]
    end

    tA = Tridiagonal(dl, d_diag, du)
    lu_factor = lu(tA)

    # Allocate workspaces
    d_workspace = Vector{T}(undef, n + 1)
    z_workspace = Vector{T}(undef, n + 1)

    bc_data = BCPair(left_bc, right_bc)

    return CubicSplineCache(x, h[1:n+1], lu_factor, d_workspace, z_workspace, bc_data)
end

# ========================================
# RHS Computation
# ========================================

# ----------------------------------------
# RHS helpers for generic BC (type dispatch)
# ----------------------------------------

# First element - Deriv2: d[1] = bc.val (second derivative value)
@inline function _compute_rhs_first!(
    d::AbstractVector{T}, bc::Deriv2{T}, ::AbstractVector{T}, ::AbstractVector{T}
) where {T<:AbstractFloat}
    d[1] = bc.val
    return nothing
end

# First element - Deriv1: d[1] = 6[(y₂-y₁)/h₁ - S'(x₁)]
@inline function _compute_rhs_first!(
    d::AbstractVector{T}, bc::Deriv1{T}, y::AbstractVector{T}, h::AbstractVector{T}
) where {T<:AbstractFloat}
    d[1] = 6 * ((y[2] - y[1]) / h[2] - bc.val)
    return nothing
end

# Last element - Deriv2: d[n+1] = bc.val (second derivative value)
@inline function _compute_rhs_last!(
    d::AbstractVector{T}, bc::Deriv2{T}, ::AbstractVector{T}, ::AbstractVector{T}, n::Int
) where {T<:AbstractFloat}
    d[n+1] = bc.val
    return nothing
end

# Last element - Deriv1: d[n+1] = 6[S'(xₙ₊₁) - (yₙ₊₁-yₙ)/hₙ]
@inline function _compute_rhs_last!(
    d::AbstractVector{T}, bc::Deriv1{T}, y::AbstractVector{T}, h::AbstractVector{T}, n::Int
) where {T<:AbstractFloat}
    d[n+1] = 6 * (bc.val - (y[n+1] - y[n]) / h[n+1])
    return nothing
end

"""
Compute RHS vector for generic derivative BC system in-place.
"""
@inline function compute_rhs!(
    d::AbstractVector{T}, y::AbstractVector{T}, h::AbstractVector{T},
    bc_data::BCPair{T,L,R}
) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    n = length(y) - 1
    _compute_rhs_first!(d, bc_data.left, y, h)
    @inbounds for i in 2:n
        d[i] = 6 * ((y[i+1] - y[i]) / h[i+1] - (y[i] - y[i-1]) / h[i])
    end
    _compute_rhs_last!(d, bc_data.right, y, h, n)
    return nothing
end

# ----------------------------------------
# Periodic RHS function
# ----------------------------------------

"Compute RHS vector d for periodic cubic spline system in-place."
@inline function compute_rhs_periodic!(d::Vector{T}, y::AbstractVector{T}, h::Vector{T}) where {T}
    N = length(y)
    n = N - 1

    @inbounds d[1] = 6 * (y[2] - y[1]) / h[2] - 6 * (y[1] - y[N-1]) / h[n+1]

    @inbounds for i in 2:n-1
        d[i] = 6 * (y[i+1] - y[i]) / h[i+1] - 6 * (y[i] - y[i-1]) / h[i]
    end

    @inbounds d[n] = 6 * (y[N] - y[N-1]) / h[n+1] - 6 * (y[N-1] - y[N-2]) / h[n]

    return nothing
end

# ========================================
# System Solvers
# ========================================

"Solve periodic cyclic tridiagonal system using Sherman-Morrison formula."
@inline function _solve_cubic_system_periodic!(
    z_workspace::Vector{T},
    d_workspace::Vector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T}
) where {T<:AbstractFloat, X, F}
    N = length(y)
    n = N - 1

    compute_rhs_periodic!(d_workspace, y, cache.h)

    y_temp = cache.bc_data.y_temp
    ldiv!(y_temp, cache.lu_factor, d_workspace)

    α = cache.h[n+1]
    q = cache.bc_data.q

    vTy = α * (y_temp[1] + y_temp[n])
    vTq = α * (q[1] + q[n])

    denom = one(T) + vTq
    # Use sqrt(eps) for numerical stability: catches near-degenerate cases
    # where division would cause significant precision loss (~1e-8 for Float64)
    tol = sqrt(eps(T))
    if abs(denom) < tol
        throw(DomainError(denom,
            "Sherman-Morrison formula failed: denominator (1 + v'q) ≈ 0.\n" *
            "  denom = $denom (tol = $tol), α = $α, q[1] = $(q[1]), q[n] = $(q[n])\n" *
            "  This usually indicates a degenerate or ill-conditioned periodic grid."))
    end
    factor = vTy / denom

    @inbounds for i in 1:n
        z_workspace[i] = y_temp[i] - factor * q[i]
    end

    z_workspace[n+1] = z_workspace[1]

    return z_workspace
end

# ========================================
# Unified System Solver Entry Point
# ========================================
#
# All callers must explicitly pass bc_data:
#   _solve_system!(cache, y, cache.bc_data)
#
# This enables cache reuse with different BC values at solve time.

"Solve cubic spline system (Periodic BC). bc_data ignored, for API consistency."
@inline function _solve_system!(
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    ::PeriodicData{T}  # Unused, for API consistency with BCPair version
) where {T<:AbstractFloat, X, F}
    _solve_cubic_system_periodic!(cache.z_workspace, cache.d_workspace, cache, y)
end

"""
Solve cubic spline system with explicit BC values.
BC types must match cache BC types.
"""
@inline function _solve_system!(
    cache::CubicSplineCache{T,X,F,BCPair{T,L,R}},
    y::AbstractVector{T},
    bc_pair::BCPair{T,L,R}
) where {T<:AbstractFloat, X, F, L<:PointBC{T}, R<:PointBC{T}}
    compute_rhs!(cache.d_workspace, y, cache.h, bc_pair)
    ldiv!(cache.z_workspace, cache.lu_factor, cache.d_workspace)
    return cache.z_workspace
end
