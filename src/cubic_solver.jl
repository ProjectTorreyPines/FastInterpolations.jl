# ========================================
# Cubic Spline System Builders and Solvers
# ========================================
# Internal functions for building cache and solving tridiagonal systems.
# Include order: cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

"Build cache for natural cubic spline (z[1] = z[n+1] = 0)."
function _build_natural_cache(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x) - 1

    # Compute grid spacing h[i] = x[i+1] - x[i]
    # h is padded: [0, h1, h2, ..., hn, 0]
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

    # First and last rows: z[1] = 0 and z[n+1] = 0 (natural boundary)
    d_diag[1] = one(T)
    du[1] = zero(T)

    # Interior rows
    @inbounds for i in 2:n
        dl[i-1] = h[i]
        d_diag[i] = 2 * (h[i] + h[i+1])
        du[i] = h[i+1]
    end

    # Last row
    dl[n] = zero(T)
    d_diag[n+1] = one(T)

    tA = Tridiagonal(dl, d_diag, du)
    lu_factor = lu(tA)

    # Allocate workspaces
    d_workspace = Vector{T}(undef, n + 1)
    z_workspace = Vector{T}(undef, n + 1)

    return CubicSplineCache(x, h[1:n+1], lu_factor, d_workspace, z_workspace, nothing)
end

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

# ========================================
# RHS Computation
# ========================================

"Compute RHS vector d for natural cubic spline system in-place."
@inline function compute_rhs!(d::Vector{T}, y::AbstractVector{T}, h::Vector{T}) where {T}
    n = length(y) - 1
    d[1] = zero(T)
    d[n+1] = zero(T)
    @inbounds for i in 2:n
        d[i] = 6 * (y[i+1] - y[i]) / h[i+1] - 6 * (y[i] - y[i-1]) / h[i]
    end
    return nothing
end

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

"Solve tridiagonal system for natural BC using pre-computed LU factorization."
@inline function _solve_cubic_system!(
    z_workspace::Vector{T},
    d_workspace::Vector{T},
    cache::CubicSplineCache{T,X,F,Nothing},
    y::AbstractVector{T}
) where {T<:AbstractFloat, X, F}
    compute_rhs!(d_workspace, y, cache.h)
    ldiv!(z_workspace, cache.lu_factor, d_workspace)
    return z_workspace
end

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

    factor = vTy / (one(T) + vTq)

    @inbounds for i in 1:n
        z_workspace[i] = y_temp[i] - factor * q[i]
    end

    z_workspace[n+1] = z_workspace[1]

    return z_workspace
end

# ========================================
# Unified System Solver Entry Point
# ========================================

"Solve cubic spline system (Natural BC)."
@inline function _solve_system!(
    cache::CubicSplineCache{T,X,F,Nothing},
    y::AbstractVector{T}
) where {T<:AbstractFloat, X, F}
    _solve_cubic_system!(cache.z_workspace, cache.d_workspace, cache, y)
end

"Solve cubic spline system (Periodic BC)."
@inline function _solve_system!(
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T}
) where {T<:AbstractFloat, X, F}
    _solve_cubic_system_periodic!(cache.z_workspace, cache.d_workspace, cache, y)
end
