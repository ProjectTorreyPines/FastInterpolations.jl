# ========================================
# Cubic Spline Type Definitions
# ========================================
# Structs for cubic spline interpolation.
# Separated from cubic_interp.jl for clarity.
# Include order: utils.jl → cubic_types.jl → cubic_interp.jl

"""
    PeriodicData{T}

Pre-computed data for Sherman-Morrison periodic spline solver.

# Fields
- `q::Vector{T}`: Pre-computed A'^{-1} * u vector for Sherman-Morrison formula
- `y_temp::Vector{T}`: Workspace for intermediate solution (for zero-allocation ldiv!)
- `period::T`: Period T = x[end] - x[1]

# Notes
For cyclic tridiagonal system A_cyclic = A' + u * v^T, Sherman-Morrison gives:
    z = y - ((v^T * y) / (1 + v^T * q)) * q
where q = A'^{-1} * u is pre-computed and reused for different y vectors.
"""
struct PeriodicData{T<:AbstractFloat}
    q::Vector{T}       # Pre-computed A'^{-1} * u
    y_temp::Vector{T}  # Workspace for ldiv! (zero-allocation solver)
    period::T          # x[end] - x[1]
end

"""
    CubicSplineCache{T,X,F,BC}

Cache structure for cubic spline interpolation with reusable LU factorization.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `X`: Grid type (Vector{T} or AbstractRange{T})
- `F`: LU factorization type
- `BC`: Boundary condition data type (Nothing for natural, PeriodicData{T} for periodic)

# Fields
- `x::X`: Grid points (immutable after construction, can be Range or Vector)
- `h::Vector{T}`: Grid spacing h[i] = x[i+1] - x[i]
- `lu_factor::F`: LU factorization of tridiagonal matrix A
- `d_workspace::Vector{T}`: Workspace for RHS vector computation
- `z_workspace::Vector{T}`: Workspace for solution vector
- `bc_data::BC`: Boundary condition data (Nothing for natural, PeriodicData for periodic)

# Notes
The LU factorization depends ONLY on x geometry and can be reused for:
- Different y vectors (varying function values)
- Different x_query vectors (varying query points)

When x is an AbstractRange, O(1) index lookup is used instead of O(log n) binary search.

# Boundary Conditions
- `bc=:natural` (default): Natural spline with z[1] = z[n+1] = 0
- `bc=:periodic`: Periodic spline with C2 continuity at boundaries
"""
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F,BC}
    x::X
    h::Vector{T}
    lu_factor::F
    d_workspace::Vector{T}
    z_workspace::Vector{T}
    bc_data::BC
end

"""
    CubicInterpolant{T,C,Y,Z,E}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `cubic_interp(x, y)` (2-argument form).

# Fields
- `cache::C`: Pre-computed CubicSplineCache (LU factorization)
- `y::Y`: y-values (function values at grid points)
- `z::Z`: Pre-computed second derivative coefficients (solves system once!)
- `extrap::E`: Extrapolation mode (Val{:none}, Val{:constant}, Val{:extension}, Val{:wrap})

# Usage
```julia
itp = cubic_interp(x, y)
result = @. coef * itp(rho) * other_terms  # fused, zero-allocation per call
val = itp(0.5)                              # scalar (zero-allocation)
```

# Performance Notes
- System solved ONCE at construction -> z coefficients pre-computed
- Each scalar call just evaluates cubic polynomial (zero-allocation!)
- Broadcast operations are perfectly fused (no intermediate arrays)
"""
struct CubicInterpolant{T<:AbstractFloat,C<:CubicSplineCache{T},Y<:AbstractVector{T},Z<:AbstractVector{T},E<:Val}
    cache::C
    y::Y
    z::Z  # Pre-computed second derivative coefficients
    extrap::E  # Extrapolation mode

    function CubicInterpolant(
        cache::C,
        y::Y,
        z::Z,
        extrap::E
    ) where {T<:AbstractFloat, C<:CubicSplineCache{T}, Y<:AbstractVector{T}, Z<:AbstractVector{T}, E<:Val}
        @assert length(cache.x) == length(y) "cache grid and y must have same length"
        @assert length(cache.x) == length(z) "z coefficients must match grid length"
        new{T,C,Y,Z,E}(cache, y, z, extrap)
    end
end
