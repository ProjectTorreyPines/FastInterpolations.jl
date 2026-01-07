# ========================================
# Cubic Spline Type Definitions
# ========================================
# Structs for cubic spline interpolation.
# Separated from cubic_interp.jl for clarity.
# Include order: utils.jl → bc_types.jl → cubic_types.jl → cubic_solver.jl → cubic_interp.jl

# Boundary condition types (AbstractBC, PointBC, Deriv1, Deriv2, BCPair, PeriodicBC) are defined in bc_types.jl

"""
    PeriodicData{T}

Pre-computed data for Sherman-Morrison periodic spline solver.

# Fields
- `q::Vector{T}`: Pre-computed A'^{-1} * u vector for Sherman-Morrison formula
- `period::T`: Period T = x[end] - x[1]

# Notes
For cyclic tridiagonal system A_cyclic = A' + u * v^T, Sherman-Morrison gives:
    z = y - ((v^T * y) / (1 + v^T * q)) * q
where q = A'^{-1} * u is pre-computed and reused for different y vectors.

# Thread-Safety
Workspaces for the periodic solver are allocated from task-local pools via `@with_pool`,
not stored in this struct. This eliminates shared mutable state.
"""
struct PeriodicData{T<:AbstractFloat}
    q::Vector{T}  # Pre-computed A'^{-1} * u
    period::T     # x[end] - x[1]
end

"""
    CubicSplineCache{T,X,F,BC}

Cache structure for cubic spline interpolation with reusable LU factorization.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `X`: Grid type (Vector{T} or AbstractRange{T})
- `F`: LU factorization type
- `BC`: Boundary condition data type (BCPair{T,L,R} for derivative BC, PeriodicData{T} for periodic)

# Fields
- `x::X`: Grid points (immutable after construction, can be Range or Vector)
- `h::Vector{T}`: Grid spacing h[i] = x[i+1] - x[i] (standard 1-based indexing, size n)
- `inv_h::Vector{T}`: Precomputed reciprocals inv_h[i] = 1/h[i] (eliminates fdiv in kernels)
- `lu_factor::F`: LU factorization of tridiagonal matrix A
- `bc_config::BC`: Boundary condition data (BCPair for derivative BC, PeriodicData for periodic)

# Notes
The LU factorization depends ONLY on x geometry and can be reused for:
- Different y vectors (varying function values)
- Different x_query vectors (varying query points)

When x is an AbstractRange, O(1) index lookup is used instead of O(log n) binary search.

# Thread-Safety
Workspaces (d, z) are allocated from task-local pools via `@with_pool`,
not stored in this struct. This makes the cache thread-safe by design.

# Boundary Conditions
- `bc=NaturalBC()` (default): Natural spline with z[1] = z[n+1] = 0
- `bc=PeriodicBC()`: Periodic spline with C2 continuity at boundaries
"""
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F,BC}
    x::X
    h::Vector{T}
    inv_h::Vector{T}
    lu_factor::F
    bc_config::BC
end

# ExtrapVal is defined in ops.jl (shared between linear and cubic)

"""
    CubicInterpolant{T,C}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `cubic_interp(x, y)` (2-argument form).

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `C`: CubicSplineCache type (preserves grid type info for O(1) vs O(log n) lookup)

# Fields
- `cache::C`: Pre-computed CubicSplineCache (LU factorization)
- `y::Vector{T}`: y-values (function values at grid points)
- `z::Vector{T}`: Pre-computed second derivative coefficients (solves system once!)
- `extrap::ExtrapVal`: Extrapolation mode (union-split for efficient dispatch)

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
- Extrapolation mode uses union-splitting for near-zero overhead dispatch
"""
struct CubicInterpolant{T<:AbstractFloat,C<:CubicSplineCache{T}}
    cache::C
    y::Vector{T}
    z::Vector{T}  # Pre-computed second derivative coefficients
    extrap::ExtrapVal  # Extrapolation mode (concrete union for union-splitting)

    function CubicInterpolant(
        cache::C,
        y::AbstractVector{T},
        z::AbstractVector{T},
        extrap::ExtrapVal
    ) where {T<:AbstractFloat, C<:CubicSplineCache{T}}
        @assert length(cache.x) == length(y) "cache grid and y must have same length"
        @assert length(cache.x) == length(z) "z coefficients must match grid length"
        # Always copy to ensure immutability: once constructed, the interpolant
        # owns its data and always returns identical results for the same query.
        # Without copying, external modifications to y or cache reuse could
        # silently corrupt results.
        new{T,C}(cache, Vector{T}(y), Vector{T}(z), extrap)
    end
end
