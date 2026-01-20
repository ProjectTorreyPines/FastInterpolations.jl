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
    CubicSplineCache{T,X,F,BC,S}

Cache structure for cubic spline interpolation with reusable LU factorization.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `X`: Grid type (Vector{T} or AbstractRange{T})
- `F`: LU factorization type
- `BC`: Boundary condition data type (BCPair{T,L,R} for derivative BC, PeriodicData{T} for periodic)
- `S`: Grid spacing type (ScalarSpacing{T} for Range, VectorSpacing{T} for Vector)

# Fields
- `x::X`: Grid points (immutable after construction, can be Range or Vector)
- `spacing::S`: Grid spacing data (ScalarSpacing for uniform, VectorSpacing for non-uniform)
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

# Performance
The `spacing` field stores grid spacing with precomputed reciprocals.
- `ScalarSpacing`: O(1) memory for uniform grids (Range inputs) with constant propagation
- `VectorSpacing`: O(N) memory for non-uniform grids (Vector inputs)

Both store `inv_h` for eliminating floating-point division in kernel hot paths.
On ARM64 (Apple Silicon), this reduces per-evaluation latency from ~10 cycles (fdiv)
to ~4 cycles (fmul) — a 2.5× speedup in the inner loop.

# Boundary Conditions
- `bc=NaturalBC()` (default): Natural spline with z[1] = z[n+1] = 0
- `bc=PeriodicBC()`: Periodic spline with C2 continuity at boundaries
"""
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F,BC,S<:AbstractGridSpacing{T}}
    x::X
    spacing::S
    lu_factor::F
    bc_config::BC
end

# ExtrapVal is defined in ops.jl (shared between linear and cubic)

"""
    CubicInterpolant{T,C,P}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `cubic_interp(x, y)` (2-argument form).

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `C`: CubicSplineCache type (preserves grid type info for O(1) vs O(log n) lookup)
- `P`: Search policy type (Binary, HintedBinary, LinearBounded, etc.)

# Fields
- `cache::C`: Pre-computed CubicSplineCache (LU factorization)
- `y::Vector{T}`: y-values (function values at grid points)
- `z::Vector{T}`: Pre-computed second derivative coefficients (solves system once!)
- `extrap::ExtrapVal`: Extrapolation mode (union-split for efficient dispatch)
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
itp = cubic_interp(x, y)
result = @. coef * itp(rho) * other_terms  # fused, zero-allocation per call
val = itp(0.5)                              # scalar (zero-allocation)

# Create with custom search policy
itp = cubic_interp(x, y; search=LinearBounded())
val = itp(0.5)                              # uses LinearBounded() by default
val = itp(0.5; search=Binary())             # override with Binary()
```

# Performance Notes
- System solved ONCE at construction -> z coefficients pre-computed
- Each scalar call just evaluates cubic polynomial (zero-allocation!)
- Broadcast operations are perfectly fused (no intermediate arrays)
- Extrapolation mode uses union-splitting for near-zero overhead dispatch
"""
struct CubicInterpolant{T<:AbstractFloat,C<:CubicSplineCache{T},P<:AbstractSearchPolicy} <: AbstractInterpolant{T}
    cache::C
    y::Vector{T}
    z::Vector{T}  # Pre-computed second derivative coefficients
    extrap::ExtrapVal  # Extrapolation mode (concrete union for union-splitting)
    search_policy::P  # Default search policy (immutable, thread-safe)
    function CubicInterpolant(
        cache::C,
        y::AbstractVector{T},
        z::AbstractVector{T},
        extrap::ExtrapVal,
        search::P=Binary()
    ) where {T<:AbstractFloat, C<:CubicSplineCache{T}, P<:AbstractSearchPolicy}
        @assert length(cache.x) == length(y) "cache grid and y must have same length"
        @assert length(cache.x) == length(z) "z coefficients must match grid length"
        # Always copy to ensure immutability: once constructed, the interpolant
        # owns its data and always returns identical results for the same query.
        # Without copying, external modifications to y or cache reuse could
        # silently corrupt results.
        new{T,C,P}(cache, Vector{T}(y), Vector{T}(z), extrap, search)
    end
end

# ========================================
# TransposeSnapshot Type (shared between multi-series interpolants)
# ========================================

"""
    TransposeSnapshot{T}

Immutable snapshot of point-contiguous (transposed) matrices.

Used for atomic swap in multi-series cubic interpolants to ensure thread-safe
lazy initialization of point-contiguous layout.

# Fields
- `y_point::Union{Nothing, Matrix{T}}`: Point-contiguous y values (n_series × n_points)
- `z_point::Union{Nothing, Matrix{T}}`: Point-contiguous z values (n_series × n_points)

# Thread Safety
Used with atomic operations for lock-free lazy initialization.
Multiple threads may compute the transpose simultaneously (benign duplication).
"""
struct TransposeSnapshot{T<:AbstractFloat}
    y_point::Union{Nothing, Matrix{T}}
    z_point::Union{Nothing, Matrix{T}}
end

# Empty snapshot constructor
TransposeSnapshot{T}() where {T<:AbstractFloat} = TransposeSnapshot{T}(nothing, nothing)
