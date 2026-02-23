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

Cache structure for cubic spline interpolation with reusable Thomas factorization.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `X`: Grid type (Vector{T} or AbstractRange{T})
- `F`: Factorization type (ThomasFactorization{T,V})
- `BC`: Boundary condition data type (BCPair{L,R} for derivative BC, PeriodicData{T} for periodic)
- `S`: Grid spacing type (ScalarSpacing{T} for Range, VectorSpacing{T} for Vector)

# Fields
- `x::X`: Grid points (immutable after construction, can be Range or Vector)
- `spacing::S`: Grid spacing data (ScalarSpacing for uniform, VectorSpacing for non-uniform)
- `thomas::F`: Thomas factorization of tridiagonal matrix A (contains dl, du, inv_d)
- `bc_config::BC`: Boundary condition data (BCPair for derivative BC, PeriodicData for periodic)

# Notes
The Thomas factorization depends ONLY on x geometry and can be reused for:
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
- `bc=CubicFit()` (default): 4-point polynomial fit at endpoints
- `bc=ZeroCurvBC()`: Zero-curvature spline with z[1] = z[n+1] = 0
- `bc=PeriodicBC()`: Periodic spline with C2 continuity at boundaries
"""
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F,BC,S<:AbstractGridSpacing{T}}
    x::X
    spacing::S
    thomas::F
    bc_config::BC
end

# AbstractExtrap types are defined in eval_ops.jl (shared across all interpolants)

"""
    CubicInterpolant{Tg,Tv,C,E,P,BC}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `cubic_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type (Float32 or Float64) for x-coordinates
- `Tv`: Value type for y-values (can be Tg, Complex{Tg}, or other Number)
- `C`: CubicSplineCache type (preserves grid type info for O(1) vs O(log n) lookup)
- `E`: Extrapolation mode type (compile-time specialized)
- `P`: Search policy type (Binary, HintedBinary, LinearBinary, etc.)
- `BC`: Boundary condition type (BCPair or PeriodicBC)

# Fields
- `cache::C`: Pre-computed CubicSplineCache (LU factorization)
- `y::Vector{Tv}`: y-values (function values at grid points)
- `z::Vector{Tv}`: Pre-computed second derivative coefficients (solves system once!)
- `bc::BC`: Boundary condition used for this interpolant
- `extrap::E`: Extrapolation mode (compile-time specialized via type parameter)
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
itp = cubic_interp(x, y)
result = @. coef * itp(rho) * other_terms  # fused, zero-allocation per call
val = itp(0.5)                              # scalar (zero-allocation)

# Create with custom search policy
itp = cubic_interp(x, y; search=LinearBinary())
val = itp(0.5)                              # uses LinearBinary() by default
val = itp(0.5; search=Binary())             # override with Binary()

# Complex values
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im, 9.0+10.0im]
itp = cubic_interp(x, y)
val = itp(0.5)  # returns ComplexF64
```

# Performance Notes
- System solved ONCE at construction -> z coefficients pre-computed
- Each scalar call just evaluates cubic polynomial (zero-allocation!)
- Broadcast operations are perfectly fused (no intermediate arrays)
- Extrapolation mode uses type-parametrized dispatch for zero overhead
"""
struct CubicInterpolant{Tg<:AbstractFloat,Tv,C<:CubicSplineCache{Tg},E<:AbstractExtrap,P<:AbstractSearchPolicy,BC<:CubicBC} <: AbstractInterpolant{Tg, Tv}
    cache::C
    y::Vector{Tv}
    z::Vector{Tv}  # Pre-computed second derivative coefficients (value type)
    bc::BC  # Boundary condition used for this interpolant
    extrap::E  # Extrapolation mode (compile-time specialized via type parameter)
    search_policy::P  # Default search policy (immutable, thread-safe)
    function CubicInterpolant(
        cache::C,
        y::AbstractVector{Tv},
        z::AbstractVector{Tv},
        bc::BC,
        extrap::E,
        search::P=Binary()
    ) where {Tg<:AbstractFloat, Tv, C<:CubicSplineCache{Tg}, E<:AbstractExtrap, P<:AbstractSearchPolicy, BC<:CubicBC}
        @assert length(cache.x) == length(y) "cache grid and y must have same length"
        @assert length(cache.x) == length(z) "z coefficients must match grid length"
        # Always copy to ensure immutability: once constructed, the interpolant
        # owns its data and always returns identical results for the same query.
        # Without copying, external modifications to y or cache reuse could
        # silently corrupt results.
        new{Tg,Tv,C,E,P,BC}(cache, Vector{Tv}(y), Vector{Tv}(z), bc, extrap, search)
    end
end

# ========================================
# TransposeSnapshot Type (shared between multi-series interpolants)
# ========================================

"""
    TransposeSnapshot{Tv}

Immutable snapshot of point-contiguous (transposed) matrices.

Used for atomic swap in multi-series cubic interpolants to ensure thread-safe
lazy initialization of point-contiguous layout.

# Type Parameters
- `Tv`: Value type (can be Real or Complex)

# Fields
- `y_point::Union{Nothing, Matrix{Tv}}`: Point-contiguous y values (n_series × n_points)
- `z_point::Union{Nothing, Matrix{Tv}}`: Point-contiguous z values (n_series × n_points)

# Thread Safety
Used with atomic operations for lock-free lazy initialization.
Multiple threads may compute the transpose simultaneously (benign duplication).
"""
struct TransposeSnapshot{Tv}
    y_point::Union{Nothing, Matrix{Tv}}
    z_point::Union{Nothing, Matrix{Tv}}
end

# Empty snapshot constructor
TransposeSnapshot{Tv}() where {Tv} = TransposeSnapshot{Tv}(nothing, nothing)
