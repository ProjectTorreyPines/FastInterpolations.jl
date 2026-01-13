# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CUBIC UNIFIED INTERPOLANT TYPES                        ║
# ║      Adaptive layout type for optimal scalar/vector query performance     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Key difference from existing types:
# - CubicMultiInterpolant: Now uses unified-style matrix storage (n_points × n_series)
# - CubicMultiInterpolantFused: Interleaved matrix layout (n_series × n_points)
# - CubicMultiInterpolantUnified: Adaptive layout based on query pattern
#   - Vector queries use series-contiguous layout (n_points × n_series)
#   - Scalar queries use lazy point-contiguous layout (n_series × n_points)
#
# Include order: ... → cubic_fused_kernels.jl → cubic_unified_types.jl → ...
#
# Note: TransposeSnapshot is defined in cubic_types.jl (shared type)
#

# ========================================
# CubicMultiInterpolantUnified Type
# ========================================

"""
    CubicMultiInterpolantUnified{T, C, B} <: AbstractMultiInterpolant{T}

Unified multi-series cubic interpolant with adaptive memory layout.

Automatically uses optimal layout based on query pattern:
- **Vector queries**: Series-contiguous layout (y[:, k] is contiguous for series k)
- **Scalar queries**: Point-contiguous layout (y[k, :] is contiguous for all series at point)

The point-contiguous layout is lazily constructed on first scalar query.
For latency-sensitive applications, use `precompute_transpose=true` or call
`precompute_transpose!(mitp)` before hot loops.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `C`: Cache type (`CubicSplineCache{T}`)
- `B`: Boundary condition config type (BCPair or PeriodicData)

# Fields
- `cache::C`: Shared CubicSplineCache with LU factorization
- `bc_for_solve::B`: BC configuration (preserves derivative values for solving)
- `y::Matrix{T}`: Function values in (n_points × n_series) layout
- `z::Matrix{T}`: Second derivatives in (n_points × n_series) layout
- `_point_snapshot`: Atomic field for lazy point-contiguous layout
- `extrap::ExtrapVal`: Extrapolation mode (:none, :constant, :extension, :wrap)

# Memory Layout

**Primary storage (series-contiguous)**:
```
y[i, k] = y_k(x_i)  # i=point index, k=series index
z[i, k] = z_k(x_i)
```
`y[:, k]` is contiguous → optimal for vector queries

**Lazy storage (point-contiguous)**:
```
y_point[k, i] = y_k(x_i)  # transposed layout
z_point[k, i] = z_k(x_i)
```
`y_point[:, i]` is contiguous → optimal for SIMD scalar queries

# Thread Safety

The lazy point-contiguous layout uses atomic snapshot publish (RCU pattern):
- Multiple threads may compute the transpose simultaneously (benign duplication)
- All threads see a consistent pair after atomic store
- For large data, use `precompute_transpose=true` to avoid thundering herd

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
y1, y2 = sin.(2π .* x), cos.(2π .* x)

# Create unified interpolant (lazy by default)
mitp = cubic_interp_unified(x, [y1, y2])

# Vector query - uses series-contiguous layout (no transpose needed)
outputs = mitp([0.1, 0.5, 0.9])

# Scalar query - triggers point-contiguous layout creation on first call
vals = mitp(0.5)
```

See also: [`cubic_interp_unified`](@ref), [`precompute_transpose!`](@ref)
"""
mutable struct CubicMultiInterpolantUnified{
    T<:AbstractFloat,
    C<:CubicSplineCache{T},
    B
} <: AbstractMultiInterpolant{T}
    const cache::C                    # Shared cache with LU factorization
    const bc_for_solve::B             # BC config for solving
    const y::Matrix{T}                # Series-contiguous y (n_points × n_series)
    const z::Matrix{T}                # Series-contiguous z (n_points × n_series)
    @atomic _point_snapshot::TransposeSnapshot{T}  # Lazy point-contiguous layout
    const extrap::ExtrapVal           # Extrapolation mode

    function CubicMultiInterpolantUnified(
        cache::C,
        bc_for_solve::B,
        y::Matrix{T},
        z::Matrix{T},
        extrap::ExtrapVal
    ) where {T<:AbstractFloat, C<:CubicSplineCache{T}, B}
        new{T, C, B}(
            cache, bc_for_solve, y, z,
            TransposeSnapshot{T}(),
            extrap
        )
    end
end

# ========================================
# Helper Functions
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(mitp::CubicMultiInterpolantUnified) = mitp.extrap === Val(:wrap)

"""Number of series in the interpolant."""
@inline n_series(mitp::CubicMultiInterpolantUnified) = size(mitp.y, 2)

"""Number of grid points in the interpolant."""
@inline n_points(mitp::CubicMultiInterpolantUnified) = size(mitp.y, 1)

# ========================================
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(mitp::CubicMultiInterpolantUnified{T}) -> (y_point, z_point)

Ensure point-contiguous layout exists. Thread-safe via atomic snapshot.

RCU-style implementation:
- Fast path: atomic acquire read, return if populated
- Slow path: compute transpose, atomic release publish
- Duplicate compute is acceptable (benign); last publish wins

# Returns
Tuple of `(y_point::Matrix{T}, z_point::Matrix{T})` in (n_series × n_points) layout.

# Thread Safety
Uses atomic operations for lock-free read. Multiple threads may compute
the transpose simultaneously but all will converge to consistent state.

# Warning
For large data (>100MB), consider `precompute_transpose=true` at construction
or call `precompute_transpose!(mitp)` from a single thread before concurrent use
to avoid thundering herd memory allocation.
"""
@inline function _ensure_point_layout!(mitp::CubicMultiInterpolantUnified{T}) where T
    # Fast path: check if already populated
    snap = @atomic :acquire mitp._point_snapshot
    if snap.y_point !== nothing
        return (snap.y_point::Matrix{T}, snap.z_point::Matrix{T})
    end

    # Slow path: build point-contiguous layout
    # permutedims creates (n_series × n_points) from (n_points × n_series)
    y_point = permutedims(mitp.y)
    z_point = permutedims(mitp.z)
    new_snap = TransposeSnapshot{T}(y_point, z_point)

    # Atomic publish - readers always see consistent pair
    @atomic :release mitp._point_snapshot = new_snap

    return (y_point, z_point)
end

"""
    precompute_transpose!(mitp::CubicMultiInterpolantUnified) -> mitp

Pre-allocate point-contiguous matrices for scalar queries.

Call this before hot loops to avoid first-call scalar latency.
Returns the same interpolant for method chaining.

# Example
```julia
mitp = cubic_interp_unified(x, ys)
precompute_transpose!(mitp)

# Now scalar queries won't trigger allocation
for t in times
    vals = mitp(t)
end
```

# Thread Safety
Safe to call from any thread. If called concurrently, all threads
will complete successfully (benign duplication).
"""
function precompute_transpose!(mitp::CubicMultiInterpolantUnified)
    _ensure_point_layout!(mitp)
    return mitp
end
