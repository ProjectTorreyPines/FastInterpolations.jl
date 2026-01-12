# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CUBIC FUSED INTERPOLANT CONSTRUCTOR                     ║
# ║         Constructor API for CubicMultiInterpolantFused                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides cubic_interp_fused(x, ys; bc, extrap) constructor
# for building high-performance fused multi-series interpolants.
#
# Include order: ... → cubic_fused_types.jl → cubic_fused_interp.jl → ...
#

# ========================================
# Internal: Coefficient Solver
# ========================================

"""
    _solve_fused_coefficients!(z_mat, y_mat, cache, bc_for_solve)

Solve cubic spline systems for all series in parallel.
Uses `@with_pool` for thread-safe workspace allocation.

# Arguments
- `z_mat::Matrix{T}`: Output matrix [n_series × n_points] for second derivatives
- `y_mat::Matrix{T}`: Input matrix [n_series × n_points] of function values
- `cache::CubicSplineCache`: Pre-built cache with LU factorization
- `bc_for_solve`: BC configuration to pass to _solve_system!

# Thread-Safety
Pool-allocated workspace is used per-call, making this function thread-safe.
"""
@with_pool pool function _solve_fused_coefficients!(
    z_mat::Matrix{T},
    y_mat::Matrix{T},
    cache::CubicSplineCache{T},
    bc_for_solve
) where {T<:AbstractFloat}
    n_series, n_points = size(y_mat)

    # Temporary vectors from pool
    y_tmp = acquire!(pool, T, n_points)
    z_tmp = acquire!(pool, T, n_points)

    @inbounds for k in 1:n_series
        # Copy row k to temporary vector
        for j in 1:n_points
            y_tmp[j] = y_mat[k, j]
        end

        # Solve for z coefficients
        _solve_system!(z_tmp, cache, y_tmp, bc_for_solve)

        # Copy result back to matrix row k
        for j in 1:n_points
            z_mat[k, j] = z_tmp[j]
        end
    end

    return z_mat
end

# ========================================
# Public Constructor: Vector{Vector}
# ========================================

"""
    cubic_interp_fused(x, ys; bc=NaturalBC(), extrap=:none)

Create a high-performance fused multi-series cubic interpolant.

# Arguments
- `x::AbstractVector`: Grid points (sorted, length ≥ 2)
- `ys::AbstractVector{<:AbstractVector}`: Vector of y-value vectors (all same length as x)
- `bc::AbstractBC`: Boundary condition (NaturalBC, ClampedBC, PeriodicBC)
- `extrap::Symbol`: Extrapolation mode (:none, :constant, :extension, :wrap)

# Returns
`CubicMultiInterpolantFused` object with interleaved memory layout.

# Memory Layout
Data is stored in [n_series × n_points] matrices for cache-friendly column access:
- `y[k, i]` = value of series k at grid point i
- `z[k, i]` = second derivative of series k at grid point i

# Performance
For m ≥ 40,000 series, provides 15-20× speedup over `CubicMultiInterpolant`
due to eliminated cache thrashing.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

mitp = cubic_interp_fused(x, [y1, y2, y3])
vals = mitp(0.5)  # Returns [sin(π), cos(π), exp(-0.5)]
```

See also: [`CubicMultiInterpolantFused`](@ref), [`cubic_interp`](@ref)
"""
function cubic_interp_fused(
    x::AbstractVector{T},
    ys::AbstractVector{<:AbstractVector{T}};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    # Validation
    isempty(ys) && throw(ArgumentError("ys must not be empty"))

    n_points = length(x)
    n_series = length(ys)

    # Validate all y-series have same length as x
    for (k, y) in enumerate(ys)
        if length(y) != n_points
            throw(DimensionMismatch(
                "y-series $k has length $(length(y)), expected $n_points (length of x)"
            ))
        end
    end

    # Build y matrix [n_series × n_points]
    y_mat = Matrix{T}(undef, n_series, n_points)
    @inbounds for k in 1:n_series
        for j in 1:n_points
            y_mat[k, j] = ys[k][j]
        end
    end

    # Handle periodic BC separately
    if _is_periodic_bc(bc)
        return _build_fused_periodic(x, y_mat, n_series, n_points)
    end

    # Get cache for derivative BC
    bc_pair = _normalize_bc(bc, T)
    cache = _get_cubic_cache(x, bc_pair)

    # Build z matrix by solving systems
    z_mat = Matrix{T}(undef, n_series, n_points)
    _solve_fused_coefficients!(z_mat, y_mat, cache, bc_pair)

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    return CubicMultiInterpolantFused(
        cache.x, cache.spacing, y_mat, z_mat, bc_pair, extrap_val, n_series, n_points
    )
end

"""
    _build_fused_periodic(x, y_mat, n_series, n_points)

Internal helper to build periodic BC fused interpolant.
"""
@with_pool pool function _build_fused_periodic(
    x::AbstractVector{T},
    y_mat::Matrix{T},
    n_series::Int,
    n_points::Int
) where {T<:AbstractFloat}
    # Validate periodic endpoints for all series
    @inbounds for k in 1:n_series
        y_first = y_mat[k, 1]
        y_last = y_mat[k, n_points]
        if !isapprox(y_first, y_last; rtol=sqrt(eps(T)))
            throw(ArgumentError(
                "Periodic BC requires y[1] ≈ y[end] for series $k, " *
                "got y[1]=$y_first, y[end]=$y_last"
            ))
        end
    end

    # Get periodic cache
    cache = _get_cubic_cache(x, PeriodicBC())

    # Build z matrix
    z_mat = Matrix{T}(undef, n_series, n_points)
    _solve_fused_coefficients!(z_mat, y_mat, cache, cache.bc_config)

    # Periodic BC always uses :wrap extrapolation
    return CubicMultiInterpolantFused(
        cache.x, cache.spacing, y_mat, z_mat, cache.bc_config, Val(:wrap), n_series, n_points
    )
end

# ========================================
# Real Type Wrapper
# ========================================

"""
Real type wrapper - auto-promotes Int, Rational, etc. to Float.
"""
function cubic_interp_fused(
    x::AbstractVector{Tx},
    ys::AbstractVector{<:AbstractVector{Ty}};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    ys_float = [T.(y) for y in ys]
    return cubic_interp_fused(x_float, ys_float; bc=bc, extrap=extrap)
end
