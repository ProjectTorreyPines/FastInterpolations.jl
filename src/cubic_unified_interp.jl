# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CUBIC UNIFIED INTERPOLANT CONSTRUCTOR                  ║
# ║         Constructor API for CubicMultiInterpolantUnified                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides cubic_interp_unified(x, ys; bc, extrap, precompute_transpose) constructor
# for building adaptive layout multi-series interpolants.
#
# Include order: ... → cubic_unified_types.jl → cubic_unified_interp.jl → ...
#

# ========================================
# Internal: Coefficient Solver
# ========================================

"""
    _solve_unified_coefficients!(z_mat, y_mat, cache, bc_for_solve)

Solve cubic spline systems for all series using shared LU factorization.
Uses `@with_pool` for thread-safe workspace allocation.

Series-contiguous layout: y_mat and z_mat are (n_points × n_series).
Each column y_mat[:, k] is a series that gets solved into z_mat[:, k].

# Arguments
- `z_mat::Matrix{T}`: Output matrix (n_points × n_series) for second derivatives
- `y_mat::Matrix{T}`: Input matrix (n_points × n_series) of function values
- `cache::CubicSplineCache`: Pre-built cache with LU factorization
- `bc_for_solve`: BC configuration to pass to _solve_system!

# Thread-Safety
Pool-allocated workspace is used per-call, making this function thread-safe.
"""
@with_pool pool function _solve_unified_coefficients!(
    z_mat::Matrix{T},
    y_mat::Matrix{T},
    cache::CubicSplineCache{T},
    bc_for_solve
) where {T<:AbstractFloat}
    n_series = size(y_mat, 2)

    # Solve each series column
    @inbounds for k in 1:n_series
        _solve_system!(@view(z_mat[:, k]), cache, @view(y_mat[:, k]), bc_for_solve)
    end

    return z_mat
end

# ========================================
# Public Constructor: Vector{Vector}
# ========================================

"""
    cubic_interp_unified(x, ys; bc=NaturalBC(), extrap=:none, precompute_transpose=false)

Create a unified multi-series cubic interpolant with adaptive memory layout.

# Arguments
- `x::AbstractVector`: Grid points (sorted, length ≥ 2)
- `ys::AbstractVector{<:AbstractVector}`: Vector of y-value vectors (all same length as x)
- `bc::AbstractBC`: Boundary condition (NaturalBC, ClampedBC, PeriodicBC)
- `extrap::Symbol`: Extrapolation mode (:none, :constant, :extension, :wrap)
- `precompute_transpose::Bool`: If true, build point-contiguous layout immediately
  (useful for latency-sensitive scalar queries; default false for memory efficiency)

# Returns
`CubicMultiInterpolantUnified` object with series-contiguous primary storage.

# Memory Layout
Data is stored in (n_points × n_series) matrices for cache-friendly column access:
- `y[i, k]` = value of series k at grid point i
- Vector queries use this layout directly
- Scalar queries use a lazily-created transpose

# Performance
- **Vector queries**: Optimal immediately (series-contiguous layout)
- **Scalar queries**: First call creates point-contiguous layout, subsequent calls reuse it

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

# Lazy (default): transpose created on first scalar query
mitp = cubic_interp_unified(x, [y1, y2, y3])

# Eager: transpose created at construction (for latency-sensitive scalar use)
mitp = cubic_interp_unified(x, [y1, y2, y3]; precompute_transpose=true)
```

See also: [`CubicMultiInterpolantUnified`](@ref), [`precompute_transpose!`](@ref)
"""
function cubic_interp_unified(
    x::AbstractVector{T},
    ys::AbstractVector{<:AbstractVector{T}};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    precompute_transpose::Bool=false
) where {T<:AbstractFloat}
    # Validation
    @assert !isempty(ys) "ys must not be empty"

    n_points = length(x)
    n_series_count = length(ys)

    # Validate all y-series have same length as x
    for (k, y) in enumerate(ys)
        if length(y) != n_points
            throw(DimensionMismatch(
                "y-series $k has length $(length(y)), expected $n_points (length of x)"
            ))
        end
    end

    # Build y matrix (n_points × n_series) series-contiguous
    y_mat = Matrix{T}(undef, n_points, n_series_count)
    @inbounds for k in 1:n_series_count
        y_mat[:, k] .= ys[k]
    end

    # Handle periodic BC separately
    if _is_periodic_bc(bc)
        return _build_unified_periodic(x, y_mat, n_points, n_series_count, precompute_transpose)
    end

    # Get cache for derivative BC
    bc_pair = _normalize_bc(bc, T)
    cache = _get_cubic_cache(x, bc_pair)

    # Build z matrix by solving systems
    z_mat = Matrix{T}(undef, n_points, n_series_count)
    _solve_unified_coefficients!(z_mat, y_mat, cache, bc_pair)

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    mitp = CubicMultiInterpolantUnified(cache, bc_pair, y_mat, z_mat, extrap_val)

    # Optionally precompute transpose for deterministic latency
    if precompute_transpose
        _ensure_point_layout!(mitp)
    end

    return mitp
end

"""
    _build_unified_periodic(x, y_mat, n_points, n_series, precompute_transpose)

Internal helper to build periodic BC unified interpolant.
"""
function _build_unified_periodic(
    x::AbstractVector{T},
    y_mat::Matrix{T},
    n_points::Int,
    n_series_count::Int,
    precompute_transpose::Bool
) where {T<:AbstractFloat}
    # Validate periodic endpoints for all series
    atol = T === Float32 ? _PERIODIC_ATOL_F32 : _PERIODIC_ATOL_F64
    @inbounds for k in 1:n_series_count
        y_first = y_mat[1, k]
        y_last = y_mat[n_points, k]
        if !isapprox(y_first, y_last; atol=atol)
            throw(ArgumentError(
                "Periodic BC requires y[1] ≈ y[end] for series $k, " *
                "got y[1]=$y_first, y[end]=$y_last (diff=$(abs(y_last-y_first)))"
            ))
        end
    end

    # Get periodic cache
    cache = _get_cubic_cache(x, PeriodicBC())

    # Build z matrix
    z_mat = Matrix{T}(undef, n_points, n_series_count)
    _solve_unified_coefficients!(z_mat, y_mat, cache, cache.bc_config)

    # Periodic BC always uses :wrap extrapolation
    mitp = CubicMultiInterpolantUnified(cache, cache.bc_config, y_mat, z_mat, Val(:wrap))

    if precompute_transpose
        _ensure_point_layout!(mitp)
    end

    return mitp
end

# ========================================
# Public Constructor: Matrix
# ========================================

"""
    cubic_interp_unified(x, Y::AbstractMatrix; bc=NaturalBC(), extrap=:none, precompute_transpose=false)

Create a unified multi-series cubic interpolant from matrix input.

Each **column** of Y is treated as a y-series. The input layout matches
the internal series-contiguous storage, avoiding unnecessary transpose
when data is already in matrix form.

# Arguments
- `x::AbstractVector`: Grid points (sorted, length ≥ 2)
- `Y::AbstractMatrix`: Data matrix (n_points × n_series), each column is a y-series
- `bc::AbstractBC`: Boundary condition (NaturalBC, ClampedBC, PeriodicBC)
- `extrap::Symbol`: Extrapolation mode (:none, :constant, :extension, :wrap)
- `precompute_transpose::Bool`: If true, build point-contiguous layout immediately

# Example
```julia
x = collect(range(0.0, 1.0, 101))
Y = hcat(sin.(2π .* x), cos.(2π .* x), exp.(-x))  # 101 × 3 matrix

mitp = cubic_interp_unified(x, Y)
vals = mitp(0.5)  # Returns 3-element vector
```

See also: [`cubic_interp_unified`](@ref), [`CubicMultiInterpolantUnified`](@ref)
"""
function cubic_interp_unified(
    x::AbstractVector{T},
    Y::AbstractMatrix{T};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    precompute_transpose::Bool=false
) where {T<:AbstractFloat}
    n_points = length(x)

    # Validate dimensions
    if size(Y, 1) != n_points
        throw(DimensionMismatch(
            "Y has $(size(Y, 1)) rows but x has $n_points points (expected n_points × n_series matrix)"
        ))
    end

    n_series_count = size(Y, 2)

    # Copy to ensure ownership (for immutability)
    y_mat = copy(Y)

    # Handle periodic BC separately
    if _is_periodic_bc(bc)
        return _build_unified_periodic(x, y_mat, n_points, n_series_count, precompute_transpose)
    end

    # Get cache for derivative BC
    bc_pair = _normalize_bc(bc, T)
    cache = _get_cubic_cache(x, bc_pair)

    # Build z matrix by solving systems
    z_mat = Matrix{T}(undef, n_points, n_series_count)
    _solve_unified_coefficients!(z_mat, y_mat, cache, bc_pair)

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    mitp = CubicMultiInterpolantUnified(cache, bc_pair, y_mat, z_mat, extrap_val)

    if precompute_transpose
        _ensure_point_layout!(mitp)
    end

    return mitp
end

# ========================================
# Real Type Wrappers
# ========================================

"""
Real type wrapper for Vector{Vector} - auto-promotes Int, Rational, etc. to Float.
"""
function cubic_interp_unified(
    x::AbstractVector{Tx},
    ys::AbstractVector{<:AbstractVector{Ty}};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    precompute_transpose::Bool=false
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    ys_float = [T.(y) for y in ys]
    return cubic_interp_unified(x_float, ys_float; bc=bc, extrap=extrap, precompute_transpose=precompute_transpose)
end

"""
Real type wrapper for Matrix - auto-promotes Int, Rational, etc. to Float.
"""
function cubic_interp_unified(
    x::AbstractVector{Tx},
    Y::AbstractMatrix{Ty};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    precompute_transpose::Bool=false
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    Y_float = T.(Y)
    return cubic_interp_unified(x_float, Y_float; bc=bc, extrap=extrap, precompute_transpose=precompute_transpose)
end

# ========================================
# Conversion from CubicMultiInterpolantFused
# ========================================

"""
    CubicMultiInterpolantUnified(fused::CubicMultiInterpolantFused)

Convert a `CubicMultiInterpolantFused` to a `CubicMultiInterpolantUnified`.

This extracts the coefficients and transposes them from point-contiguous
(n_series × n_points) to series-contiguous (n_points × n_series) layout.

# Example
```julia
mitp_fused = cubic_interp_fused(x, [y1, y2, y3])
mitp_unified = CubicMultiInterpolantUnified(mitp_fused)

# Both produce identical results
@assert mitp_unified(0.5) ≈ mitp_fused(0.5)
```
"""
function CubicMultiInterpolantUnified(fused::CubicMultiInterpolantFused{T}) where {T<:AbstractFloat}
    # Fused uses (n_series × n_points), unified uses (n_points × n_series)
    # Transpose to get series-contiguous layout
    y_mat = permutedims(fused.y)  # (n_points × n_series)
    z_mat = permutedims(fused.z)  # (n_points × n_series)

    # Get cache from the fused interpolant's grid
    cache = _get_cubic_cache(fused.x, fused.bc_config)

    return CubicMultiInterpolantUnified(cache, fused.bc_config, y_mat, z_mat, fused.extrap)
end
