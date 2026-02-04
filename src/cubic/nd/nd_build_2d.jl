# ========================================
# 2D Bicubic Coefficient Construction (Temporary)
# ========================================
#
# Functions to compute precomputed partial derivatives for 2D bicubic interpolation.
# Uses existing 1D cubic spline infrastructure (CubicSplineCache, solvers).
#
# This file is TEMPORARY and will be deprecated once generic ND is validated.
#
# Optimizations:
# - @with_pool for zero-allocation task-local memory
# - Batch SIMD solver for axis 2 (rows)
# - Contiguous memory access patterns
#
# Partials layout (4, nx, ny):
#   partials[1, i, j] = f(xᵢ, yⱼ)
#   partials[2, i, j] = ∂f/∂x at (xᵢ, yⱼ)
#   partials[3, i, j] = ∂f/∂y at (xᵢ, yⱼ)
#   partials[4, i, j] = ∂²f/∂x∂y at (xᵢ, yⱼ)

"""
    _build_bicubic_coeffs(x, y, data, bc_x, bc_y) -> NodalDerivatives2D{Tv}

Compute all partial derivatives for bicubic interpolation.

# Arguments
- `x::AbstractVector{Tg}`: X-axis grid points
- `y::AbstractVector{Tg}`: Y-axis grid points
- `data::AbstractMatrix{Tv}`: Function values at grid points (nx × ny)
- `bc_x::AbstractBC`: Boundary condition for x-axis
- `bc_y::AbstractBC`: Boundary condition for y-axis

# Returns
- `NodalDerivatives2D{Tv}` containing the partials array

# Algorithm
1. Copy f values to partials[1,:,:]
2. Compute ∂f/∂y (along dim 2) - uses batch SIMD for NaturalBC
3. Compute ∂f/∂x (along dim 1) - per-column loop
4. Compute ∂²f/∂x∂y with data-derived edge BCs - batch SIMD via Val(2)
"""
@with_pool pool function _build_bicubic_coeffs(
    x::AbstractVector{Tg},
    y::AbstractVector{Tg},
    data::AbstractMatrix{Tv},
    bc_x::AbstractBC,
    bc_y::AbstractBC
) where {Tg<:AbstractFloat, Tv}
    nx, ny = length(x), length(y)
    is_periodic_x = _is_periodic_bc(bc_x)
    is_periodic_y = _is_periodic_bc(bc_y)
    polyfit_deg_x = get_polyfit_degree(bc_x)
    polyfit_deg_y = get_polyfit_degree(bc_y)

    # Validate periodic data
    if is_periodic_x
        _check_periodic_data_x(data, nx, ny, Tv)
    end
    if is_periodic_y
        _check_periodic_data_y(data, nx, ny, Tv)
    end

    # Validate minimum points for PolyFit BCs
    if polyfit_deg_x > 0 && nx < polyfit_deg_x + 1
        throw(ArgumentError("PolyFit BC on x requires at least $(polyfit_deg_x + 1) points, got $nx"))
    end
    if polyfit_deg_y > 0 && ny < polyfit_deg_y + 1
        throw(ArgumentError("PolyFit BC on y requires at least $(polyfit_deg_y + 1) points, got $ny"))
    end

    # Allocate partials array
    partials = Array{Tv, 3}(undef, 4, nx, ny)

    # Step 1: Copy f values
    @inbounds for j in 1:ny, i in 1:nx
        partials[1, i, j] = data[i, j]
    end

    # Step 2: Compute ∂f/∂y (differentiate along dim 2) - uses batch SIMD
    if polyfit_deg_y > 0
        _partial_deriv_dim!(view(partials, 3, :, :), data, y, CubicFit(), Val(2))
    else
        _partial_deriv_dim!(view(partials, 3, :, :), data, y, bc_y, Val(2))
    end

    # Step 3: Compute ∂f/∂x (differentiate along dim 1) - per-column loop
    if polyfit_deg_x > 0
        _partial_deriv_dim!(view(partials, 2, :, :), data, x, CubicFit(), Val(1))
    else
        _partial_deriv_dim!(view(partials, 2, :, :), data, x, bc_x, Val(1))
    end

    # Step 4: Compute ∂²f/∂x∂y with data-derived edge BCs (batch SIMD via Val(2))
    # Key insight: Use fx_view and differentiate along y (Val(2)) for SIMD benefits.
    fx_view = view(partials, 2, :, :)
    if is_periodic_y
        _partial_deriv_dim!(view(partials, 4, :, :), fx_view, y, PeriodicBC(), Val(2))
    else
        # Extract ∂f/∂y edges (contiguous row access - enables @simd!)
        fy_top = acquire!(pool, Tv, nx)
        fy_bottom = acquire!(pool, Tv, nx)
        @inbounds @simd for i in 1:nx
            fy_top[i] = partials[3, i, 1]       # top row (y=y[1])
            fy_bottom[i] = partials[3, i, ny]   # bottom row (y=y[ny])
        end

        # Compute ∂²f/∂x∂y at edges by differentiating ∂f/∂y along x
        fxy_top = acquire!(pool, Tv, nx)
        fxy_bottom = acquire!(pool, Tv, nx)
        if is_periodic_x
            _deriv_1d!(fxy_top, fy_top, x, PeriodicBC())
            _deriv_1d!(fxy_bottom, fy_bottom, x, PeriodicBC())
        elseif nx >= 4
            _deriv_1d!(fxy_top, fy_top, x, CubicFit())
            _deriv_1d!(fxy_bottom, fy_bottom, x, CubicFit())
        else
            _deriv_1d!(fxy_top, fy_top, x, NaturalBC())
            _deriv_1d!(fxy_bottom, fy_bottom, x, NaturalBC())
        end

        _partial_deriv_dim!(view(partials, 4, :, :), fx_view, y, (fxy_top, fxy_bottom), Val(2))
    end

    return NodalDerivatives2D{Tv}(partials)
end

# ========================================
# 1D Differentiation Helpers
# ========================================
# Unified API: _deriv_1d!(pool, deriv, values, grid, bc)
# BC dispatch: AbstractBC or CubicFit

"""
    _deriv_1d!(pool, deriv, values, grid, bc)

Differentiate 1D vector using cubic splines. BC type determines the method:
- `AbstractBC` (NaturalBC, ClampedBC, PeriodicBC, etc.): Use specified BC
- `CubicFit`: Estimate endpoint derivatives via polynomial fitting
"""
@with_pool pool function _deriv_1d!(
    deriv::AbstractVector{Tv}, values::AbstractVector{Tv},
    grid::AbstractVector{Tg}, bc::AbstractB
) where {Tg<:AbstractFloat, Tv}
    n = length(values)
    # Cache uses grid type Tg for matrix structure (factorization)
    # Computation uses value type Tv for actual BC values
    bc_cache = _is_periodic_bc(bc) ? PeriodicBC() : _normalize_bc(bc, Tg)
    bc_compute = _is_periodic_bc(bc) ? PeriodicBC() : _normalize_bc(bc, Tv)
    cache = _get_cubic_cache(grid, bc_cache, true)
    actual_bc = cache.bc_config isa PeriodicData ? cache.bc_config : bc_compute
    m = acquire!(pool, Tv, n)
    _solve_system!(m, cache, values, actual_bc)
    _moments_to_derivatives_1d!(deriv, m, values, cache.spacing)
    _apply_derivative_bc!(deriv, actual_bc)
    return deriv
end

@with_pool pool function _deriv_1d!(
    deriv::AbstractVector{Tv}, values::AbstractVector{Tv},
    grid::AbstractVector{Tg}, ::CubicFit
) where {Tg<:AbstractFloat, Tv}
    n = length(values)
    @assert n >= 4 "Need at least 4 points for CubicFit"

    deriv_left = _estimate_endpoint_derivative(grid, values, Val(:left), CubicFit())
    deriv_right = _estimate_endpoint_derivative(grid, values, Val(:right), CubicFit())

    # BC values are Tv type (can be Complex)
    bc = BCPair(Deriv1(Tv(deriv_left)), Deriv1(Tv(deriv_right)))
    # Cache uses grid type Tg for matrix structure
    bc_cache = BCPair(Deriv1(zero(Tg)), Deriv1(zero(Tg)))
    cache = _get_cubic_cache(grid, bc_cache, true)
    m = acquire!(pool, Tv, n)
    _solve_system!(m, cache, values, bc)
    _moments_to_derivatives_1d!(deriv, m, values, cache.spacing)
    _apply_derivative_bc!(deriv, bc)
    return deriv
end

# ========================================
# Partial Derivatives (2D)
# ========================================
# Unified API: _partial_deriv_dim!(pool, out, data, grid, bc, Val(D))
# BC dispatch: AbstractBC, CubicFit, or Tuple{Vec,Vec} for per-slice edge BC

"""
    _partial_deriv_dim!(pool, out, data, grid, bc, ::Val{D})

Compute partial derivative of 2D data along dimension D. BC type determines the method:
- `AbstractBC` (NaturalBC, ClampedBC, PeriodicBC, etc.): Cubic spline differentiation
- `CubicFit`: Polynomial fitting at boundaries
- `Tuple{Vector, Vector}`: Per-slice edge BC values (for cross-derivatives)

# Dimensions
- `Val(1)`: Differentiate along dim 1 (x-direction, columns)
- `Val(2)`: Differentiate along dim 2 (y-direction, rows) - SIMD optimized
"""
@with_pool pool function _partial_deriv_dim!(
    out::AbstractMatrix{Tv}, data::AbstractMatrix{Tv},
    grid::AbstractVector{Tg}, bc::AbstractBC,
    ::Val{2}
) where {Tg<:AbstractFloat, Tv}
    nx, ny = size(data)
    # Cache uses grid type Tg for matrix structure (factorization)
    # Computation uses value type Tv for actual BC values
    bc_cache = _is_periodic_bc(bc) ? PeriodicBC() : _normalize_bc(bc, Tg)
    bc_compute = _is_periodic_bc(bc) ? PeriodicBC() : _normalize_bc(bc, Tv)
    cache = _get_cubic_cache(grid, bc_cache, true)
    actual_bc = cache.bc_config isa PeriodicData ? cache.bc_config : bc_compute

    # Periodic BC requires different solve path (Sherman-Morrison)
    if actual_bc isa PeriodicData
        # Fall back to per-row solve for periodic
        m = acquire!(pool, Tv, ny)
        line = acquire!(pool, Tv, ny)
        dline = acquire!(pool, Tv, ny)
        @inbounds for i in 1:nx
            for j in 1:ny; line[j] = data[i, j]; end
            _solve_system!(m, cache, line, actual_bc)
            _moments_to_derivatives_1d!(dline, m, line, cache.spacing)
            _apply_derivative_bc!(dline, actual_bc)
            for j in 1:ny; out[i, j] = dline[j]; end
        end
    else
        # Batch solve along axis 2 (SIMD optimized)
        M = acquire!(pool, Tv, (nx, ny))
        solve_along_dim!(M, cache, data, actual_bc, Val(2))
        moments_to_derivatives_along_dim!(out, M, data, cache.spacing, actual_bc, Val(2))
    end
    return out
end

@with_pool pool function _partial_deriv_dim!(
    out::AbstractMatrix{Tv}, data::AbstractMatrix{Tv},
    grid::AbstractVector{Tg}, bc::AbstractBC,
    ::Val{1}
) where {Tg<:AbstractFloat, Tv}
    # Note: Benchmarking showed that for dim 1 (columns), per-column solving
    # is faster than batch solving due to view creation overhead in RHS.
    nx, ny = size(data)
    # Cache uses grid type Tg for matrix structure (factorization)
    # Computation uses value type Tv for actual BC values
    bc_cache = _is_periodic_bc(bc) ? PeriodicBC() : _normalize_bc(bc, Tg)
    bc_compute = _is_periodic_bc(bc) ? PeriodicBC() : _normalize_bc(bc, Tv)
    cache = _get_cubic_cache(grid, bc_cache, true)
    actual_bc = cache.bc_config isa PeriodicData ? cache.bc_config : bc_compute
    m = acquire!(pool, Tv, nx)
    line = acquire!(pool, Tv, nx)
    dline = acquire!(pool, Tv, nx)

    @inbounds for j in 1:ny
        for i in 1:nx; line[i] = data[i, j]; end
        _solve_system!(m, cache, line, actual_bc)
        _moments_to_derivatives_1d!(dline, m, line, cache.spacing)
        _apply_derivative_bc!(dline, actual_bc)
        for i in 1:nx; out[i, j] = dline[i]; end
    end
    return out
end

# CubicFit dispatch: Use polynomial fitting at boundaries
@with_pool pool function _partial_deriv_dim!(
    out::AbstractMatrix{Tv}, data::AbstractMatrix{Tv}, grid::AbstractVector{Tg},
    ::CubicFit, ::Val{2}
) where {Tg<:AbstractFloat, Tv}
    nx, ny = size(data)
    line = acquire!(pool, Tv, ny)
    dline = acquire!(pool, Tv, ny)

    @inbounds for i in 1:nx
        for j in 1:ny; line[j] = data[i, j]; end
        _deriv_1d!(dline, line, grid, CubicFit())
        for j in 1:ny; out[i, j] = dline[j]; end
    end
    return out
end

@with_pool pool function _partial_deriv_dim!(
    out::AbstractMatrix{Tv}, data::AbstractMatrix{Tv}, grid::AbstractVector{Tg},
    ::CubicFit, ::Val{1}
) where {Tg<:AbstractFloat, Tv}
    nx, ny = size(data)
    line = acquire!(pool, Tv, nx)
    dline = acquire!(pool, Tv, nx)

    @inbounds for j in 1:ny
        for i in 1:nx; line[i] = data[i, j]; end
        _deriv_1d!(dline, line, grid, CubicFit())
        for i in 1:nx; out[i, j] = dline[i]; end
    end
    return out
end

# Tuple dispatch: Per-slice edge BC values for cross-derivatives
# Val(1): (left, right) BC for each column - per-column loop (no batch benefit)
@with_pool pool function _partial_deriv_dim!(
    out::AbstractMatrix{Tv}, data::AbstractMatrix{Tv}, grid::AbstractVector{Tg},
    edge_bc::Tuple{AbstractVector{Tv}, AbstractVector{Tv}},
    ::Val{1}
) where {Tg<:AbstractFloat, Tv}
    bc_left, bc_right = edge_bc
    nx, ny = size(data)
    # Cache uses grid type Tg for matrix structure
    canonical_bc = BCPair(Deriv1(zero(Tg)), Deriv1(zero(Tg)))
    cache = _get_cubic_cache(grid, canonical_bc, true)
    m = acquire!(pool, Tv, nx)
    line = acquire!(pool, Tv, nx)
    dline = acquire!(pool, Tv, nx)

    @inbounds for j in 1:ny
        line_bc = BCPair(Deriv1(bc_left[j]), Deriv1(bc_right[j]))
        for i in 1:nx; line[i] = data[i, j]; end
        _solve_system!(m, cache, line, line_bc)
        _moments_to_derivatives_1d!(dline, m, line, cache.spacing)
        _apply_derivative_bc!(dline, line_bc)
        for i in 1:nx; out[i, j] = dline[i]; end
    end
    return out
end

# Val(2): (top, bottom) BC for each row - BATCH SIMD optimized!
# Key insight: Thomas factorization is the same for all rows, only RHS differs.
@with_pool pool function _partial_deriv_dim!(
    out::AbstractMatrix{Tv}, data::AbstractMatrix{Tv}, grid::AbstractVector{Tg},
    edge_bc::Tuple{AbstractVector{Tv}, AbstractVector{Tv}},
    ::Val{2}
) where {Tg<:AbstractFloat, Tv}
    nx, ny = size(data)

    # Get cache with canonical BC using grid type (Deriv1 structure - actual values in edge_bc)
    # Cache uses Tg type for matrix structure
    canonical_bc = BCPair(Deriv1(zero(Tg)), Deriv1(zero(Tg)))
    cache = _get_cubic_cache(grid, canonical_bc, true)

    # Workspace for moments
    M = acquire!(pool, Tv, (nx, ny))

    # Step 1: Compute RHS with per-row edge BC (SIMD optimized)
    compute_rhs_along_dim!(M, data, cache.spacing, edge_bc, Val(2))

    # Step 2: Batch solve (SIMD optimized) - same Thomas factorization for all rows
    _ldiv_along_dim!(M, cache.thomas, Val(2))

    # Step 3: Convert moments to derivatives with edge BC (SIMD, no double-write)
    moments_to_derivatives_along_dim!(out, M, data, cache.spacing, edge_bc, Val(2))

    return out
end

# ========================================
# Periodic Data Validation
# ========================================

function _check_periodic_data_x(data::AbstractMatrix{Tv}, nx::Int, ny::Int, ::Type{Tv}) where {Tv}
    atol = Tv <: Complex ? _PERIODIC_ATOL_F64 : (real(Tv) === Float32 ? _PERIODIC_ATOL_F32 : _PERIODIC_ATOL_F64)
    @inbounds for j in 1:ny
        d1, dn = data[1, j], data[nx, j]
        if !isapprox(d1, dn; atol=atol)
            throw(ArgumentError(
                "Periodic BC on x requires data[1,j] ≈ data[end,j], but at j=$j: diff=$(abs(dn-d1))"
            ))
        end
    end
end

function _check_periodic_data_y(data::AbstractMatrix{Tv}, nx::Int, ny::Int, ::Type{Tv}) where {Tv}
    atol = Tv <: Complex ? _PERIODIC_ATOL_F64 : (real(Tv) === Float32 ? _PERIODIC_ATOL_F32 : _PERIODIC_ATOL_F64)
    @inbounds for i in 1:nx
        d1, dn = data[i, 1], data[i, ny]
        if !isapprox(d1, dn; atol=atol)
            throw(ArgumentError(
                "Periodic BC on y requires data[i,1] ≈ data[i,end], but at i=$i: diff=$(abs(dn-d1))"
            ))
        end
    end
end
