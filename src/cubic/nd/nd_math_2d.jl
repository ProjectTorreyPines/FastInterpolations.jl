# ========================================
# 2D Batch Solvers (Temporary)
# ========================================
#
# High-performance batch solvers optimized for 2D cubic interpolation.
# Key optimization: Loop transposition for cache-friendly SIMD access.
#
# These are 2D-specific and will be superseded by generic ND batch solvers.
#
# When solving along axis D (D > 1), we transpose the LOOP ORDER (not data):
# - Outer loop: system step (k = 1:n_sys)
# - Inner loop: axis 1 (i = 1:n_batch) - CONTIGUOUS in column-major!
#
# This enables @simd vectorization over the contiguous dimension.
#
# TYPE DESIGN (Tg/Tv separation):
# - Tg: Grid type (AbstractFloat) - coordinates, spacing, factorization
# - Tv: Value type - can be Real, Complex, or AD types (ForwardDiff.Dual)
# - ThomasFactorization uses Tg, RHS/solution matrices use Tv

# ========================================
# BATCH THOMAS SOLVERS
# ========================================

"""
    _ldiv_along_dim_vectorized!(z, thomas)

Batch Thomas solver for systems along axis 2 (rows).
KEY OPTIMIZATION: Outer loop = system step, inner loop = axis 1 (contiguous).
This enables @simd vectorization over the contiguous dimension.

# Type Parameters
- `Tv`: Value type (Real, Complex, or AD type)
- `Tg`: Grid type (AbstractFloat) for ThomasFactorization

# Arguments
- `z::AbstractMatrix{Tv}`: RHS matrix (modified in-place), systems along axis 2
- `thomas::ThomasFactorization{Tg}`: Thomas factorization with dl, du, inv_d
"""
@inline function _ldiv_along_dim_vectorized!(
    z::AbstractMatrix{Tv},
    thomas::ThomasFactorization{Tg,V}
) where {Tv, Tg<:AbstractFloat, V<:AbstractVector{Tg}}
    dl = thomas.dl
    du = thomas.du
    inv_d = thomas.inv_d
    n_sys = length(inv_d)   # System size (axis 2 length)
    n_batch = size(z, 1)    # Batch size (axis 1 length, contiguous!)

    # Forward substitution: transposed loop order for contiguous access
    @inbounds for k in 2:n_sys
        factor = Tv(-dl[k-1])
        @simd for i in 1:n_batch
            z[i, k] = muladd(factor, z[i, k-1], z[i, k])
        end
    end

    # Backward substitution: final column
    inv_d_n = Tv(inv_d[n_sys])
    @inbounds @simd for i in 1:n_batch
        z[i, n_sys] *= inv_d_n
    end

    # Backward substitution: remaining columns
    @inbounds for k in (n_sys-1):-1:1
        u_factor = Tv(-du[k])
        d_factor = Tv(inv_d[k])
        @simd for i in 1:n_batch
            z[i, k] = muladd(u_factor, z[i, k+1], z[i, k]) * d_factor
        end
    end
    return z
end

# Dispatch to SIMD-optimized solver (only for D ≥ 2)
@inline _ldiv_along_dim!(z, thomas, ::Val{D}) where {D} = _ldiv_along_dim_vectorized!(z, thomas)

# Val(1) is explicitly unsupported - benchmarking showed per-column approach is faster
@noinline function _ldiv_along_dim!(z, thomas, ::Val{1})
    throw(ArgumentError(
        "Batch solving along axis 1 (Val(1)) is not supported.\n" *
        "Reason: Per-column approach is faster due to view creation overhead.\n" *
        "Use _solve_system! in a loop for axis 1, or use Val(2) for SIMD-optimized batch solving."
    ))
end

# ========================================
# HIGH-LEVEL BATCH SOLVER INTERFACE
# ========================================

"""
    solve_along_dim!(out_z, cache, data, bc, ::Val{D})

Compute cubic spline second derivatives (moments) for batch systems along dimension D.
Optimized for memory locality and SIMD execution.

# Use Cases
- 2D grids: `Val(2)` for y-direction (rows) - SIMD optimized
- 3D grids: `Val(2)`, `Val(3)` for non-primary axes - SIMD optimized
- For axis 1: Use `_solve_system!` in a loop (per-column approach is faster)

# Type Parameters
- `Tv`: Value type (Real, Complex, or AD type)
- `Tg`: Grid type (AbstractFloat) for cache and spacing

# Arguments
- `out_z::AbstractMatrix{Tv}`: Output array (same size as data, modified in-place)
- `cache::CubicSplineCache{Tg}`: Precomputed Thomas factorization and grid info
- `data::AbstractMatrix{Tv}`: Input data array
- `bc::BCPair`: Boundary condition pair
- `::Val{D}`: Dimension along which to solve (D ≥ 2 only)
"""
function solve_along_dim!(
    out_z::AbstractMatrix{Tv},
    cache::CubicSplineCache{Tg,X,F,BC_cache,S},
    data::AbstractMatrix{Tv},
    bc::BCPair,
    dim::Val{D}
) where {Tv, Tg<:AbstractFloat, X, F, BC_cache, S<:AbstractGridSpacing{Tg}, D}
    # Step 1: Compute RHS for all systems
    # Note: bc can have different value type than cache.bc_config (e.g., ComplexF64 vs Float64)
    compute_rhs_along_dim!(out_z, data, cache.x, cache.spacing, bc, dim)

    # Step 2: Batch solve (SIMD for D≥2)
    _ldiv_along_dim!(out_z, cache.thomas, dim)

    return out_z
end

# ========================================
# BATCH RHS COMPUTATION
# ========================================

"""
    compute_rhs_along_dim!(D, data, x, spacing, bc, ::Val{D})

Compute RHS for batch systems along dimension `D`.

# Arguments
- `D::AbstractMatrix{T}`: Output RHS matrix (modified in-place)
- `data::AbstractMatrix{T}`: Input data matrix
- `x::AbstractVector{T}`: Grid points
- `spacing::AbstractGridSpacing{T}`: Grid spacing object
- `bc::BCPair{T}`: Boundary condition pair
- `::Val{D}`: Dimension along which to compute RHS
"""
# Val(1) is explicitly unsupported
@noinline function compute_rhs_along_dim!(
    D::AbstractMatrix{Tv},
    data::AbstractMatrix{Tv},
    x::AbstractVector{Tg},
    spacing::AbstractGridSpacing{Tg},
    bc::BCPair,
    ::Val{1}
) where {Tv, Tg<:AbstractFloat}
    throw(ArgumentError(
        "Batch RHS computation along axis 1 (Val(1)) is not supported.\n" *
        "Use compute_rhs! in a loop for axis 1, or use Val(2) for batch computation."
    ))
end

function compute_rhs_along_dim!(
    D::AbstractMatrix{Tv},
    data::AbstractMatrix{Tv},
    x::AbstractVector{Tg},
    spacing::AbstractGridSpacing{Tg},
    bc::BCPair,
    ::Val{2}
) where {Tv, Tg<:AbstractFloat}
    n_batch = size(data, 1)
    @inbounds for i in 1:n_batch
        compute_rhs!(view(D, i, :), view(data, i, :), x, spacing, bc)
    end
    return D
end

# Edge BC tuple version: per-row boundary values (SIMD optimized)
"""
    compute_rhs_along_dim!(D, data, spacing, edge_bc::Tuple, ::Val{2})

Compute RHS for batch systems along axis 2 with per-row edge boundary conditions.
SIMD optimized - inner loop over contiguous axis 1.

# Type Parameters
- `Tv`: Value type (Real, Complex, or AD type)
- `Tg`: Grid type (AbstractFloat) for spacing

# Arguments
- `D::AbstractMatrix{Tv}`: Output RHS matrix (modified in-place)
- `data::AbstractMatrix{Tv}`: Input data matrix
- `spacing::AbstractGridSpacing{Tg}`: Grid spacing object
- `edge_bc::Tuple{Vector, Vector}`: (bc_first, bc_last) - per-row Deriv1 BC values
- `::Val{2}`: Solve along axis 2 (rows)
"""
function compute_rhs_along_dim!(
    D::AbstractMatrix{Tv},
    data::AbstractMatrix{Tv},
    spacing::AbstractGridSpacing{Tg},
    edge_bc::Tuple{AbstractVector{Tv}, AbstractVector{Tv}},
    ::Val{2}
) where {Tv, Tg<:AbstractFloat}
    bc_first, bc_last = edge_bc
    nx, ny = size(data)
    n = ny - 1

    @inbounds begin
        # First column of RHS: d[1] = 6 * ((y[2] - y[1])/h[1] - bc_first)
        h1 = Tv(_get_h(spacing, 1))
        inv_h1 = inv(h1)
        six_inv_h1 = Tv(6) * inv_h1
        @simd for i in 1:nx
            D[i, 1] = muladd(six_inv_h1, data[i, 2] - data[i, 1], Tv(-6) * bc_first[i])
        end

        # Interior columns: d[j] = 6 * ((y[j+1] - y[j])/h[j] - (y[j] - y[j-1])/h[j-1])
        _compute_rhs_interior_batch!(D, data, spacing, Val(2))

        # Last column of RHS: d[end] = 6 * (bc_last - (y[end] - y[end-1])/h[n])
        h_n = Tv(_get_h(spacing, n))
        inv_h_n = inv(h_n)
        six_inv_h_n = Tv(6) * inv_h_n
        @simd for i in 1:nx
            D[i, ny] = muladd(-six_inv_h_n, data[i, ny] - data[i, ny-1], Tv(6) * bc_last[i])
        end
    end

    return D
end

"""
    _compute_rhs_interior_batch!(D, data, spacing, ::Val{2})

Compute interior RHS values (indices 2:n-1) for batch systems along axis 2.
SIMD optimized with muladd.

# Type Parameters
- `Tv`: Value type (Real, Complex, or AD type)
- `Tg`: Grid type (AbstractFloat) for spacing
"""
@inline function _compute_rhs_interior_batch!(
    D::AbstractMatrix{Tv},
    data::AbstractMatrix{Tv},
    spacing::AbstractGridSpacing{Tg},
    ::Val{2}
) where {Tv, Tg<:AbstractFloat}
    nx, ny = size(data)
    @inbounds for j in 2:(ny-1)
        h_prev = Tv(_get_h(spacing, j - 1))
        h_curr = Tv(_get_h(spacing, j))
        inv_h_prev = inv(h_prev)
        inv_h_curr = inv(h_curr)
        six_inv_h_curr = Tv(6) * inv_h_curr
        six_inv_h_prev = Tv(6) * inv_h_prev

        @simd for i in 1:nx
            forward = (data[i, j+1] - data[i, j]) * six_inv_h_curr
            backward = (data[i, j] - data[i, j-1]) * six_inv_h_prev
            D[i, j] = forward - backward
        end
    end
    return D
end

# ========================================
# BATCH MOMENT-TO-DERIVATIVE CONVERSION
# ========================================

"""
    moments_to_derivatives_along_dim!(out, M, data, spacing, bc, ::Val{D})

Convert moments to derivatives for batch systems along dimension D.

# Arguments
- `out::AbstractMatrix{T}`: Output derivatives (modified in-place)
- `M::AbstractMatrix{T}`: Input moments (second derivatives)
- `data::AbstractMatrix{T}`: Original function values
- `spacing::AbstractGridSpacing{T}`: Grid spacing
- `bc`: Boundary condition configuration
- `::Val{D}`: Dimension along which conversion was performed
"""
# Val(1) is explicitly unsupported
@noinline function moments_to_derivatives_along_dim!(
    out::AbstractMatrix{Tv},
    M::AbstractMatrix{Tv},
    data::AbstractMatrix{Tv},
    spacing,
    bc,
    ::Val{1}
) where {Tv}
    throw(ArgumentError(
        "Batch moment-to-derivative along axis 1 (Val(1)) is not supported.\n" *
        "Use _moments_to_derivatives_1d! in a loop for axis 1, or use Val(2) for batch conversion."
    ))
end

function moments_to_derivatives_along_dim!(
    out::AbstractMatrix{Tv},
    M::AbstractMatrix{Tv},
    data::AbstractMatrix{Tv},
    spacing::AbstractGridSpacing{Tg},
    bc,
    ::Val{2}
) where {Tv, Tg<:AbstractFloat}
    n_batch = size(data, 1)
    @inbounds for i in 1:n_batch
        _moments_to_derivatives_1d!(
            view(out, i, :), view(M, i, :), view(data, i, :), spacing
        )
        _apply_derivative_bc!(view(out, i, :), bc)
    end
    return out
end

# Edge BC tuple version: per-row boundary derivative values (SIMD optimized)
"""
    moments_to_derivatives_along_dim!(out, M, data, spacing, edge_bc::Tuple, ::Val{2})

Convert moments to derivatives with per-row edge boundary conditions.
SIMD optimized - applies edge BC values directly.

# Type Parameters
- `Tv`: Value type (Real, Complex, or AD type)
- `Tg`: Grid type (AbstractFloat) for spacing

# Arguments
- `out::AbstractMatrix{Tv}`: Output derivatives (modified in-place)
- `M::AbstractMatrix{Tv}`: Input moments (second derivatives)
- `data::AbstractMatrix{Tv}`: Original function values
- `spacing::AbstractGridSpacing{Tg}`: Grid spacing
- `edge_bc::Tuple{Vector, Vector}`: (bc_first, bc_last) - per-row derivative values at boundaries
- `::Val{2}`: Conversion along axis 2
"""
function moments_to_derivatives_along_dim!(
    out::AbstractMatrix{Tv},
    M::AbstractMatrix{Tv},
    data::AbstractMatrix{Tv},
    spacing::AbstractGridSpacing{Tg},
    edge_bc::Tuple{AbstractVector{Tv}, AbstractVector{Tv}},
    ::Val{2}
) where {Tv, Tg<:AbstractFloat}
    bc_first, bc_last = edge_bc
    nx, ny = size(data)

    # Step 1: Compute interior derivatives from moments (SIMD over axis 1)
    _moments_to_derivatives_interior_batch!(out, M, data, spacing, Val(2))

    # Step 2: Apply edge BC values directly at boundaries
    @inbounds begin
        @simd for i in 1:nx
            out[i, 1] = bc_first[i]
        end
        @simd for i in 1:nx
            out[i, ny] = bc_last[i]
        end
    end

    return out
end

"""
    _moments_to_derivatives_interior_batch!(out, M, data, spacing, ::Val{2})

Compute interior derivatives (indices 2:n) from moments for batch systems.
SIMD optimized with muladd.

Formula: dydx[i+1] = (y[i+1] - y[i])/h + h/6 * (m[i] + 2*m[i+1])

# Type Parameters
- `Tv`: Value type (Real, Complex, or AD type)
- `Tg`: Grid type (AbstractFloat) for spacing
"""
@inline function _moments_to_derivatives_interior_batch!(
    out::AbstractMatrix{Tv},
    M::AbstractMatrix{Tv},
    data::AbstractMatrix{Tv},
    spacing::AbstractGridSpacing{Tg},
    ::Val{2}
) where {Tv, Tg<:AbstractFloat}
    nx, ny = size(data)
    inv_6 = inv(Tv(6))

    @inbounds for j in 1:(ny-1)
        h = Tv(_get_h(spacing, j))
        inv_h = Tv(_get_inv_h(spacing, j))
        h_over_6 = h * inv_6

        @simd for i in 1:nx
            linear_slope = (data[i, j+1] - data[i, j]) * inv_h
            moment_sum = muladd(Tv(2), M[i, j+1], M[i, j])
            out[i, j+1] = muladd(h_over_6, moment_sum, linear_slope)
        end
    end

    return out
end
