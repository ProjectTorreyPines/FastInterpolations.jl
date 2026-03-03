# ========================================
# Generic ND Quadratic Coefficient Construction
# ========================================
#
# Functions to compute precomputed partial derivatives for N-dimensional
# quadratic interpolation using per-axis slope recurrence.
#
# Key features:
# - Tg/Tv type separation (grid vs value types)
# - Bit-encoding build-up algorithm for 2^N partials (shared with cubic)
# - Memory-efficient reshape trick for ND→1D slicing
# - Uses quadratic recurrence (_fill_slopes!) instead of Thomas tridiagonal
#
# Partials layout (2^N, n₁, n₂, ..., nₙ):
#   Index p encodes derivatives via binary representation
#   Bit d-1 set → differentiate w.r.t. dimension d

# ========================================
# 1D Slice Differentiation (Quadratic Recurrence)
# ========================================

"""
    _slope_1d_quadratic!(d_out, values, grid, bc)

Compute slopes (first derivatives) for a 1D data slice using quadratic
spline recurrence. This is the quadratic analog of the cubic `_deriv_1d!`.

# Algorithm
1. Compute secant slopes: s[i] = (y[i+1] - y[i]) / h[i]
2. BC-dispatched recurrence via `_fill_slopes!`:
   - Left BC:  d[1] from BC, forward recurrence d[i+1] = 2*s[i] - d[i]
   - Right BC: d[n] from BC, backward recurrence d[i] = 2*s[i] - d[i+1]

# Arguments
- `d_out::AbstractVector{Tv}`: Output slope array (length n)
- `values::AbstractVector{Tv}`: Input 1D values (length n)
- `grid::AbstractVector{Tg}`: Grid points for this dimension
- `bc::QuadraticBC`: Boundary condition (Left, Right, or MinCurvFit)
"""
@with_pool pool function _slope_1d_quadratic!(
    d_out::AbstractVector{Tv},
    values::AbstractVector{Tv},
    grid::AbstractVector{Tg},
    bc::QuadraticBC
) where {Tv, Tg<:AbstractFloat}
    n = length(values)
    @assert n == length(grid) "values and grid must have same length"
    @assert n >= 2 "Need at least 2 points"

    nm1 = n - 1
    h = acquire!(pool, Tg, nm1)
    inv_h = acquire!(pool, Tg, nm1)
    secant = acquire!(pool, Tv, nm1)

    # 1. Compute grid spacing
    _compute_grid_spacing!(h, inv_h, grid)

    # 2. Compute secant slopes
    _compute_quadratic_secants!(secant, values, inv_h)

    # 3. Normalize BC values to Tv (lazy normalization — Tv is known here)
    # This ensures _fill_slopes! always receives Tv-typed values,
    # making convert(Tv, bc.val) an identity operation.
    bc_typed = _normalize_bc(bc, first(values))

    # 4. Fill slopes via BC-dispatched recurrence
    _fill_slopes!(d_out, secant, h, bc_typed, grid, values)

    return d_out
end

# Dispatch wrappers for non-QuadraticBC types (lazy normalization pattern).
# Convert AbstractBC → QuadraticBC at the point where Tv is known, then delegate
# to the core @with_pool method. Matches cubic's pattern in _deriv_1d!.
@inline function _slope_1d_quadratic!(
    d_out::AbstractVector{Tv}, values::AbstractVector{Tv},
    grid::AbstractVector{Tg}, ::ZeroCurvBC
) where {Tv, Tg<:AbstractFloat}
    _slope_1d_quadratic!(d_out, values, grid, Right(Deriv2(0 * first(values))))
end

@inline function _slope_1d_quadratic!(
    d_out::AbstractVector{Tv}, values::AbstractVector{Tv},
    grid::AbstractVector{Tg}, ::ZeroSlopeBC
) where {Tv, Tg<:AbstractFloat}
    _slope_1d_quadratic!(d_out, values, grid, Left(Deriv1(0 * first(values))))
end

@inline function _slope_1d_quadratic!(
    d_out::AbstractVector{Tv}, values::AbstractVector{Tv},
    grid::AbstractVector{Tg}, bc::PolyFit
) where {Tv, Tg<:AbstractFloat}
    _slope_1d_quadratic!(d_out, values, grid, Right(bc))
end

# Fallback: reject unsupported BC types with a clear error message
function _slope_1d_quadratic!(
    ::AbstractVector, ::AbstractVector,
    ::AbstractVector, bc::AbstractBC
)
    throw(ArgumentError(
        "Unsupported boundary condition for quadratic interpolation: $(typeof(bc)). " *
        "Supported: Left(...), Right(...), MinCurvFit, ZeroCurvBC, ZeroSlopeBC, or PolyFit variants."
    ))
end

# ========================================
# ND Along-Dimension Differentiation (Quadratic)
# ========================================

"""
    _differentiate_nd_along_dim_quadratic!(out, data, grid, bc, d)

Compute ∂f/∂xₐ for N-dimensional array along dimension `d` using quadratic recurrence.

Uses the same "reshape to 3D" trick as the cubic version:
- Reshape N-D array to (before × nₐ × after)
- Apply `_slope_1d_quadratic!` to each 1D slice

# Arguments
- `out::AbstractArray{Tv,N}`: Output array (same shape as data)
- `data::AbstractArray{Tv,N}`: Input N-D array
- `grid::AbstractVector{Tg}`: Grid points along dimension d
- `bc::QuadraticBC`: Boundary condition
- `d::Int`: Dimension to differentiate along (1 ≤ d ≤ N)
"""
@with_pool pool function _differentiate_nd_along_dim_quadratic!(
    out::AbstractArray{Tv,N},
    data::AbstractArray{Tv,N},
    grid::AbstractVector{Tg},
    bc::AbstractBC,
    d::Int
) where {Tv, Tg<:AbstractFloat, N}
    @boundscheck begin
        1 ≤ d ≤ N || throw(ArgumentError("dimension d=$d out of range 1:$N"))
        size(out) == size(data) || throw(DimensionMismatch("out and data must have same size"))
        size(data, d) == length(grid) || throw(DimensionMismatch(
            "size(data, $d)=$(size(data,d)) must match length(grid)=$(length(grid))"))
    end

    n_d = size(data, d)

    # Compute reshape dimensions
    shape_before = 1
    @inbounds for i in 1:(d-1)
        shape_before *= size(data, i)
    end
    shape_after = 1
    @inbounds for i in (d+1):N
        shape_after *= size(data, i)
    end

    # Reshape to 3D views (no allocation, just pointer arithmetic)
    data_3d = reshape(data, shape_before, n_d, shape_after)
    out_3d = reshape(out, shape_before, n_d, shape_after)

    # Acquire workspace buffers
    line = acquire!(pool, Tv, n_d)
    dline = acquire!(pool, Tv, n_d)

    # Iterate over all 1D slices
    @inbounds for j in 1:shape_after
        for i in 1:shape_before
            # Extract 1D slice along dimension d
            for k in 1:n_d
                line[k] = data_3d[i, k, j]
            end

            # Differentiate using quadratic recurrence
            _slope_1d_quadratic!(dline, line, grid, bc)

            # Write back
            for k in 1:n_d
                out_3d[i, k, j] = dline[k]
            end
        end
    end

    return out
end

# ========================================
# Effective BC Selection for Mixed Partials (Quadratic)
# ========================================

"""
    _get_effective_bc_quadratic(bc, p_src, grid)

Select the effective boundary condition for computing a quadratic mixed partial.

# Selection Rules
1. `p_src == 1` (pure derivative, source is f): Use specified BC unchanged
2. Periodic BC: Always propagate
3. With enough grid points (≥3): Use Right(QuadraticFit()) for mixed partials
4. Fallback: Right(ZeroCurvBC()) → Right(Deriv2(0))
"""
@inline function _get_effective_bc_quadratic(bc::AbstractBC, p_src::Int, grid::AbstractVector)
    p_src == 1 && return bc                       # raw AbstractBC preserved (lazy normalization)
    length(grid) >= 3 && return Right(QuadraticFit())
    return MinCurvFit()                           # 2-point grid: min curvature ≡ zero curvature
end

# ========================================
# Generic ND Partial Derivative Computation (Quadratic)
# ========================================

@inline _build_nd_partials_dim_quadratic!(
    partials::AbstractArray{Tv, NP1},
    grids::NTuple{N, AbstractVector{Tg}},
    bcs::NTuple{N, AbstractBC},
    ::Val{N}
) where {Tv, Tg<:AbstractFloat, N, NP1} =
    _build_nd_partials_dim_quadratic!(partials, grids, bcs, Val(1), Val(N))

@inline function _build_nd_partials_dim_quadratic!(
    partials::AbstractArray{Tv, NP1},
    grids::NTuple{N, AbstractVector{Tg}},
    bcs::NTuple{N, AbstractBC},
    ::Val{D},
    ::Val{N}
) where {Tv, Tg<:AbstractFloat, D, N, NP1}
    bit_d = 1 << (D - 1)
    @inbounds for p_src in 1:bit_d
        p_dst = p_src + bit_d
        effective_bc = _get_effective_bc_quadratic(bcs[D], p_src, grids[D])
        src_view = selectdim(partials, 1, p_src)
        dst_view = selectdim(partials, 1, p_dst)
        _differentiate_nd_along_dim_quadratic!(dst_view, src_view, grids[D], effective_bc, D)
    end
    if D < N
        _build_nd_partials_dim_quadratic!(partials, grids, bcs, Val(D + 1), Val(N))
    end
    return partials
end

"""
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)

Compute all 2^N partial derivatives for N-dimensional quadratic interpolation.

Uses the same bit-encoding build-up algorithm as cubic, but with quadratic
recurrence for 1D differentiation instead of Thomas tridiagonal.
"""
function _compute_nd_partials_quadratic!(
    partials::AbstractArray{Tv, NP1},
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    bcs::NTuple{N, AbstractBC}
) where {Tv, Tg<:AbstractFloat, N, NP1}
    # Validate dimensions
    @boundscheck begin
        NP1 == N + 1 || throw(DimensionMismatch("partials must have N+1 dimensions"))
        n_partials = 1 << N
        size(partials, 1) == n_partials || throw(DimensionMismatch(
            "partials first dimension must be 2^N=$(n_partials), got $(size(partials, 1))"
        ))
    end

    # Stage 0: Copy f (the function values) into partials[1, ...]
    f_partial = selectdim(partials, 1, 1)
    copyto!(f_partial, data)

    # Build up higher-order partials stage by stage
    _build_nd_partials_dim_quadratic!(partials, grids, bcs, Val(N))

    return partials
end

# ========================================
# Build ND Coefficients (High-Level API)
# ========================================

"""
    _build_nd_coeffs_quadratic(grids, data, bcs) -> NodalDerivativesND{Tv, N, NP1}

Compute all partial derivatives for N-dimensional quadratic interpolation.

# Returns
- `NodalDerivativesND{Tv, N, N+1}` containing the partials array
"""
function _build_nd_coeffs_quadratic(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    bcs::NTuple{N, AbstractBC}
) where {Tg<:AbstractFloat, Tv, N}
    # Allocate partials array: (2^N, n₁, n₂, ..., nₙ)
    n_partials = 1 << N
    partials_shape = (n_partials, size(data)...)
    partials = Array{Tv, N+1}(undef, partials_shape)

    # Compute all partial derivatives
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)

    return NodalDerivativesND{Tv, N, N+1}(partials)
end
