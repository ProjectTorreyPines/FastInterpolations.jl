# ========================================
# Generic ND Coefficient Construction
# ========================================
#
# Functions to compute precomputed partial derivatives for N-dimensional
# cubic Hermite interpolation.
#
# Key features:
# - Tg/Tv type separation (grid vs value types)
# - Batch SIMD optimization for dimensions d≥2
# - Bit-encoding build-up algorithm for 2^N partials
# - Memory-efficient reshape trick for ND→1D slicing
#
# Partials layout (2^N, n₁, n₂, ..., nₙ):
#   Index p encodes derivatives via binary representation
#   Bit d-1 set → differentiate w.r.t. dimension d

# ========================================
# 1D Slice Differentiation (Generic ND)
# ========================================

"""
    _differentiate_nd_along_dim!(out, data, grid, bc, d)

Compute ∂f/∂xₐ for N-dimensional array along dimension `d` using cubic splines.

# Algorithm
Uses the "reshape to 3D" trick to iterate over all 1D slices:
- Reshape N-D array to (before × nₐ × after) where nₐ = size(data, d)
- Each slice `data_3d[i, :, j]` is a 1D slice along dim d
- Apply `_deriv_1d!` to each slice

This approach is truly N-D generic because:
- Axis 1 (before): flattened indices for dims 1 to d-1
- Axis 2 (nₐ): the dimension we differentiate along
- Axis 3 (after): flattened indices for dims d+1 to N

# Type Parameters
- `Tg`: Grid type (AbstractFloat) for coordinates
- `Tv`: Value type for data (can be Real, Complex, or AD types)

# Arguments
- `out::AbstractArray{Tv,N}`: Output array (same shape as data), stores ∂f/∂xₐ
- `data::AbstractArray{Tv,N}`: Input N-D array f(x₁, x₂, ..., xₙ)
- `grid::AbstractVector{Tg}`: Grid points along dimension d
- `bc::AbstractBC`: Boundary condition for differentiation
- `d::Int`: Dimension to differentiate along (1 ≤ d ≤ N)

# Example
For a 4D array f(x₁,x₂,x₃,x₄) with shape (10, 20, 30, 40), computing ∂f/∂x₂ (d=2):
- shape_before = 10
- n₂ = 20
- shape_after = 30 × 40 = 1200
- Reshaped view: (10, 20, 1200)
- Process 10 × 1200 = 12,000 independent 1D slices of length 20
"""
@with_pool pool function _differentiate_nd_along_dim!(
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

            # Differentiate using existing _deriv_1d!
            _deriv_1d!(dline, line, grid, bc)

            # Write back
            for k in 1:n_d
                out_3d[i, k, j] = dline[k]
            end
        end
    end

    return out
end

# ========================================
# Batch SIMD Differentiation (d ≥ 2)
# ========================================

"""
    _differentiate_nd_along_dim_batch!(out, data, grid, bc, d)

Batch-optimized version of `_differentiate_nd_along_dim!` using SIMD vectorization.

Uses the same reshape trick but when `shape_before > 1`, processes all "before"
slices simultaneously using the batch solver from 2D implementation.

# Type Parameters
- `Tg`: Grid type (AbstractFloat) for coordinates
- `Tv`: Value type for data (can be Real, Complex, or AD types)

# Performance Characteristics
- **d=1** (`shape_before=1`): Falls back to per-slice approach (no batch benefit)
- **d≥2** (`shape_before>1`): Uses batch SIMD - significant speedup for large grids
- **PeriodicBC**: Uses per-slice approach (batch solver doesn't support periodic)

# See Also
- `_differentiate_nd_along_dim!`: Original per-slice implementation
- `solve_along_dim!`: 2D batch solver used internally
"""
@with_pool pool function _differentiate_nd_along_dim_batch!(
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

    # Reshape to 3D views
    data_3d = reshape(data, shape_before, n_d, shape_after)
    out_3d = reshape(out, shape_before, n_d, shape_after)

    # Check if batch optimization is applicable
    is_periodic = _is_periodic_bc(bc)
    can_batch = shape_before > 1 && !is_periodic

    if can_batch
        # Batch SIMD path: use 2D batch solver along axis 2
        # Grid type Tg for cache, value type Tv for computation
        bc_cache = _is_periodic_bc(bc) ? PeriodicBC() : _normalize_bc(bc, Tg)
        cache = _get_cubic_cache(grid, bc_cache, true)
        actual_bc = cache.bc_config isa PeriodicData ? cache.bc_config : _normalize_bc(bc, Tv)

        # Acquire workspace for moments matrix
        M = acquire!(pool, Tv, (shape_before, n_d))

        @inbounds for j in 1:shape_after
            # Get 2D views for this "after" slice
            data_2d = view(data_3d, :, :, j)
            out_2d = view(out_3d, :, :, j)

            # Batch solve: compute moments for all rows simultaneously
            solve_along_dim!(M, cache, data_2d, actual_bc, Val(2))

            # Batch convert: moments → derivatives for all rows
            moments_to_derivatives_along_dim!(out_2d, M, data_2d, cache.spacing, actual_bc, Val(2))
        end
    else
        # Fall back to per-slice approach (d=1 or PeriodicBC)
        line = acquire!(pool, Tv, n_d)
        dline = acquire!(pool, Tv, n_d)

        @inbounds for j in 1:shape_after
            for i in 1:shape_before
                # Extract 1D slice
                for k in 1:n_d
                    line[k] = data_3d[i, k, j]
                end

                # Differentiate
                _deriv_1d!(dline, line, grid, bc)

                # Write back
                for k in 1:n_d
                    out_3d[i, k, j] = dline[k]
                end
            end
        end
    end

    return out
end

# ========================================
# Periodic Data Validation (Generic ND)
# ========================================

"""
    _check_periodic_data_nd(data::AbstractArray{Tv, N}, d::Int)

Validate that data is periodic along dimension `d`.

Checks that `data[..., 1, ...] ≈ data[..., end, ...]` for the given dimension.
Uses `selectdim` for clean N-D slice comparison.

# Arguments
- `data::AbstractArray{Tv,N}`: N-dimensional array to validate
- `d::Int`: Dimension to check for periodicity (1 ≤ d ≤ N)

# Throws
- `ArgumentError`: If first and last slices along dimension d differ significantly
"""
function _check_periodic_data_nd(data::AbstractArray{Tv, N}, d::Int) where {Tv, N}
    @boundscheck 1 ≤ d ≤ N || throw(ArgumentError("dimension d=$d out of range 1:$N"))

    n_d = size(data, d)
    atol = real(Tv) === Float32 ? _PERIODIC_ATOL_F32 : _PERIODIC_ATOL_F64

    # selectdim returns views: data[:,...,1,...,:] and data[:,...,end,...,:]
    slice_first = selectdim(data, d, 1)
    slice_last = selectdim(data, d, n_d)

    # Element-wise comparison with tolerance
    @inbounds for i in eachindex(slice_first, slice_last)
        if !isapprox(slice_first[i], slice_last[i]; atol=atol)
            throw(ArgumentError(
                "Periodic BC on dim $d requires data to match at first/last indices, " *
                "but found diff=$(abs(slice_last[i] - slice_first[i])) at linear index $i"
            ))
        end
    end

    return nothing
end

"""
    _check_periodic_data_noalloc!(data::AbstractArray{Tv, N}, ::Val{D})

Zero-allocation periodic data validation for dimension D.

Validates `data[..., 1, ...] ≈ data[..., end, ...]` along compile-time dimension D.

Uses `@generated` to emit explicit nested for-loops with direct array indexing
(`data[i1, ..., 1, ..., iN]` vs `data[i1, ..., n_D, ..., iN]`) — no closures,
no `SubArray`, no `CartesianIndex` construction from runtime tuples.  Closure-based
approaches (`ntuple(i -> f(runtime_val, i), Val(N))`) would heap-allocate a closure
box for each captured runtime variable; `@generated` avoids this entirely by baking
the index expressions into the method body at specialization time.
"""
@generated function _check_periodic_data_noalloc!(
    data::AbstractArray{Tv, N},
    ::Val{D}
) where {Tv, N, D}
    # Symbolic loop variables: i1, i2, ..., iN
    idx_vars = [Symbol("i", d) for d in 1:N]

    # Direct indexing expressions for first and last slice along dim D.
    # D is a compile-time constant here, so the literal 1 / :n_D are baked in.
    first_idx = [d == D ? 1    : idx_vars[d] for d in 1:N]
    last_idx  = [d == D ? :n_D : idx_vars[d] for d in 1:N]

    # Inner comparison body: direct indexing, no intermediary objects
    check = quote
        v1 = @inbounds data[$(first_idx...)]
        vn = @inbounds data[$(last_idx...)]
        if !isapprox(v1, vn; atol=atol)
            throw(ArgumentError(
                "Periodic BC on dim $D requires data to match at first/last indices, " *
                "but found diff=$(abs(v1 - vn))"
            ))
        end
    end

    # Wrap in nested loops over all dims except D (outermost = N, innermost = 1)
    body = check
    for d in N:-1:1
        d == D && continue
        body = quote
            for $(idx_vars[d]) in axes(data, $d)
                $body
            end
        end
    end

    return quote
        n_D  = size(data, $D)
        atol = real(Tv) === Float32 ? _PERIODIC_ATOL_F32 : _PERIODIC_ATOL_F64
        $body
        return nothing
    end
end

# ========================================
# Effective BC Selection for Mixed Partials
# ========================================

"""
    _get_effective_bc(bc::AbstractBC, p_src::Int, grid::AbstractVector)

Select the effective boundary condition for computing a mixed partial derivative.

When computing higher-order mixed partials (e.g., ∂²f/∂x∂y from ∂f/∂x), the
"effective" BC may differ from the user-specified BC for better accuracy.

# Selection Rules
1. `p_src == 1` (pure derivative, source is f): Use specified BC unchanged
2. `_is_periodic_bc(bc)`: Always propagate periodic BC for consistency
3. `length(grid) ≥ 4` and PolyFit available: Use CubicFit for better edge accuracy
4. Fallback: Use NaturalBC

# Arguments
- `bc::AbstractBC`: User-specified boundary condition
- `p_src::Int`: Source partial index (1 = f, 2 = ∂f/∂x₁, 3 = ∂f/∂x₂, etc.)
- `grid::AbstractVector`: Grid points for the differentiation dimension

# Returns
- `AbstractBC`: The effective BC to use for differentiation
"""
@inline function _get_effective_bc(bc::AbstractBC, p_src::Int, grid::AbstractVector)
    # Rule 1: Pure derivative (source is f) - use specified BC
    if p_src == 1
        return bc
    end

    # Rule 2: Periodic BC always propagates
    if _is_periodic_bc(bc)
        return bc
    end

    # Rule 3: For mixed partials with enough grid points, use CubicFit
    if get_polyfit_degree(bc) > 0 || length(grid) >= 4
        return CubicFit()
    end

    # Fallback: NaturalBC
    return NaturalBC()
end

# ========================================
# Generic ND Partial Derivative Computation
# ========================================

@inline _validate_nd_partials_dims!(
    partials::AbstractArray,
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    ::Val{N}
) where {Tv, Tg<:AbstractFloat, N} = _validate_nd_partials_dims!(partials, grids, data, Val(1), Val(N))

@inline function _validate_nd_partials_dims!(
    partials::AbstractArray,
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    ::Val{D},
    ::Val{N}
) where {Tv, Tg<:AbstractFloat, D, N}
    size(partials, D + 1) == size(data, D) || throw(DimensionMismatch(
        "partials dim $(D+1) must match data dim $D"
    ))
    size(data, D) == length(grids[D]) || throw(DimensionMismatch(
        "data dim $D must match grid $D length"
    ))
    if D < N
        _validate_nd_partials_dims!(partials, grids, data, Val(D + 1), Val(N))
    end
    return nothing
end

@inline _validate_nd_bcs!(
    grids::NTuple{N, AbstractVector{Tg}},
    bcs::NTuple{N, AbstractBC},
    data::AbstractArray{Tv, N},
    ::Val{N}
) where {Tv, Tg<:AbstractFloat, N} = _validate_nd_bcs!(grids, bcs, data, Val(1), Val(N))

@inline function _validate_nd_bcs!(
    grids::NTuple{N, AbstractVector{Tg}},
    bcs::NTuple{N, AbstractBC},
    data::AbstractArray{Tv, N},
    ::Val{D},
    ::Val{N}
) where {Tv, Tg<:AbstractFloat, D, N}
    # Only validate inclusive PeriodicBC: for exclusive, the endpoint is not yet present
    # in the data (it is added by _prepare_periodic_nd/_prepare_periodic_nd_pooled after
    # this validation).  Checking data[1] ≈ data[end] on unextended exclusive data would
    # produce false positives for perfectly valid periodic inputs.
    if bcs[D] isa PeriodicBC{:inclusive}
        _check_periodic_data_noalloc!(data, Val(D))
    end
    polyfit_deg = get_polyfit_degree(bcs[D])
    if polyfit_deg > 0 && length(grids[D]) < polyfit_deg + 1
        throw(ArgumentError("PolyFit BC on dimension $D requires at least $(polyfit_deg+1) points"))
    end
    if D < N
        _validate_nd_bcs!(grids, bcs, data, Val(D + 1), Val(N))
    end
    return nothing
end

# PolyFit-only validation (zero-alloc). Kept for callers that have no `data` array.
# One-shot paths now use _validate_nd_bcs! directly (also zero-alloc after the
# _check_periodic_data_noalloc! refactor).
@inline _validate_polyfit_bcs(
    grids::NTuple{N, AbstractVector},
    bcs::NTuple{N, AbstractBC},
    ::Val{N}
) where {N} = _validate_polyfit_bcs(grids, bcs, Val(1), Val(N))

@inline function _validate_polyfit_bcs(
    grids::NTuple{N, AbstractVector},
    bcs::NTuple{N, AbstractBC},
    ::Val{D},
    ::Val{N}
) where {D, N}
    polyfit_deg = get_polyfit_degree(bcs[D])
    if polyfit_deg > 0 && length(grids[D]) < polyfit_deg + 1
        throw(ArgumentError("PolyFit BC on dimension $D requires at least $(polyfit_deg+1) points"))
    end
    if D < N
        _validate_polyfit_bcs(grids, bcs, Val(D + 1), Val(N))
    end
    return nothing
end

@inline _build_nd_partials_dim!(
    partials::AbstractArray{Tv, NP1},
    grids::NTuple{N, AbstractVector{Tg}},
    bcs::NTuple{N, AbstractBC},
    ::Val{N}
) where {Tv, Tg<:AbstractFloat, N, NP1} = _build_nd_partials_dim!(partials, grids, bcs, Val(1), Val(N))

@inline function _build_nd_partials_dim!(
    partials::AbstractArray{Tv, NP1},
    grids::NTuple{N, AbstractVector{Tg}},
    bcs::NTuple{N, AbstractBC},
    ::Val{D},
    ::Val{N}
) where {Tv, Tg<:AbstractFloat, D, N, NP1}
    bit_d = 1 << (D - 1)
    @inbounds for p_src in 1:bit_d
        p_dst = p_src + bit_d
        effective_bc = _get_effective_bc(bcs[D], p_src, grids[D])
        src_view = selectdim(partials, 1, p_src)
        dst_view = selectdim(partials, 1, p_dst)
        _differentiate_nd_along_dim_batch!(dst_view, src_view, grids[D], effective_bc, D)
    end
    if D < N
        _build_nd_partials_dim!(partials, grids, bcs, Val(D + 1), Val(N))
    end
    return partials
end

"""
    _compute_nd_partials!(partials, grids, data, bcs)

Compute all 2ᴺ partial derivatives for N-dimensional Hermite interpolation.

Uses the **bit-encoding build-up algorithm**:
- Index `p` in `partials[p, ...]` encodes which dimensions have been differentiated
- Bit `d-1` set in `p-1` means differentiated w.r.t. dimension `d`
- Higher-order partials are built stage-by-stage from lower-order ones

# Type Parameters
- `Tg`: Grid type (AbstractFloat) for coordinates
- `Tv`: Value type for data (can be Real, Complex, or AD types)

# Algorithm
```
Stage 0: partials[1, ...] = data  (f, no derivatives)

For d = 1 to N:
    bit_d = 2^(d-1)
    For p_src = 1 to bit_d:  # partials without bit d-1 set
        p_dst = p_src + bit_d  # add bit d-1
        effective_bc = _get_effective_bc(bcs[d], p_src, grids[d])
        _differentiate_nd_along_dim!(partials[p_dst], partials[p_src], grids[d], bc, d)
```

# Partial Index Encoding (Example for N=3)

| Index | Binary | Partial Derivative | Built From |
|-------|--------|-------------------|------------|
| 1     | 000    | f                 | data (copy)|
| 2     | 001    | ∂f/∂x₁            | d=1: diff f |
| 3     | 010    | ∂f/∂x₂            | d=2: diff f |
| 4     | 011    | ∂²f/∂x₁∂x₂        | d=2: diff ∂f/∂x₁ |
| 5     | 100    | ∂f/∂x₃            | d=3: diff f |
| 6     | 101    | ∂²f/∂x₁∂x₃        | d=3: diff ∂f/∂x₁ |
| 7     | 110    | ∂²f/∂x₂∂x₃        | d=3: diff ∂f/∂x₂ |
| 8     | 111    | ∂³f/∂x₁∂x₂∂x₃     | d=3: diff ∂²f/∂x₁∂x₂ |

# Arguments
- `partials::AbstractArray{Tv, N+1}`: Output array with shape (2ᴺ, size(data)...)
- `grids::NTuple{N, AbstractVector{Tg}}`: Grid points for each dimension
- `data::AbstractArray{Tv, N}`: Input N-dimensional data array
- `bcs::NTuple{N, AbstractBC}`: Boundary conditions for each dimension

# Notes
- Validates periodic data if PeriodicBC is specified
- Uses batch SIMD optimization for d≥2 with NaturalBC
"""
function _compute_nd_partials!(
    partials::AbstractArray{Tv, NP1},
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    bcs::NTuple{N, AbstractBC}
) where {Tv, Tg<:AbstractFloat, N, NP1}
    # Validate dimensions (fast, no allocation)
    @boundscheck begin
        NP1 == N + 1 || throw(DimensionMismatch("partials must have N+1 dimensions"))
        n_partials = 1 << N  # 2^N
        size(partials, 1) == n_partials || throw(DimensionMismatch(
            "partials first dimension must be 2^N=$(n_partials), got $(size(partials, 1))"
        ))
        _validate_nd_partials_dims!(partials, grids, data, Val(N))
    end

    # Stage 0: Copy f (the function values) into partials[1, ...]
    f_partial = selectdim(partials, 1, 1)
    copyto!(f_partial, data)

    # Build up higher-order partials stage by stage
    _build_nd_partials_dim!(partials, grids, bcs, Val(N))

    return partials
end

# ========================================
# Build ND Coefficients (High-Level API)
# ========================================

"""
    _build_nd_coeffs(grids, data, bcs) -> NodalDerivativesND{Tv, N, NP1}

Compute all partial derivatives for N-dimensional Hermite interpolation.

# Type Parameters
- `Tg`: Grid type (AbstractFloat) for coordinates
- `Tv`: Value type for data (can be Real, Complex, or AD types)

# Arguments
- `grids::NTuple{N, AbstractVector{Tg}}`: Grid points for each dimension
- `data::AbstractArray{Tv, N}`: Function values at grid points
- `bcs::NTuple{N, AbstractBC}`: Boundary conditions for each dimension

# Returns
- `NodalDerivativesND{Tv, N, N+1}` containing the partials array
"""
function _build_nd_coeffs(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    bcs::NTuple{N, AbstractBC}
) where {Tg<:AbstractFloat, Tv, N}
    # Validate periodic BCs and PolyFit requirements (runs once at construction time)
    _validate_nd_bcs!(grids, bcs, data, Val(N))

    # Allocate partials array: (2^N, n₁, n₂, ..., nₙ)
    n_partials = 1 << N
    partials_shape = (n_partials, size(data)...)
    partials = Array{Tv, N+1}(undef, partials_shape)

    # Compute all partial derivatives
    _compute_nd_partials!(partials, grids, data, bcs)

    return NodalDerivativesND{Tv, N, N+1}(partials)
end
