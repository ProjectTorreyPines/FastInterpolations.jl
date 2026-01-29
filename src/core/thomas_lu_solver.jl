# ========================================
# Thomas Algorithm LU Solver
# ========================================
# Generic Thomas algorithm (TDMA) solvers for tridiagonal systems.
# These work with any diagonally-dominant tridiagonal LU factorization.
#
# Currently used by:
# - Cubic spline system solvers (periodic + derivative BC)
# - Batch ND interpolation (SIMD-optimized vectorized solver)
#
# Design: Accepts LinearAlgebra.LU{Tridiagonal} with precomputed inv_d
# (reciprocals of diagonal) for cache-friendly, zero-allocation hot loops.

# ========================================
# Scalar Thomas Solver
# ========================================

"""
    _ldiv_tridiagonal_nopiv!(b, lu_factor, inv_d) -> b

Fast no-pivot Thomas algorithm solver for cached tridiagonal LU with precomputed `inv_d`.

Solves `Ax = b` in-place where `A = L*U` is the tridiagonal LU factorization.
Uses precomputed `inv_d = 1 ./ diag(U)` for cache-friendly backward substitution.

# Algorithm
1. **Forward elimination**: `b[i+1] -= L[i] * b[i]` for i = 1:n-1
2. **Backward substitution**: `b[i] = (b[i] - U[i] * b[i+1]) * inv_d[i]` for i = n:-1:1

# Arguments
- `b::AbstractVector{T}`: RHS vector (modified in-place to hold solution)
- `lu_factor`: LU factorization from `lu(Tridiagonal(...), NoPivot())`
- `inv_d::AbstractVector{T}`: Precomputed `1 ./ lu_factor.factors.d`

# Performance
- O(n) time complexity
- Zero allocations in hot path
- Uses `muladd` for fused multiply-add when available
"""
@inline function _ldiv_tridiagonal_nopiv!(
    b::AbstractVector{T},
    lu_factor::LinearAlgebra.LU{T,Tridiagonal{T,V},P},
    inv_d::AbstractVector{T},
) where {T<:AbstractFloat,V<:AbstractVector{T},P}
    dl = lu_factor.factors.dl
    du = lu_factor.factors.du

    n = length(inv_d)

    # Forward elimination (while loop eliminates iterator protocol overhead)
    i = 1
    @inbounds while i < n
        b[i + 1] = muladd(-dl[i], b[i], b[i + 1])
        i += 1
    end

    # Backward substitution (while loop for StepRange overhead elimination)
    @inbounds b[n] *= inv_d[n]
    i = n - 1
    @inbounds while i >= 1
        b[i] = muladd(-du[i], b[i + 1], b[i]) * inv_d[i]
        i -= 1
    end

    return b
end

# ========================================
# Batch Thomas Solver - Dimension-Aware Dispatch
# ========================================
#
# Design: _ldiv_along_dim!(z, lu, inv_d, Val{D}) dispatches based on:
#   1. Array dimensionality (Matrix vs 3D Array vs ...)
#   2. Solve dimension D (which axis contains the tridiagonal systems)
#
# Memory layout matters for SIMD:
#   - Julia is column-major: z[i,j,k] → i varies fastest
#   - SIMD works best when inner loop accesses contiguous memory
#   - Solving along axis 2 for Matrix{T}: axis 1 is contiguous → SIMD-friendly
#
# Extension points (not yet implemented):
#   - 3D Array, Val{2}: reshape (n1, n2, n3) → (n1*n3, n2), solve, reshape back
#   - 3D Array, Val{3}: loop over (n1, n2) slices, each is length-n3 system

"""
    _ldiv_along_dim!(z::AbstractMatrix, lu, inv_d, Val{2}) -> z

Batch Thomas solver for 2D matrix: solve tridiagonal systems along axis 2.

Each row `z[i, :]` is an independent RHS vector. Solves all rows simultaneously
using SIMD vectorization along the contiguous axis 1.

# Memory Layout (column-major)
```
z[n_batch, n_sys] where:
  axis 1 (n_batch): contiguous in memory → SIMD inner loop
  axis 2 (n_sys):   tridiagonal system dimension
```

# Performance
- O(n_batch × n_sys) operations
- SIMD vectorization on axis 1 (contiguous)
- Zero allocations
"""
@inline function _ldiv_along_dim!(
    z::AbstractMatrix{T},
    lu::LinearAlgebra.LU{T,Tridiagonal{T,V},P},
    inv_d::AbstractVector{T},
    ::Val{2},
) where {T<:AbstractFloat, V<:AbstractVector{T}, P}
    dl = lu.factors.dl
    du = lu.factors.du
    n_sys = length(inv_d)   # System size (axis 2 length)
    n_batch = size(z, 1)    # Batch size (axis 1 length, contiguous)

    # Forward substitution: outer loop over system step, inner loop SIMD
    @inbounds for k in 2:n_sys
        factor = -dl[k - 1]
        @simd for i in 1:n_batch
            z[i, k] = muladd(factor, z[i, k - 1], z[i, k])
        end
    end

    # Backward substitution: final column
    inv_d_n = inv_d[n_sys]
    @inbounds @simd for i in 1:n_batch
        z[i, n_sys] *= inv_d_n
    end

    # Backward substitution: remaining columns
    @inbounds for k in (n_sys - 1):-1:1
        u_factor = -du[k]
        d_factor = inv_d[k]
        @simd for i in 1:n_batch
            z[i, k] = muladd(u_factor, z[i, k + 1], z[i, k]) * d_factor
        end
    end

    return z
end

"""
    _ldiv_along_dim!(z::AbstractMatrix, lu, inv_d, Val{1})

Not supported: solving along axis 1 for matrices.

Axis 1 is the contiguous dimension in column-major layout. Solving along it
would require strided access defeating SIMD. Instead, use sequential calls
to `_ldiv_tridiagonal_nopiv!` for each column.
"""
@noinline function _ldiv_along_dim!(
    ::AbstractMatrix, ::LinearAlgebra.LU, ::AbstractVector, ::Val{1}
)
    throw(ArgumentError(
        "Batch solving along axis 1 (Val{1}) is not supported for matrices.\n" *
        "Axis 1 is contiguous in column-major layout; solving along it defeats SIMD.\n" *
        "Use per-column calls to _ldiv_tridiagonal_nopiv! instead."
    ))
end

# ========================================
# Future: 3D Array Support (placeholder docstrings)
# ========================================
#
# When ND interpolation needs 3D support, add these methods:
#
# _ldiv_along_dim!(z::AbstractArray{T,3}, lu, inv_d, Val{2}):
#   Solve along axis 2 for 3D array z[n1, n2, n3].
#   Strategy: reshape to (n1*n3, n2) matrix, solve with Val{2}, reshape back.
#   - n1*n3 independent systems, each of size n2
#   - SIMD on axis 1 (n1 elements per slice)
#
# _ldiv_along_dim!(z::AbstractArray{T,3}, lu, inv_d, Val{3}):
#   Solve along axis 3 for 3D array z[n1, n2, n3].
#   Strategy: loop over (i,j) ∈ (1:n1, 1:n2), solve each z[i,j,:] vector.
#   - n1*n2 independent systems, each of size n3
#   - Alternative: permutedims to make axis 3 contiguous, but allocation cost
