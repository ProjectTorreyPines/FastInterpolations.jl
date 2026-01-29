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
# Design: ThomasFactorization stores dl, du, inv_d for zero-allocation hot loops.
# Backward-compatible methods for LinearAlgebra.LU are provided for testing.

# ========================================
# ThomasFactorization Type
# ========================================

"""
    ThomasFactorization{T<:AbstractFloat, V<:AbstractVector{T}}

Lightweight LU factorization for tridiagonal matrices using Thomas algorithm.
Zero-allocation alternative to `LinearAlgebra.LU` for diagonally-dominant systems.

# Fields
- `dl::V`: Lower diagonal - L multipliers after factorization (n-1 elements)
- `du::V`: Upper diagonal - unchanged from input (n-1 elements)
- `inv_d::V`: Inverse U diagonal - 1/d[i] for fast back-substitution (n elements)

# Memory Layout
For n×n tridiagonal system:
- `dl`: n-1 elements (L multipliers)
- `du`: n-1 elements (U super-diagonal, unchanged)
- `inv_d`: n elements (1/U diagonal for backward substitution)

Total: 3n-2 elements. Compared to `LinearAlgebra.LU` which stores dl, d, du, ipiv, info.

# Thread Safety
Immutable after construction. Safe for concurrent reads from multiple tasks.

# Example
```julia
# Allocate persistent arrays (NOT from pool!)
dl = Vector{Float64}(undef, n-1)
d = Vector{Float64}(undef, n)
du = Vector{Float64}(undef, n-1)

# Fill with tridiagonal entries...

# One-pass factorization: d becomes inv_d
thomas = thomas_factorize!(dl, d, du)

# Solve Ax = b in-place
_ldiv_tridiagonal_nopiv!(b, thomas)
```
"""
struct ThomasFactorization{T<:AbstractFloat, V<:AbstractVector{T}}
    dl::V     # Lower diagonal (L multipliers after factorization)
    du::V     # Upper diagonal (unchanged from input)
    inv_d::V  # Inverse of U diagonal (1/d[i])
end

# ========================================
# ThomasFactorization Constructor
# ========================================

"""
    thomas_factorize!(dl, d, du) -> ThomasFactorization

In-place Thomas algorithm factorization. Computes L, U, and inv_d in ONE PASS.

# Arguments
- `dl::AbstractVector{T}`: Lower diagonal (n-1), modified in-place to store L multipliers
- `d::AbstractVector{T}`: Main diagonal (n), modified in-place to store 1/U diagonal (inv_d)
- `du::AbstractVector{T}`: Upper diagonal (n-1), unchanged

# Returns
`ThomasFactorization(dl, du, d)` where `d` now contains `inv_d`.

# Algorithm (Thomas/TDMA)
For i = 1 to n:
    inv_d[i] = 1/d[i]
    if i < n:
        l[i] = dl[i] * inv_d[i]           # L multiplier
        d[i+1] = d[i+1] - l[i] * du[i]    # U diagonal update

# Safety
This function assumes the matrix is **diagonally dominant** (no pivoting required).
For non-diagonally-dominant matrices, use `LinearAlgebra.lu` with pivoting.

# Performance
- O(n) time and space
- Zero allocations (modifies inputs in-place)
- Single pass through data (cache-friendly)

# Example
```julia
n = 100
dl = rand(n-1)
d = 4.0 .+ rand(n)  # Diagonally dominant
du = rand(n-1)

thomas = thomas_factorize!(dl, d, du)
# dl now contains L multipliers
# d now contains inv_d (aliased as thomas.inv_d)
```
"""
function thomas_factorize!(
    dl::V, d::V, du::V
) where {T<:AbstractFloat, V<:AbstractVector{T}}
    n = length(d)

    @inbounds begin
        # First diagonal element: compute inverse
        inv_d_val = inv(d[1])
        d[1] = inv_d_val

        # Forward elimination with simultaneous inverse computation
        for i in 1:n-1
            # L factor: l_i = dl[i] / d_i (use pre-computed inverse)
            l_val = dl[i] * inv_d_val
            dl[i] = l_val

            # U diagonal update: d_{i+1} = d_{i+1} - l_i * du[i]
            d_next = d[i+1] - l_val * du[i]

            # Compute inverse for next step and final storage
            inv_d_val = inv(d_next)
            d[i+1] = inv_d_val
        end
    end

    return ThomasFactorization(dl, du, d)
end

# ========================================
# Scalar Thomas Solver - ThomasFactorization (Primary)
# ========================================

"""
    _ldiv_tridiagonal_nopiv!(b, thomas::ThomasFactorization) -> b

Fast Thomas algorithm solver using `ThomasFactorization`.

Solves `Ax = b` in-place where `A = L*U` is the pre-computed factorization.

# Arguments
- `b::AbstractVector{T}`: RHS vector (modified in-place to hold solution)
- `thomas::ThomasFactorization{T,V}`: Pre-computed factorization with dl, du, inv_d

# Algorithm
1. **Forward elimination**: `b[i+1] -= L[i] * b[i]` for i = 1:n-1
2. **Backward substitution**: `b[i] = (b[i] - U[i] * b[i+1]) * inv_d[i]` for i = n:-1:1

# Performance
- O(n) time complexity
- Zero allocations in hot path
- Uses `muladd` for fused multiply-add when available
"""
@inline function _ldiv_tridiagonal_nopiv!(
    b::AbstractVector{T},
    thomas::ThomasFactorization{T,V},
) where {T<:AbstractFloat, V<:AbstractVector{T}}
    dl = thomas.dl
    du = thomas.du
    inv_d = thomas.inv_d

    n = length(inv_d)

    # Forward elimination
    @inbounds for i in 1:n-1
        b[i + 1] = muladd(-dl[i], b[i], b[i + 1])
    end

    # Backward substitution
    @inbounds b[n] *= inv_d[n]
    @inbounds for i in n-1:-1:1
        b[i] = muladd(-du[i], b[i + 1], b[i]) * inv_d[i]
    end

    return b
end

# ========================================
# Scalar Thomas Solver - LinearAlgebra.LU (Backward Compatible)
# ========================================

"""
    _ldiv_tridiagonal_nopiv!(b, lu_factor::LinearAlgebra.LU, inv_d) -> b

Backward-compatible solver accepting `LinearAlgebra.LU` objects.

# Deprecation Notice
This method exists for backward compatibility with existing tests.
Prefer using `ThomasFactorization` directly for better performance.
"""
@inline function _ldiv_tridiagonal_nopiv!(
    b::AbstractVector{T},
    lu_factor::LinearAlgebra.LU{T,Tridiagonal{T,V},P},
    inv_d::AbstractVector{T},
) where {T<:AbstractFloat, V<:AbstractVector{T}, P}
    dl = lu_factor.factors.dl
    du = lu_factor.factors.du

    n = length(inv_d)

    # Forward elimination
    @inbounds for i in 1:n-1
        b[i + 1] = muladd(-dl[i], b[i], b[i + 1])
    end

    # Backward substitution
    @inbounds b[n] *= inv_d[n]
    @inbounds for i in n-1:-1:1
        b[i] = muladd(-du[i], b[i + 1], b[i]) * inv_d[i]
    end

    return b
end

# ========================================
# Batch Thomas Solver - ThomasFactorization (Primary)
# ========================================
#
# Design: _ldiv_along_dim!(z, thomas, Val{D}) dispatches based on:
#   1. Array dimensionality (Matrix vs 3D Array vs ...)
#   2. Solve dimension D (which axis contains the tridiagonal systems)
#
# Memory layout matters for SIMD:
#   - Julia is column-major: z[i,j,k] → i varies fastest
#   - SIMD works best when inner loop accesses contiguous memory
#   - Solving along axis 2 for Matrix{T}: axis 1 is contiguous → SIMD-friendly

"""
    _ldiv_along_dim!(z::AbstractMatrix, thomas::ThomasFactorization, Val{2}) -> z

Batch Thomas solver for 2D matrix using `ThomasFactorization`.

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
    thomas::ThomasFactorization{T,V},
    ::Val{2},
) where {T<:AbstractFloat, V<:AbstractVector{T}}
    dl = thomas.dl
    du = thomas.du
    inv_d = thomas.inv_d
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
    _ldiv_along_dim!(z::AbstractMatrix, thomas::ThomasFactorization, Val{1})

Not supported: solving along axis 1 for matrices.

Axis 1 is the contiguous dimension in column-major layout. Solving along it
would require strided access defeating SIMD. Instead, use sequential calls
to `_ldiv_tridiagonal_nopiv!` for each column.
"""
@noinline function _ldiv_along_dim!(
    ::AbstractMatrix, ::ThomasFactorization, ::Val{1}
)
    throw(ArgumentError(
        "Batch solving along axis 1 (Val{1}) is not supported for matrices.\n" *
        "Axis 1 is contiguous in column-major layout; solving along it defeats SIMD.\n" *
        "Use per-column calls to _ldiv_tridiagonal_nopiv! instead."
    ))
end

# ========================================
# Batch Thomas Solver - LinearAlgebra.LU (Backward Compatible)
# ========================================

"""
    _ldiv_along_dim!(z::AbstractMatrix, lu::LinearAlgebra.LU, inv_d, Val{2}) -> z

Backward-compatible batch solver accepting `LinearAlgebra.LU` objects.
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
    _ldiv_along_dim!(z::AbstractMatrix, lu::LinearAlgebra.LU, inv_d, Val{1})

Not supported: solving along axis 1 for matrices.
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
# _ldiv_along_dim!(z::AbstractArray{T,3}, thomas, Val{2}):
#   Solve along axis 2 for 3D array z[n1, n2, n3].
#   Strategy: reshape to (n1*n3, n2) matrix, solve with Val{2}, reshape back.
#   - n1*n3 independent systems, each of size n2
#   - SIMD on axis 1 (n1 elements per slice)
#
# _ldiv_along_dim!(z::AbstractArray{T,3}, thomas, Val{3}):
#   Solve along axis 3 for 3D array z[n1, n2, n3].
#   Strategy: loop over (i,j) ∈ (1:n1, 1:n2), solve each z[i,j,:] vector.
#   - n1*n2 independent systems, each of size n3
#   - Alternative: permutedims to make axis 3 contiguous, but allocation cost
