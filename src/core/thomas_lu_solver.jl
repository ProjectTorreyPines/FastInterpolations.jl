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
# Unit spaces: the matrix entries carry the grid unit `X`, but the
# factorization does not stay in that space — `l` is dimensionless and
# `inv_d` lives in `1/X`, and a solve maps a `B`-space RHS to a `B/X`-space
# result. The storage is therefore per-field typed and the solve writes into a
# caller-provided output; overwriting an `X`-typed buffer with those values is
# exactly the write a unit-carrying axis cannot represent. For `Real` grids
# every space collapses to the same element type and the loops below are
# op-for-op identical to the historic in-place implementation (bit-identical
# results — pinned by test_thomas_lu_solver.jl).

# ========================================
# ThomasFactorization Type
# ========================================

"""
    ThomasFactorization{Vl, Vu, Vd}

Lightweight LU factorization for tridiagonal matrices using the Thomas
algorithm. Zero-allocation alternative to `LinearAlgebra.LU` for
diagonally-dominant systems.

# Fields (one unit space per field)
- `dl::Vl`: L multipliers (n-1) — dimensionless (`dl_in[i] * inv_d[i]`)
- `du::Vu`: U super-diagonal (n-1) — grid space `X`, captured by reference
  from the assembly input, unchanged
- `inv_d::Vd`: inverse U diagonal (n) — `1/X`

For `Real` grids all three are plain `Vector{T}` — same layout the old
single-`T` struct had.

# Thread Safety
Immutable after construction. Safe for concurrent reads from multiple tasks.

# Example
```julia
dl = rand(n - 1); d = 4.0 .+ rand(n); du = rand(n - 1)
thomas = thomas_factorize(dl, d, du)   # inputs read-only
x = similar(b)
_ldiv_tridiagonal_nopiv!(x, b, thomas) # b consumed as forward-sweep scratch
_ldiv_tridiagonal_nopiv!(b, b, thomas) # alias form ≡ historic in-place solve
```
"""
struct ThomasFactorization{Vl <: AbstractVector, Vu <: AbstractVector, Vd <: AbstractVector}
    dl::Vl     # L multipliers (dimensionless)
    du::Vu     # Upper diagonal (unchanged input, grid space)
    inv_d::Vd  # Inverse of U diagonal (1/grid space)
end

# ========================================
# ThomasFactorization Constructor
# ========================================

"""
    thomas_factorize(dl, d, du) -> ThomasFactorization

One-pass Thomas factorization. Inputs are **read-only**; `l` and `inv_d` are
allocated with witness-computed element types (`_thomas_l_op`, `_inv_op`) so a
unit-carrying axis factorizes natively. `du` is captured by reference.

# Algorithm (Thomas/TDMA)
For i = 1 to n:
    inv_d[i] = 1/d[i]
    if i < n:
        l[i] = dl[i] * inv_d[i]            # L multiplier (dimensionless)
        d'[i+1] = d[i+1] - l[i] * du[i]    # U diagonal update (grid space)

The loop is op-for-op the historic in-place factorization — `Real` results
are bit-identical; only the destination of the two unit-changing writes moved
(they used to overwrite the `X`-typed inputs).

# Safety
Assumes the matrix is **diagonally dominant** (no pivoting). For general
matrices use `LinearAlgebra.lu`.

# Performance
O(n) single pass; allocates the two output vectors (build-time only — the
factorization is cached, never rebuilt on the eval hot path).
"""
function thomas_factorize(
        dl::AbstractVector, d::AbstractVector, du::AbstractVector
    )
    Tg = eltype(d)
    Tl = _promote_eltype(_thomas_l_op, Tg, Tg)
    Tinv = _promote_eltype(_inv_op, Tg)
    n = length(d)
    l = Vector{Tl}(undef, n - 1)
    inv_d = Vector{Tinv}(undef, n)

    @inbounds begin
        # First diagonal element: compute inverse
        inv_d_val = inv(d[1])
        inv_d[1] = inv_d_val

        # Forward elimination with simultaneous inverse computation
        for i in 1:(n - 1)
            # L factor: l_i = dl[i] / d'_i (use pre-computed inverse)
            l_val = dl[i] * inv_d_val
            l[i] = l_val

            # U diagonal update: d'_{i+1} = d[i+1] - l_i * du[i]
            d_next = d[i + 1] - l_val * du[i]

            # Compute inverse for next step and final storage
            inv_d_val = inv(d_next)
            inv_d[i + 1] = inv_d_val
        end
    end

    return ThomasFactorization(l, du, inv_d)
end

# ========================================
# Scalar Thomas Solver - ThomasFactorization (Primary)
# ========================================

"""
    _ldiv_tridiagonal_nopiv!(x, b, thomas::ThomasFactorization) -> x

Fast Thomas solve of `Ax = b` using the pre-computed factorization.

# Arguments
- `x::AbstractVector`: output, in `[b]/X` space (caller allocates with the
  witness eltype; for `Real` grids that is `eltype(b)`)
- `b::AbstractVector`: RHS — **consumed as scratch** by the forward sweep
  (L is dimensionless, so the sweep stays in b's space)
- `thomas`: factorization from [`thomas_factorize`](@ref)

# Aliasing
`x === b` is allowed and degenerates to the historic in-place solve
bit-for-bit (`b[i]` is read before `x[i]` is written; `x[i+1]` reads are the
intended already-substituted values). Partially-overlapping *distinct* arrays
are not supported.

# Algorithm
1. Forward elimination (in place on `b`): `b[i+1] -= l[i] * b[i]`
2. Backward substitution (into `x`): `x[i] = (b[i] - du[i] * x[i+1]) * inv_d[i]`

# Performance
O(n), zero allocations, `muladd`-fused.
"""
@inline function _ldiv_tridiagonal_nopiv!(
        x::AbstractVector,
        b::AbstractVector,
        thomas::ThomasFactorization,
    )
    dl = thomas.dl
    du = thomas.du
    inv_d = thomas.inv_d

    n = length(inv_d)

    # Forward elimination
    @inbounds for i in 1:(n - 1)
        b[i + 1] = muladd(-dl[i], b[i], b[i + 1])
    end

    # Backward substitution
    @inbounds x[n] = b[n] * inv_d[n]
    @inbounds for i in (n - 1):-1:1
        x[i] = muladd(-du[i], x[i + 1], b[i]) * inv_d[i]
    end

    return x
end

# ========================================
# Transpose Thomas Solver - ThomasFactorization
# ========================================

"""
    _ldiv_tridiagonal_transpose!(x, b, thomas::ThomasFactorization) -> x

Solve `Aᵀx = b` using the same `ThomasFactorization` as the forward solve.

Given `A = L·U` with L unit lower bidiagonal (dl) and U upper bidiagonal
(du, inv_d), `Aᵀ = Uᵀ·Lᵀ` is solved in two sweeps:

1. **Uᵀ forward sweep** (writes `x`, reads `b`):
   `x[1] = b[1] * inv_d[1]`; `x[i] = (b[i] - du[i-1] * x[i-1]) * inv_d[i]`
2. **Lᵀ backward sweep** (in place on `x`): `x[i] -= dl[i] * x[i+1]`

`b` is read-only here (unlike the non-transpose solve). Output space is
`[b]/X`. `x === b` is allowed and matches the historic in-place solve
bit-for-bit; partially-overlapping distinct arrays are not supported.
"""
@inline function _ldiv_tridiagonal_transpose!(
        x::AbstractVector,
        b::AbstractVector,
        thomas::ThomasFactorization,
    )
    dl = thomas.dl
    du = thomas.du
    inv_d = thomas.inv_d

    n = length(inv_d)

    # Step 1: Uᵀ forward sweep (lower bidiagonal with du as sub-diagonal)
    @inbounds x[1] = b[1] * inv_d[1]
    @inbounds for i in 2:n
        x[i] = muladd(-du[i - 1], x[i - 1], b[i]) * inv_d[i]
    end

    # Step 2: Lᵀ backward sweep (upper bidiagonal with dl as super-diagonal)
    @inbounds for i in (n - 1):-1:1
        x[i] = muladd(-dl[i], x[i + 1], x[i])
    end

    return x
end

# ========================================
# Batch Thomas Solver - ThomasFactorization
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

**Real-only in-place contract**: the RHS is overwritten with the solution, so
the two spaces must share one element type — the signature pins `eltype(z)`
to the factorization eltype. Unit-carrying ND builds never reach this path
(the ND PreCompute boundary refuses them).

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
        thomas::ThomasFactorization{
            <:AbstractVector{T}, <:AbstractVector{T}, <:AbstractVector{T},
        },
        ::Val{2},
    ) where {T}
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
    throw(
        ArgumentError(
            "Batch solving along axis 1 (Val{1}) is not supported for matrices.\n" *
                "Axis 1 is contiguous in column-major layout; solving along it defeats SIMD.\n" *
                "Use per-column calls to _ldiv_tridiagonal_nopiv! instead."
        )
    )
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
