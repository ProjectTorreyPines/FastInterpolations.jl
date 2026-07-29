# ========================================
# Thomas Algorithm LU Solver
# ========================================
# TDMA solvers for diagonally-dominant tridiagonal systems (cubic spline
# builds, batch ND).
#
# Unit spaces: matrix entries carry the grid unit `X`, but `l` is
# dimensionless, `inv_d` is `1/X`, and a solve maps a `B`-space RHS to `B/X` —
# so storage is per-field typed and solves write to a caller output. Real
# grids collapse every space to one eltype; the loops are op-for-op the
# historic in-place implementation (bit-identity pinned in tests).

# ========================================
# ThomasFactorization Type
# ========================================

"""
    ThomasFactorization{Vl, Vu, Vd}

Thomas LU factorization, one unit space per field:
- `dl::Vl`: L multipliers (n-1) — dimensionless
- `du::Vu`: U super-diagonal (n-1) — grid space `X`, by reference, unchanged
- `inv_d::Vd`: inverse U diagonal (n) — `1/X`

Real grids: all three are `Vector{T}` (old single-`T` layout). Immutable —
safe for concurrent reads.
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

One-pass Thomas factorization. **Inputs are donated — do not reuse after the
call.** Storage per witness space (`_thomas_storage`):
- witness eltype == input eltype (Real/Dual): donated `dl`/`d` ARE the
  `l`/`inv_d` storage — old in-place layout, zero allocation.
- differs (unit axis): fresh witness-typed outputs, inputs untouched.

`du` is captured by reference either way. Assumes diagonal dominance (no
pivoting).
"""
function thomas_factorize(
        dl::AbstractVector, d::AbstractVector, du::AbstractVector
    )
    Tg = eltype(d)
    Tl = _promote_eltype(_thomas_l_op, Tg, Tg)
    Tinv = _promote_eltype(_inv_op, Tg)
    l = _thomas_storage(dl, Tl)
    inv_d = _thomas_storage(d, Tinv)
    n = length(d)

    @inbounds begin
        # First diagonal element: compute inverse
        inv_d_val = inv(d[1])
        inv_d[1] = inv_d_val

        # Forward elimination with simultaneous inverse computation.
        # Alias-safe for the reuse arm: dl[i] / d[i+1] are read before
        # l[i] / inv_d[i+1] overwrite those slots — op-for-op the historic
        # in-place loop.
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

# Storage selector for `thomas_factorize` outputs: reuse the donated input when
# the witness space equals its eltype (Real/Dual — old in-place layout, zero
# allocation); allocate fresh when the spaces differ (unit grids). Dispatch,
# not a runtime branch — folds at specialization time.
@inline _thomas_storage(v::AbstractVector{T}, ::Type{T}) where {T} = v
@inline _thomas_storage(v::AbstractVector, ::Type{T}) where {T} =
    Vector{T}(undef, length(v))

# ========================================
# Scalar Thomas Solver - ThomasFactorization (Primary)
# ========================================

"""
    _ldiv_tridiagonal_nopiv!(x, b, thomas::ThomasFactorization) -> x

Thomas solve of `Ax = b`. `x` is the output in `[b]/X` space; `b` is
**consumed as scratch** by the forward sweep (L is dimensionless, so that
sweep stays in b's space).

Aliasing: `x === b` is allowed — it degenerates to the historic in-place
solve bit-for-bit. Partially-overlapping *distinct* arrays are not.
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

Solve `Aᵀx = b` (`Aᵀ = Uᵀ·Lᵀ`): Uᵀ sweep writes `x` from `b`, then Lᵀ sweep
in place on `x`. Output space `[b]/X`; **`b` is read-only here** (unlike the
non-transpose solve). `x === b` allowed (bit-identical to the historic
in-place solve); partial overlap is not.
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

Batch solve: each row `z[i, :]` is an independent RHS; SIMD over the
contiguous axis 1. **Real-only in-place contract** — the RHS is overwritten
with the solution, so the signature pins `eltype(z)` to the factorization
eltype (unit-carrying ND builds never reach this path; the ND PreCompute
boundary refuses them).
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
