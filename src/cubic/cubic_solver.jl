# ========================================
# Cubic Spline System Builders and Solvers
# ========================================
# Internal functions for building cache and solving tridiagonal systems.
# Include order: bc_types.jl → cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

# BC types and normalization functions are defined in bc_types.jl

# ========================================
# Row Builders for Generic BC (Type Dispatch)
# ========================================
# Note: Row builders only set matrix structure (grid-only operations).
# BC types are Type-Free (no Tv parameter on abstract types).

# Row builders take the cached axis directly. `_get_h(x, idx)` reads cached
# `h[idx]` from `_CachedRange`/`_CachedVector`/`_ExclusivePeriodicAxis` — the
# grid is the single source of truth for cell widths.

# First row - Deriv2 (second derivative specified): z[1] = bc.val
@inline function _set_first_row!(
        d_diag::AbstractVector{Tg}, du::AbstractVector{Tg}, ::Deriv2, ::AbstractVector{Tg}
    ) where {Tg}
    d_diag[1] = one(Tg)
    du[1] = zero(Tg)
    return nothing
end

# First row - Deriv1 (first derivative specified): 2h₁z₁ + h₁z₂ = 6[(y₂-y₁)/h₁ - S'(x₁)]
@inline function _set_first_row!(
        d_diag::AbstractVector{Tg}, du::AbstractVector{Tg}, ::Deriv1, x::AbstractVector{Tg}
    ) where {Tg}
    h1 = _get_h(x, 1)
    d_diag[1] = 2 * h1
    du[1] = h1
    return nothing
end

# Last row - Deriv2 (second derivative specified): matrix row enforces z[end] = bc.val
@inline function _set_last_row!(
        dl::AbstractVector{Tg}, d_diag::AbstractVector{Tg}, ::Deriv2, ::AbstractVector{Tg}
    ) where {Tg}
    dl[end] = zero(Tg)
    d_diag[end] = one(Tg)
    return nothing
end

# Last row - Deriv1 (first derivative specified): hₙzₙ + 2hₙzₙ₊₁ = 6[S'(xₙ₊₁) - (yₙ₊₁-yₙ)/hₙ]
@inline function _set_last_row!(
        dl::AbstractVector{Tg}, d_diag::AbstractVector{Tg}, ::Deriv1, x::AbstractVector{Tg}
    ) where {Tg}
    # `length(dl) == n_cells` for the n+1-row matrix; `_get_h(x, n)` is the last cell.
    n = length(dl)
    h_n = _get_h(x, n)
    dl[end] = h_n
    d_diag[end] = 2 * h_n
    return nothing
end

# First row - Deriv3 (third derivative specified): (z[2] - z[1]) / h[1] = bc.val
# Rearranged: -z[1] + z[2] = h[1] * val
@inline function _set_first_row!(
        d_diag::AbstractVector{Tg}, du::AbstractVector{Tg}, ::Deriv3, ::AbstractVector{Tg}
    ) where {Tg}
    d_diag[1] = -one(Tg)
    du[1] = one(Tg)
    return nothing
end

# Last row - Deriv3 (third derivative specified): (z[n+1] - z[n]) / h[n] = bc.val
# Rearranged: -z[n] + z[n+1] = h[n] * val
@inline function _set_last_row!(
        dl::AbstractVector{Tg}, d_diag::AbstractVector{Tg}, ::Deriv3, ::AbstractVector{Tg}
    ) where {Tg}
    dl[end] = -one(Tg)
    d_diag[end] = one(Tg)
    return nothing
end

# First/Last row - PolyFit{D} (auto-estimated first derivative): same matrix
# structure as Deriv1. The derivative value is computed from data in RHS.
@inline function _set_first_row!(
        d_diag::AbstractVector{Tg}, du::AbstractVector{Tg}, ::PolyFit{D}, x::AbstractVector{Tg}
    ) where {D, Tg}
    h1 = _get_h(x, 1)
    d_diag[1] = 2 * h1
    du[1] = h1
    return nothing
end

@inline function _set_last_row!(
        dl::AbstractVector{Tg}, d_diag::AbstractVector{Tg}, ::PolyFit{D}, x::AbstractVector{Tg}
    ) where {D, Tg}
    n = length(dl)
    h_n = _get_h(x, n)
    dl[end] = h_n
    d_diag[end] = 2 * h_n
    return nothing
end

# ========================================
# Cache Builders
# ========================================

"""
Build cache for periodic cubic spline using Sherman-Morrison formula.

The axis is wrapped via `_caching_axis(x, bc, T)`:
- `:inclusive` → `_CachedRange`/`_CachedVector` (length n+1, user-supplied closed cycle)
- `:exclusive` → `_ExclusivePeriodicAxis(...)` (virtual length n+1, raw n-cell inner)

In both forms, `length(cache_x) - 1 == n_cells` and `_get_h(cache_x, idx)`
returns the cell width (interior on inclusive, seam-aware on exclusive).
The seam-cell positivity check that previously lived here is now enforced
by the `_ExclusivePeriodicAxis` constructor.
"""
function _build_periodic_cache(x::AbstractVector{T}, bc::PeriodicBC) where {T}
    cache_x = _caching_axis(x, bc, T)
    n = length(cache_x) - 1   # n_cells (uniform across :inclusive / :exclusive)

    n >= 3 || throw(ArgumentError("Periodic spline requires at least 3 cells (length(x) >= 4 for inclusive, >= 3 for exclusive)"))

    h_n = _get_h(cache_x, n)   # seam (exclusive) or last real cell (inclusive)

    # Build modified tridiagonal matrix A' for Sherman-Morrison.
    # CRITICAL: Use Vector allocation (NOT pool!) for persistent arrays.
    dl = Vector{T}(undef, n - 1)
    d_diag = Vector{T}(undef, n)  # becomes inv_d after factorization
    du = Vector{T}(undef, n - 1)

    h_1 = _get_h(cache_x, 1)
    d_diag[1] = h_n + 2 * h_1

    @inbounds for i in 2:(n - 1)
        h_im1 = _get_h(cache_x, i - 1)
        h_i = _get_h(cache_x, i)
        dl[i - 1] = h_im1
        d_diag[i] = 2 * (h_im1 + h_i)
        du[i - 1] = h_i
    end

    h_nm1 = _get_h(cache_x, n - 1)
    dl[n - 1] = h_nm1
    d_diag[n] = 2 * h_nm1 + h_n

    if n > 1
        du[n - 1] = h_nm1
    end

    thomas = thomas_factorize!(dl, d_diag, du)

    # Pre-compute q = A'⁻¹ · u (u = [1, 0, …, 0, 1]ᵀ).
    q = Vector{T}(undef, n)
    fill!(q, zero(T))
    q[1] = one(T)
    q[n] = one(T)
    _ldiv_tridiagonal_nopiv!(q, thomas)

    # Persist resolved-period bc on the cache so display / cache-pool comparison
    # / `_with_resolved_period(itp.bc, cache.bc.period)` work without a separate
    # period field. `:inclusive` reads period from the extended grid span;
    # `:exclusive` from the user (or auto-inferred for Range).
    period_resolved = bc isa PeriodicBC{:exclusive} ?
        _resolve_seam_period(x, bc) :
        last(cache_x) - first(cache_x)
    bc_resolved = _with_resolved_period(bc, period_resolved)

    return CubicSplineCache(cache_x, bc_resolved, thomas, q)
end

# Helper to get polynomial degree for PolyFit validation
# Returns D for PolyFit{D}, 0 for other PointBC types (no extra point requirement)
_polyfit_degree(::PolyFit{D}) where {D} = D
_polyfit_degree(::PointBC) = 0

"""
Build cache for generic derivative BC (Deriv1/Deriv2 combinations).
Uses type dispatch for zero-overhead specialization.
"""
function _build_derivative_bc_cache(
        x::AbstractVector{T},
        left_bc::L,
        right_bc::R
    ) where {T, L <: PointBC, R <: PointBC}
    cache_x = _caching_axis(x, NoBC(), T)   # `_CachedRange`/`_CachedVector` for non-periodic
    n = length(cache_x) - 1

    # Validate PolyFit requirements: PolyFit{D} requires D+1 points
    max_degree = max(_polyfit_degree(left_bc), _polyfit_degree(right_bc))
    if max_degree > 0
        min_points = max_degree + 1
        length(cache_x) >= min_points || throw(
            ArgumentError(
                "PolyFit{$max_degree} requires at least $min_points data points (got $(length(cache_x))). " *
                    "A degree-$max_degree polynomial needs $(min_points) points to estimate endpoint derivatives."
            )
        )
    end

    # Build tridiagonal matrix A
    # CRITICAL: Use Vector allocation (NOT pool!) for persistent arrays
    # These arrays are stored in CubicSplineCache which outlives the function.
    dl = Vector{T}(undef, n)       # Lower diagonal (n elements)
    d_diag = Vector{T}(undef, n + 1) # Main diagonal (n+1 elements) → becomes inv_d
    du = Vector{T}(undef, n)       # Upper diagonal (n elements)

    # First and last rows depend on BC type (type dispatch)
    _set_first_row!(d_diag, du, left_bc, cache_x)
    _set_last_row!(dl, d_diag, right_bc, cache_x)

    # Interior rows (same for all BC types)
    @inbounds for i in 2:n
        h_im1 = _get_h(cache_x, i - 1)
        h_i = _get_h(cache_x, i)
        dl[i - 1] = h_im1
        d_diag[i] = 2 * (h_im1 + h_i)
        du[i] = h_i
    end

    # ONE-PASS Thomas factorization: d_diag becomes inv_d
    thomas = thomas_factorize!(dl, d_diag, du)

    bc_pair = BCPair(left_bc, right_bc)

    # Empty `q` for non-periodic (Sherman-Morrison vector unused).
    return CubicSplineCache(cache_x, bc_pair, thomas, Vector{T}())
end

# ========================================
# RHS Computation
# ========================================

# ----------------------------------------
# RHS helpers for generic BC (type dispatch)
# ----------------------------------------
# `x` is the cached axis — `_get_h(x, i)` / `_get_inv_h(x, i)` read cached
# spacing directly. PolyFit{D} BCs use `x` + `y` for endpoint derivative
# estimation; other BC types ignore `x`.

# First element - Deriv2: d[1] = bc.val (second derivative value)
@inline function _compute_rhs_first!(
        d::AbstractVector, bc::Deriv2, ::AbstractVector, ::AbstractVector{Tg}
    ) where {Tg}
    d[1] = convert(eltype(d), bc.val)
    return nothing
end

# First element - Deriv1: d[1] = 6[(y₂-y₁)/h₁ - S'(x₁)]
@inline function _compute_rhs_first!(
        d::AbstractVector, bc::Deriv1, y::AbstractVector, x::AbstractVector{Tg}
    ) where {Tg}
    d[1] = 6 * ((y[2] - y[1]) * _get_inv_h(x, 1) - convert(eltype(d), bc.val))
    return nothing
end

# Last element - Deriv2: d[end] = bc.val (second derivative value)
@inline function _compute_rhs_last!(
        d::AbstractVector, bc::Deriv2, ::AbstractVector, ::AbstractVector{Tg}
    ) where {Tg}
    d[end] = convert(eltype(d), bc.val)
    return nothing
end

# Last element - Deriv1: d[end] = 6[S'(x_end) - (y_end - y_{end-1}) / h_end]
@inline function _compute_rhs_last!(
        d::AbstractVector, bc::Deriv1, y::AbstractVector, x::AbstractVector{Tg}
    ) where {Tg}
    n = length(y) - 1
    d[end] = 6 * (convert(eltype(d), bc.val) - (y[end] - y[end - 1]) * _get_inv_h(x, n))
    return nothing
end

# First element - Deriv3: d[1] = h[1] * bc.val (from -z[1] + z[2] = h[1] * val)
@inline function _compute_rhs_first!(
        d::AbstractVector, bc::Deriv3, ::AbstractVector, x::AbstractVector{Tg}
    ) where {Tg}
    d[1] = _get_h(x, 1) * convert(eltype(d), bc.val)
    return nothing
end

# Last element - Deriv3: d[end] = h[n] * bc.val (from -z[n] + z[n+1] = h[n] * val)
@inline function _compute_rhs_last!(
        d::AbstractVector, bc::Deriv3, ::AbstractVector, x::AbstractVector{Tg}
    ) where {Tg}
    n = length(d) - 1
    d[end] = _get_h(x, n) * convert(eltype(d), bc.val)
    return nothing
end

# Generic PolyFit{D} (LinearFit/QuadraticFit/CubicFit/...): materialize to
# Deriv1 using estimated derivative, then delegate to the Deriv1 path.
@inline function _compute_rhs_first!(
        d::AbstractVector, bc::PolyFit{D}, y::AbstractVector, x::AbstractVector{Tg}
    ) where {D, Tg}
    concrete_bc = materialize_bc(bc, x, y, LeftSide())
    _compute_rhs_first!(d, concrete_bc, y, x)
    return nothing
end

@inline function _compute_rhs_last!(
        d::AbstractVector, bc::PolyFit{D}, y::AbstractVector, x::AbstractVector{Tg}
    ) where {D, Tg}
    concrete_bc = materialize_bc(bc, x, y, RightSide())
    _compute_rhs_last!(d, concrete_bc, y, x)
    return nothing
end

"""
Compute RHS vector for generic derivative BC system in-place.

`x` is the cached axis (single source of truth for spacing). Tg = grid type,
Tv = value type (y, d). PolyFit{D} BCs use both `x` and `y` for endpoint
derivative estimation.
"""
@inline function compute_rhs!(
        d::AbstractVector, y::AbstractVector, x::AbstractVector{Tg},
        bc_pair::BCPair{L, R}
    ) where {Tg, L <: PointBC, R <: PointBC}
    n = length(y) - 1
    _compute_rhs_first!(d, bc_pair.left, y, x)
    @inbounds for i in 2:n
        d[i] = 6 * ((y[i + 1] - y[i]) * _get_inv_h(x, i) - (y[i] - y[i - 1]) * _get_inv_h(x, i - 1))
    end
    _compute_rhs_last!(d, bc_pair.right, y, x)
    return nothing
end

# ----------------------------------------
# Periodic RHS function
# ----------------------------------------

"""
    compute_rhs_periodic!(d, y, x) -> nothing

Compute the n_cells-length RHS vector for the periodic cubic spline system.

`y` and `x` both report virtual length `n_cells + 1`:
- `:inclusive`: user's closed-cycle grid + values (length n+1, `y[1] == y[n+1]`).
- `:exclusive`: `_ExclusivePeriodicAxis` wrapper for `x` + `_ExclusivePeriodicData`
  wrapper for `y` (raw n inner; `y[n+1]` auto-cycles to `y[1]`).

The seam-cell formula is uniform: `_get_inv_h(x, n)` returns the last-cell
inverse width — real cell for `:inclusive`, virtual seam for `:exclusive`.
Interior rows (2 .. n-1) reference only real (non-seam) cells.
"""
@inline function compute_rhs_periodic!(
        d::AbstractVector, y::AbstractVector, x::AbstractVector{Tg}
    ) where {Tg}
    n = length(y) - 1   # n_cells (uniform)

    @inbounds d[1] = 6 * (y[2] - y[1]) * _get_inv_h(x, 1) - 6 * (y[1] - y[n]) * _get_inv_h(x, n)

    @inbounds for i in 2:(n - 1)
        d[i] = 6 * (y[i + 1] - y[i]) * _get_inv_h(x, i) - 6 * (y[i] - y[i - 1]) * _get_inv_h(x, i - 1)
    end

    @inbounds d[n] = 6 * (y[n + 1] - y[n]) * _get_inv_h(x, n) - 6 * (y[n] - y[n - 1]) * _get_inv_h(x, n - 1)

    return nothing
end

# ========================================
# System Solvers
# ========================================
# Scalar Thomas solver moved to: core/thomas_lu_solver.jl
# - _ldiv_tridiagonal_nopiv!

"""
Solve periodic cyclic tridiagonal system using Sherman-Morrison formula.

`y` is virtual length `n_cells + 1` (closed-cycle for `:inclusive`,
`_ExclusivePeriodicData` wrap for `:exclusive`). The seam-cell width comes
from `_get_h(cache.x, n_cells)` — wrapper handles `:exclusive` virtual seam,
inclusive returns the last real cell. `q` and `z_workspace` are length
`n_cells`; the inclusive caller mirrors `z[end] = z[1]` after the solve so
eval can read `z[n+1]` at the closed-cycle endpoint.
"""
@inline function _solve_cubic_system_periodic!(
        z_workspace::AbstractVector,
        y_temp::AbstractVector,
        cache::CubicSplineCache{Tg, X, F, <:PeriodicBC},
        y::AbstractVector
    ) where {Tg, X, F}
    q = cache.q
    n = length(q)   # n_cells

    compute_rhs_periodic!(y_temp, y, cache.x)
    _ldiv_tridiagonal_nopiv!(y_temp, cache.thomas)

    α = _get_h(cache.x, n)   # seam-cell width (last real cell or virtual seam)

    vTy = α * (y_temp[1] + y_temp[n])
    vTq = α * (q[1] + q[n])

    denom = one(Tg) + vTq
    # Defensive check: unreachable under valid inputs (denom ≥ √3 for SPD systems),
    # but guards against corrupted data (NaN/Inf from invalid grid spacing).
    tol = sqrt(eps(Tg))
    if !isfinite(denom) || abs(denom) < tol
        throw(
            DomainError(
                denom,
                "Sherman-Morrison formula failed: denominator (1 + v'q) ≈ 0 or non-finite.\n" *
                    "  denom = $denom (tol = $tol), α = $α, q[1] = $(q[1]), q[n] = $(q[n])\n" *
                    "  This usually indicates corrupted input data (NaN/Inf) or degenerate grid."
            )
        )
    end
    factor = vTy * inv(denom)

    @inbounds for i in 1:n
        z_workspace[i] = y_temp[i] - factor * q[i]
    end

    # `:inclusive` eval may read `z[n+1]` (closed-cycle endpoint) — mirror.
    # `:exclusive` eval folds `idx_R = 1` at seam (wrapper specialization), so
    # `z[n+1]` is never accessed; the mirror is still safe (z is length n+1
    # always, callers allocate with cache.x length).
    _finalize_z_periodic_seam!(z_workspace, cache.bc)

    return z_workspace
end

@inline _finalize_z_periodic_seam!(z::AbstractVector, ::PeriodicBC{:inclusive}) =
    (@inbounds z[end] = z[1]; nothing)
@inline _finalize_z_periodic_seam!(z::AbstractVector, ::PeriodicBC{:exclusive}) =
    (@inbounds z[end] = z[1]; nothing)

# ========================================
# Unified System Solver Entry Point
# ========================================
#
# All solvers now require explicit output argument (out_z) and use @with_pool
# for thread-safe workspace allocation. The 3-arg versions (using cache workspaces)
# have been removed as part of thread-safety refactoring.

"""
Solve cubic spline system using cached Thomas factorization.

Dispatch on `cache.bc::AbstractBC`:
- `BCPair{L,R}` (derivative BC): direct Thomas back-substitution.
- `PeriodicBC{:inclusive}` / `PeriodicBC{:exclusive}`: Sherman-Morrison via
  `_solve_cubic_system_periodic!` (pool-allocated `y_temp`).

The 4th `bc` argument is redundant with `cache.bc` but kept for API
compatibility (callers pass `cache.bc` explicitly or a fresh BC for one-shot).
"""
@inline function _solve_system!(
        out_z::AbstractVector,
        cache::CubicSplineCache{Tg, X, F, <:BCPair},
        y::AbstractVector,
        bc_pair::BCPair{L, R}
    ) where {Tg, X, F, L <: PointBC, R <: PointBC}
    compute_rhs!(out_z, y, cache.x, bc_pair)
    _ldiv_tridiagonal_nopiv!(out_z, cache.thomas)
    return out_z
end

@inline @with_pool pool function _solve_system!(
        out_z::AbstractVector{Tz},
        cache::CubicSplineCache{Tg, X, F, <:PeriodicBC},
        y::AbstractVector,
        ::PeriodicBC   # API consistency with BCPair version (cache.bc is the source of truth)
    ) where {Tz, Tg, X, F}
    n = length(cache.q)   # n_cells
    y_temp = acquire!(pool, Tz, n)
    _solve_cubic_system_periodic!(out_z, y_temp, cache, y)
    return out_z
end

# Batch Thomas solver helper moved to: core/thomas_lu_solver.jl
