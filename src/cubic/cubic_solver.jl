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

# First row - Deriv2 (second derivative specified): z[1] = bc.val
@inline function _set_first_row!(
        d_diag::AbstractVector{Tg}, du::AbstractVector{Tg}, ::Deriv2, ::AbstractGridSpacing{Tg}
    ) where {Tg}
    d_diag[1] = one(Tg)
    du[1] = zero(Tg)
    return nothing
end

# First row - Deriv1 (first derivative specified): 2h₁z₁ + h₁z₂ = 6[(y₂-y₁)/h₁ - S'(x₁)]
@inline function _set_first_row!(
        d_diag::AbstractVector{Tg}, du::AbstractVector{Tg}, ::Deriv1, spacing::AbstractGridSpacing{Tg}
    ) where {Tg}
    h1 = _get_h(spacing, 1)
    d_diag[1] = 2 * h1
    du[1] = h1
    return nothing
end

# Last row - Deriv2 (second derivative specified): matrix row enforces z[end] = bc.val
@inline function _set_last_row!(
        dl::AbstractVector{Tg}, d_diag::AbstractVector{Tg}, ::Deriv2, ::AbstractGridSpacing{Tg}
    ) where {Tg}
    dl[end] = zero(Tg)
    d_diag[end] = one(Tg)
    return nothing
end

# Last row - Deriv1 (first derivative specified): hₙzₙ + 2hₙzₙ₊₁ = 6[S'(xₙ₊₁) - (yₙ₊₁-yₙ)/hₙ]
@inline function _set_last_row!(
        dl::AbstractVector{Tg}, d_diag::AbstractVector{Tg}, ::Deriv1, spacing::AbstractGridSpacing{Tg}
    ) where {Tg}
    # For the last row, we need h[n] which is the last spacing value
    # The number of spacing values is length(dl) for dl (n elements)
    n = length(dl)
    h_n = _get_h(spacing, n)
    dl[end] = h_n
    d_diag[end] = 2 * h_n
    return nothing
end

# First row - Deriv3 (third derivative specified): (z[2] - z[1]) / h[1] = bc.val
# Rearranged: -z[1] + z[2] = h[1] * val
@inline function _set_first_row!(
        d_diag::AbstractVector{Tg}, du::AbstractVector{Tg}, ::Deriv3, ::AbstractGridSpacing{Tg}
    ) where {Tg}
    d_diag[1] = -one(Tg)
    du[1] = one(Tg)
    return nothing
end

# Last row - Deriv3 (third derivative specified): (z[n+1] - z[n]) / h[n] = bc.val
# Rearranged: -z[n] + z[n+1] = h[n] * val
@inline function _set_last_row!(
        dl::AbstractVector{Tg}, d_diag::AbstractVector{Tg}, ::Deriv3, ::AbstractGridSpacing{Tg}
    ) where {Tg}
    dl[end] = -one(Tg)
    d_diag[end] = one(Tg)
    return nothing
end

# First row - PolyFit{D} (auto-estimated first derivative): same matrix structure as Deriv1
# The derivative value is computed from data in RHS, not specified by user
@inline function _set_first_row!(
        d_diag::AbstractVector{Tg}, du::AbstractVector{Tg}, ::PolyFit{D}, spacing::AbstractGridSpacing{Tg}
    ) where {D, Tg}
    h1 = _get_h(spacing, 1)
    d_diag[1] = 2 * h1
    du[1] = h1
    return nothing
end

# Last row - PolyFit{D} (auto-estimated first derivative): same matrix structure as Deriv1
@inline function _set_last_row!(
        dl::AbstractVector{Tg}, d_diag::AbstractVector{Tg}, ::PolyFit{D}, spacing::AbstractGridSpacing{Tg}
    ) where {D, Tg}
    n = length(dl)
    h_n = _get_h(spacing, n)
    dl[end] = h_n
    d_diag[end] = 2 * h_n
    return nothing
end

# ========================================
# Cache Builders
# ========================================

"""
Build cache for periodic cubic spline using Sherman-Morrison formula.

BC-aware: handles both `:inclusive` (length(x) = n+1, y[1] ≈ y[n+1]) and
`:exclusive` (length(x) = n, virtual seam between x[n] and x[1]+period).

For `:exclusive`, the spacing object only carries the n-1 interior cell widths
— the seam cell width `h_n = period - (last(x) - first(x))` is computed here
and stored in `PeriodicData.h_n` for solve-time access (see `α` in
`_solve_cubic_system_periodic!`).
"""
function _build_periodic_cache(x::AbstractVector{T}, bc::PeriodicBC) where {T}
    # Cycle length (number of cells) and seam-cell width depend on the endpoint variant.
    # Inclusive:  length(x) == n + 1, last cell width is on the spacing object.
    # Exclusive:  length(x) == n,     seam cell is virtual — width derived from period.
    if bc isa PeriodicBC{:exclusive}
        n = length(x)
        period = _resolve_seam_period(x, bc)
        h_n = period - (last(x) - first(x))
        # Seam cell width must be strictly positive — `period <= last(x) - first(x)`
        # places the virtual endpoint at or before the last grid point, which
        # is geometrically invalid (codex P2.1).
        h_n > 0 || throw(ArgumentError(
            "PeriodicBC(endpoint=:exclusive) requires period > (last(x) - first(x)). " *
                "Got period=$period, grid span=$(last(x) - first(x)), " *
                "yielding seam cell width h_n=$h_n (must be > 0)."
        ))
        E = :exclusive
    else
        n = length(x) - 1
        period = last(x) - first(x)
        # spacing built below has cell index 1..n; h_n = last spacing entry.
        E = :inclusive
        h_n = zero(T)  # placeholder — assigned after spacing is built
    end

    n >= 3 || throw(ArgumentError("Periodic spline requires at least 3 cells (length(x) >= 4 for inclusive, >= 3 for exclusive)"))

    # Create spacing object (ScalarSpacing for Range, VectorSpacing for Vector).
    # For exclusive, spacing has n-1 interior cell widths (from x[1..n] differences);
    # the seam cell width is held in `h_n` separately.
    spacing = _create_spacing(x)

    # Inclusive form fills `h_n` from the spacing (last real cell).
    # Exclusive form already computed `h_n = seam_h` above.
    if !(bc isa PeriodicBC{:exclusive})
        h_n = _get_h(spacing, n)
    end

    # Build modified tridiagonal matrix A' for Sherman-Morrison.
    # CRITICAL: Use Vector allocation (NOT pool!) for persistent arrays
    # These arrays are stored in CubicSplineCache which outlives the function.
    dl = Vector{T}(undef, n - 1)
    d_diag = Vector{T}(undef, n)  # Will become inv_d after factorization
    du = Vector{T}(undef, n - 1)

    h_1 = _get_h(spacing, 1)
    d_diag[1] = h_n + 2 * h_1

    @inbounds for i in 2:(n - 1)
        h_im1 = _get_h(spacing, i - 1)
        h_i = _get_h(spacing, i)
        dl[i - 1] = h_im1
        d_diag[i] = 2 * (h_im1 + h_i)
        du[i - 1] = h_i
    end

    h_nm1 = _get_h(spacing, n - 1)
    dl[n - 1] = h_nm1
    d_diag[n] = 2 * h_nm1 + h_n

    if n > 1
        du[n - 1] = h_nm1
    end

    # ONE-PASS Thomas factorization: d_diag becomes inv_d
    thomas = thomas_factorize!(dl, d_diag, du)

    # Pre-compute q = A'^{-1} * u using custom Thomas solver
    # u = [1, 0, ..., 0, 1]^T, solve directly into permanent q storage
    q = Vector{T}(undef, n)
    fill!(q, zero(T))
    q[1] = one(T)
    q[n] = one(T)
    _ldiv_tridiagonal_nopiv!(q, thomas)

    # Workspaces (z, y_temp) are allocated from task-local pools at solve time.
    # `E` (Symbol type-param) baked into PeriodicData so RHS dispatch is zero-cost.
    bc_config = PeriodicData{T, E}(q, period, h_n)

    return CubicSplineCache(x, spacing, thomas, bc_config)
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
    n = length(x) - 1

    # Validate PolyFit requirements: PolyFit{D} requires D+1 points
    max_degree = max(_polyfit_degree(left_bc), _polyfit_degree(right_bc))
    if max_degree > 0
        min_points = max_degree + 1
        length(x) >= min_points || throw(
            ArgumentError(
                "PolyFit{$max_degree} requires at least $min_points data points (got $(length(x))). " *
                    "A degree-$max_degree polynomial needs $(min_points) points to estimate endpoint derivatives."
            )
        )
    end

    # Create spacing object (ScalarSpacing for Range, VectorSpacing for Vector)
    spacing = _create_spacing(x)

    # Build tridiagonal matrix A
    # CRITICAL: Use Vector allocation (NOT pool!) for persistent arrays
    # These arrays are stored in CubicSplineCache which outlives the function.
    dl = Vector{T}(undef, n)       # Lower diagonal (n elements)
    d_diag = Vector{T}(undef, n + 1) # Main diagonal (n+1 elements) → becomes inv_d
    du = Vector{T}(undef, n)       # Upper diagonal (n elements)

    # First and last rows depend on BC type (type dispatch)
    _set_first_row!(d_diag, du, left_bc, spacing)
    _set_last_row!(dl, d_diag, right_bc, spacing)

    # Interior rows (same for all BC types)
    @inbounds for i in 2:n
        h_im1 = _get_h(spacing, i - 1)
        h_i = _get_h(spacing, i)
        dl[i - 1] = h_im1
        d_diag[i] = 2 * (h_im1 + h_i)
        du[i] = h_i
    end

    # ONE-PASS Thomas factorization: d_diag becomes inv_d
    thomas = thomas_factorize!(dl, d_diag, du)

    bc_config = BCPair(left_bc, right_bc)

    return CubicSplineCache(x, spacing, thomas, bc_config)
end

# ========================================
# RHS Computation
# ========================================

# ----------------------------------------
# RHS helpers for generic BC (type dispatch)
# ----------------------------------------
# All helpers now accept x parameter for consistency; existing BCs ignore it.
# CubicFit (PolyFit{3}) uses x to compute endpoint derivatives from data.
# Tg = grid type (x, spacing). d (RHS buffer) may have wider eltype than y
# when grid is duck-typed (e.g. Dual): d = _output_eltype(Tv, Tg).

# First element - Deriv2: d[1] = bc.val (second derivative value)
@inline function _compute_rhs_first!(
        d::AbstractVector, bc::Deriv2, ::AbstractVector, ::AbstractVector{Tg}, ::AbstractGridSpacing{Tg}
    ) where {Tg}
    d[1] = convert(eltype(d), bc.val)
    return nothing
end

# First element - Deriv1: d[1] = 6[(y₂-y₁)/h₁ - S'(x₁)]
@inline function _compute_rhs_first!(
        d::AbstractVector, bc::Deriv1, y::AbstractVector, ::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}
    ) where {Tg}
    d[1] = 6 * ((y[2] - y[1]) * _get_inv_h(spacing, 1) - convert(eltype(d), bc.val))
    return nothing
end

# Last element - Deriv2: d[end] = bc.val (second derivative value)
@inline function _compute_rhs_last!(
        d::AbstractVector, bc::Deriv2, ::AbstractVector, ::AbstractVector{Tg}, ::AbstractGridSpacing{Tg}
    ) where {Tg}
    d[end] = convert(eltype(d), bc.val)
    return nothing
end

# Last element - Deriv1: d[end] = 6[S'(x_end) - (y_end - y_{end-1}) / h_end]
@inline function _compute_rhs_last!(
        d::AbstractVector, bc::Deriv1, y::AbstractVector, ::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}
    ) where {Tg}
    n = length(y) - 1
    d[end] = 6 * (convert(eltype(d), bc.val) - (y[end] - y[end - 1]) * _get_inv_h(spacing, n))
    return nothing
end

# First element - Deriv3: d[1] = h[1] * bc.val (from -z[1] + z[2] = h[1] * val)
@inline function _compute_rhs_first!(
        d::AbstractVector, bc::Deriv3, ::AbstractVector, ::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}
    ) where {Tg}
    d[1] = _get_h(spacing, 1) * convert(eltype(d), bc.val)
    return nothing
end

# Last element - Deriv3: d[end] = h[n] * bc.val (from -z[n] + z[n+1] = h[n] * val)
@inline function _compute_rhs_last!(
        d::AbstractVector, bc::Deriv3, ::AbstractVector, ::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}
    ) where {Tg}
    n = length(d) - 1
    d[end] = _get_h(spacing, n) * convert(eltype(d), bc.val)
    return nothing
end

# First element - Generic PolyFit{D}: materialize to Deriv1, then delegate
# Supports all polynomial degrees: LinearFit (D=1), QuadraticFit (D=2), CubicFit (D=3), etc.
@inline function _compute_rhs_first!(
        d::AbstractVector, bc::PolyFit{D}, y::AbstractVector, x::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}
    ) where {D, Tg}
    # Materialize PolyFit{D} → Deriv1 using estimated derivative
    concrete_bc = materialize_bc(bc, x, y, LeftSide())
    # Delegate to existing Deriv1 code path
    _compute_rhs_first!(d, concrete_bc, y, x, spacing)
    return nothing
end

# Last element - Generic PolyFit{D}: materialize to Deriv1, then delegate
@inline function _compute_rhs_last!(
        d::AbstractVector, bc::PolyFit{D}, y::AbstractVector, x::AbstractVector{Tg}, spacing::AbstractGridSpacing{Tg}
    ) where {D, Tg}
    # Materialize PolyFit{D} → Deriv1 using estimated derivative
    concrete_bc = materialize_bc(bc, x, y, RightSide())
    # Delegate to existing Deriv1 code path
    _compute_rhs_last!(d, concrete_bc, y, x, spacing)
    return nothing
end

"""
Compute RHS vector for generic derivative BC system in-place.

The `x` parameter is needed for PolyFit{D} BCs (LinearFit, QuadraticFit, CubicFit, etc.)
which compute endpoint derivatives from data. For other BC types (Deriv1, Deriv2, Deriv3),
`x` is passed but ignored.

Type-Free design: BCPair{L,R} where L, R are PointBC subtypes (Deriv1{Tv}, PolyFit{D}, etc.)
Tg = grid type (x, spacing), Tv = value type (y, d)
"""
@inline function compute_rhs!(
        d::AbstractVector, y::AbstractVector, x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg}, bc_config::BCPair{L, R}
    ) where {Tg, L <: PointBC, R <: PointBC}
    n = length(y) - 1
    _compute_rhs_first!(d, bc_config.left, y, x, spacing)
    # Use spacing accessors
    @inbounds for i in 2:n
        d[i] = 6 * ((y[i + 1] - y[i]) * _get_inv_h(spacing, i) - (y[i] - y[i - 1]) * _get_inv_h(spacing, i - 1))
    end
    _compute_rhs_last!(d, bc_config.right, y, x, spacing)
    return nothing
end

# ----------------------------------------
# Periodic RHS function
# ----------------------------------------

"""
    compute_rhs_periodic!(d, y, spacing, bc_config) -> nothing

Compute the n_cells-length RHS vector for the periodic cubic spline system.

BC-aware via `bc_config` type-param dispatch (zero-cost):
- `PeriodicData{T, :inclusive}`: `length(y) == n_cells + 1` with `y[1] ≈ y[n+1]`.
  Last cell width comes from the spacing object.
- `PeriodicData{T, :exclusive}`: `length(y) == n_cells`. Seam-cell width
  (`bc_config.h_n`) bridges `y[n]` and the cyclic-wrapped `y[1]`.

Interior rows (`2:n_cells-1`) are identical between forms — they reference only
real (non-seam) cell widths.
"""
@inline function compute_rhs_periodic!(
        d::AbstractVector, y::AbstractVector,
        spacing::AbstractGridSpacing{Tg}, ::PeriodicData{Tg, :inclusive}
    ) where {Tg}
    n = length(y) - 1   # n_cells

    @inbounds d[1] = 6 * (y[2] - y[1]) * _get_inv_h(spacing, 1) - 6 * (y[1] - y[end - 1]) * _get_inv_h(spacing, n)

    @inbounds for i in 2:(n - 1)
        d[i] = 6 * (y[i + 1] - y[i]) * _get_inv_h(spacing, i) - 6 * (y[i] - y[i - 1]) * _get_inv_h(spacing, i - 1)
    end

    @inbounds d[n] = 6 * (y[end] - y[end - 1]) * _get_inv_h(spacing, n) - 6 * (y[end - 1] - y[end - 2]) * _get_inv_h(spacing, n - 1)

    return nothing
end

@inline function compute_rhs_periodic!(
        d::AbstractVector, y::AbstractVector,
        spacing::AbstractGridSpacing{Tg}, bc_config::PeriodicData{Tg, :exclusive}
    ) where {Tg}
    n = length(y)         # n_cells (raw exclusive form)
    inv_h_n = inv(bc_config.h_n)   # seam-cell width inverse

    # Cyclic indexing at the seam: previous-of-y[1] is y[n]; next-of-y[n] is y[1].
    @inbounds d[1] = 6 * (y[2] - y[1]) * _get_inv_h(spacing, 1) - 6 * (y[1] - y[n]) * inv_h_n

    @inbounds for i in 2:(n - 1)
        d[i] = 6 * (y[i + 1] - y[i]) * _get_inv_h(spacing, i) - 6 * (y[i] - y[i - 1]) * _get_inv_h(spacing, i - 1)
    end

    @inbounds d[n] = 6 * (y[1] - y[n]) * inv_h_n - 6 * (y[n] - y[n - 1]) * _get_inv_h(spacing, n - 1)

    return nothing
end

# ========================================
# System Solvers
# ========================================
# Scalar Thomas solver moved to: core/thomas_lu_solver.jl
# - _ldiv_tridiagonal_nopiv!

"""
Solve periodic cyclic tridiagonal system using Sherman-Morrison formula.

BC-aware: cycle length, seam-cell width, and final z-extension all read from
`cache.bc_config` (`PeriodicData{Tg, E}`), so this body works uniformly for
both `:inclusive` (length(y) = n+1) and `:exclusive` (length(y) = n).
"""
@inline function _solve_cubic_system_periodic!(
        z_workspace::AbstractVector,
        y_temp::AbstractVector,
        cache::CubicSplineCache{Tg, X, F, <:PeriodicData{Tg}, S},
        y::AbstractVector
    ) where {Tg, X, F, S <: AbstractGridSpacing{Tg}}
    bc_config = cache.bc_config
    q = bc_config.q
    n = length(q)   # n_cells (BC-form-agnostic: q is allocated to cycle length in builder)

    # RHS: BC-aware via `bc_config` type-param dispatch (compute_rhs_periodic!).
    compute_rhs_periodic!(y_temp, y, cache.spacing, bc_config)
    _ldiv_tridiagonal_nopiv!(y_temp, cache.thomas)

    α = bc_config.h_n   # seam-cell width baked at cache-build time

    vTy = α * (y_temp[1] + y_temp[n])
    vTq = α * (q[1] + q[n])

    denom = one(Tg) + vTq
    # Defensive check: unreachable under valid inputs (denom ≥ √3 for SPD systems),
    # but guards against corrupted data (NaN/Inf from invalid grid spacing).
    # The isfinite check catches NaN propagation since abs(NaN) < tol is always false.
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

    # Inclusive form: eval may read z[n+1] (idx_R can reach n+1 on the inclusive
    # closed-cycle grid) → mirror z[1] into the trailing slot.
    # Exclusive form: search seam dispatch returns idx_R = 1 directly (no n+1 access)
    # and `z_workspace` was allocated to length n, so leave as-is.
    _finalize_z_periodic_seam!(z_workspace, bc_config)

    return z_workspace
end

@inline _finalize_z_periodic_seam!(z::AbstractVector, ::PeriodicData{Tg, :inclusive}) where {Tg} =
    (@inbounds z[end] = z[1]; nothing)
@inline _finalize_z_periodic_seam!(::AbstractVector, ::PeriodicData{Tg, :exclusive}) where {Tg} = nothing

# ========================================
# Unified System Solver Entry Point
# ========================================
#
# All solvers now require explicit output argument (out_z) and use @with_pool
# for thread-safe workspace allocation. The 3-arg versions (using cache workspaces)
# have been removed as part of thread-safety refactoring.

"""
Solve cubic spline system (BCPair) using cached Thomas factorization.
Thread-safe: no workspace allocation needed (zero-allocation hot path).

Type-Free design:
- Cache stores BCPair{CL, CR} (typically Deriv1{Tg} from _cache_bc_pair conversion)
- bc_pair can be BCPair{L, R} with any PointBC types (including PolyFit{D})
- compute_rhs! uses bc_pair for RHS materialization (handles PolyFit automatically)
- Tg = grid type, Tv = value type (can be Complex)
"""
@inline function _solve_system!(
        out_z::AbstractVector,
        cache::CubicSplineCache{Tg, X, F, BCPair{CL, CR}, S},
        y::AbstractVector,
        bc_pair::BCPair{L, R}
    ) where {Tg, X, F, CL <: PointBC, CR <: PointBC, L <: PointBC, R <: PointBC, S <: AbstractGridSpacing{Tg}}
    compute_rhs!(out_z, y, cache.x, cache.spacing, bc_pair)
    _ldiv_tridiagonal_nopiv!(out_z, cache.thomas)
    return out_z
end

"""
Solve cubic spline system (Periodic BC) with explicit output and pool-based workspace.
Thread-safe: workspaces allocated from task-local pool.
Tg = grid type, Tv = value type (can be Complex)
"""
@inline @with_pool pool function _solve_system!(
        out_z::AbstractVector{Tz},
        cache::CubicSplineCache{Tg, X, F, <:PeriodicData{Tg}, S},
        y::AbstractVector,
        ::PeriodicData  # Unused, for API consistency with BCPair version
    ) where {Tz, Tg, X, F, S <: AbstractGridSpacing{Tg}}
    # Cycle length (number of cells) is BC-form-agnostic: q is allocated to that
    # length in the cache builder regardless of inclusive/exclusive input form.
    n = length(cache.bc_config.q)

    # Periodic workspaces need n elements (NOT length(y)!)
    # Use pool allocation for zero-allocation hot path
    y_temp = acquire!(pool, Tz, n)

    _solve_cubic_system_periodic!(out_z, y_temp, cache, y)
    return out_z
end

# Batch Thomas solver helper moved to: core/thomas_lu_solver.jl
