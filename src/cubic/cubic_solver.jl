# ========================================
# Cubic Spline System Builders and Solvers
# ========================================
# Internal functions for building cache and solving tridiagonal systems.
# Include order: bc_types.jl → cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

# BC types and normalization functions are defined in bc_types.jl

# ========================================
# Row Builders for Generic BC (Type Dispatch)
# ========================================

# First row - Deriv2 (second derivative specified): z[1] = bc.val
@inline function _set_first_row!(
    d_diag::AbstractVector{T}, du::AbstractVector{T}, ::Deriv2{T}, ::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    d_diag[1] = one(T)
    du[1] = zero(T)
    return nothing
end

# First row - Deriv1 (first derivative specified): 2h₁z₁ + h₁z₂ = 6[(y₂-y₁)/h₁ - S'(x₁)]
@inline function _set_first_row!(
    d_diag::AbstractVector{T}, du::AbstractVector{T}, ::Deriv1{T}, spacing::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    h1 = _get_h(spacing, 1)
    d_diag[1] = 2 * h1
    du[1] = h1
    return nothing
end

# Last row - Deriv2 (second derivative specified): matrix row enforces z[end] = bc.val
@inline function _set_last_row!(
    dl::AbstractVector{T}, d_diag::AbstractVector{T}, ::Deriv2{T}, ::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    dl[end] = zero(T)
    d_diag[end] = one(T)
    return nothing
end

# Last row - Deriv1 (first derivative specified): hₙzₙ + 2hₙzₙ₊₁ = 6[S'(xₙ₊₁) - (yₙ₊₁-yₙ)/hₙ]
@inline function _set_last_row!(
    dl::AbstractVector{T}, d_diag::AbstractVector{T}, ::Deriv1{T}, spacing::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
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
    d_diag::AbstractVector{T}, du::AbstractVector{T}, ::Deriv3{T}, ::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    d_diag[1] = -one(T)
    du[1] = one(T)
    return nothing
end

# Last row - Deriv3 (third derivative specified): (z[n+1] - z[n]) / h[n] = bc.val
# Rearranged: -z[n] + z[n+1] = h[n] * val
@inline function _set_last_row!(
    dl::AbstractVector{T}, d_diag::AbstractVector{T}, ::Deriv3{T}, ::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    dl[end] = -one(T)
    d_diag[end] = one(T)
    return nothing
end

# First row - PolyFit{D} (auto-estimated first derivative): same matrix structure as Deriv1
# The derivative value is computed from data in RHS, not specified by user
@inline function _set_first_row!(
    d_diag::AbstractVector{T}, du::AbstractVector{T}, ::PolyFit{D, T}, spacing::AbstractGridSpacing{T}
) where {D, T<:AbstractFloat}
    h1 = _get_h(spacing, 1)
    d_diag[1] = 2 * h1
    du[1] = h1
    return nothing
end

# Last row - PolyFit{D} (auto-estimated first derivative): same matrix structure as Deriv1
@inline function _set_last_row!(
    dl::AbstractVector{T}, d_diag::AbstractVector{T}, ::PolyFit{D, T}, spacing::AbstractGridSpacing{T}
) where {D, T<:AbstractFloat}
    n = length(dl)
    h_n = _get_h(spacing, n)
    dl[end] = h_n
    d_diag[end] = 2 * h_n
    return nothing
end

# ========================================
# Cache Builders
# ========================================

"Build cache for periodic cubic spline using Sherman-Morrison formula."
function _build_periodic_cache(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x) - 1  # Number of intervals

    n >= 3 || throw(ArgumentError("Periodic spline requires at least 4 points"))

    period = last(x) - first(x)

    # Create spacing object (ScalarSpacing for Range, VectorSpacing for Vector)
    spacing = _create_spacing(x)

    # Build modified tridiagonal matrix A' for Sherman-Morrison
    # CRITICAL: Use Vector allocation (NOT pool!) for persistent arrays
    # These arrays are stored in CubicSplineCache which outlives the function.
    dl = Vector{T}(undef, n - 1)
    d_diag = Vector{T}(undef, n)  # Will become inv_d after factorization
    du = Vector{T}(undef, n - 1)

    h_n = _get_h(spacing, n)
    h_1 = _get_h(spacing, 1)
    d_diag[1] = h_n + 2 * h_1

    @inbounds for i in 2:n-1
        h_im1 = _get_h(spacing, i-1)
        h_i = _get_h(spacing, i)
        dl[i-1] = h_im1
        d_diag[i] = 2 * (h_im1 + h_i)
        du[i-1] = h_i
    end

    h_nm1 = _get_h(spacing, n-1)
    dl[n-1] = h_nm1
    d_diag[n] = 2 * h_nm1 + h_n

    if n > 1
        du[n-1] = h_nm1
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

    # Workspaces (d, z, y_temp) are allocated from task-local pools at solve time
    bc_config = PeriodicData(q, period)

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
) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    n = length(x) - 1

    # Validate PolyFit requirements: PolyFit{D} requires D+1 points
    max_degree = max(_polyfit_degree(left_bc), _polyfit_degree(right_bc))
    if max_degree > 0
        min_points = max_degree + 1
        length(x) >= min_points || throw(ArgumentError(
            "PolyFit{$max_degree} requires at least $min_points data points (got $(length(x))). " *
            "A degree-$max_degree polynomial needs $(min_points) points to estimate endpoint derivatives."
        ))
    end

    # Create spacing object (ScalarSpacing for Range, VectorSpacing for Vector)
    spacing = _create_spacing(x)

    # Build tridiagonal matrix A
    # CRITICAL: Use Vector allocation (NOT pool!) for persistent arrays
    # These arrays are stored in CubicSplineCache which outlives the function.
    dl = Vector{T}(undef, n)       # Lower diagonal (n elements)
    d_diag = Vector{T}(undef, n+1) # Main diagonal (n+1 elements) → becomes inv_d
    du = Vector{T}(undef, n)       # Upper diagonal (n elements)

    # First and last rows depend on BC type (type dispatch)
    _set_first_row!(d_diag, du, left_bc, spacing)
    _set_last_row!(dl, d_diag, right_bc, spacing)

    # Interior rows (same for all BC types)
    @inbounds for i in 2:n
        h_im1 = _get_h(spacing, i-1)
        h_i = _get_h(spacing, i)
        dl[i-1] = h_im1
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

# First element - Deriv2: d[1] = bc.val (second derivative value)
@inline function _compute_rhs_first!(
    d::AbstractVector{T}, bc::Deriv2{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    d[1] = bc.val
    return nothing
end

# First element - Deriv1: d[1] = 6[(y₂-y₁)/h₁ - S'(x₁)]
@inline function _compute_rhs_first!(
    d::AbstractVector{T}, bc::Deriv1{T}, y::AbstractVector{T}, ::AbstractVector{T}, spacing::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    d[1] = 6 * ((y[2] - y[1]) / _get_h(spacing, 1) - bc.val)
    return nothing
end

# Last element - Deriv2: d[end] = bc.val (second derivative value)
@inline function _compute_rhs_last!(
    d::AbstractVector{T}, bc::Deriv2{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    d[end] = bc.val
    return nothing
end

# Last element - Deriv1: d[end] = 6[S'(x_end) - (y_end - y_{end-1}) / h_end]
@inline function _compute_rhs_last!(
    d::AbstractVector{T}, bc::Deriv1{T}, y::AbstractVector{T}, ::AbstractVector{T}, spacing::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    n = length(y) - 1
    d[end] = 6 * (bc.val - (y[end] - y[end-1]) / _get_h(spacing, n))
    return nothing
end

# First element - Deriv3: d[1] = h[1] * bc.val (from -z[1] + z[2] = h[1] * val)
@inline function _compute_rhs_first!(
    d::AbstractVector{T}, bc::Deriv3{T}, ::AbstractVector{T}, ::AbstractVector{T}, spacing::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    d[1] = _get_h(spacing, 1) * bc.val
    return nothing
end

# Last element - Deriv3: d[end] = h[n] * bc.val (from -z[n] + z[n+1] = h[n] * val)
@inline function _compute_rhs_last!(
    d::AbstractVector{T}, bc::Deriv3{T}, ::AbstractVector{T}, ::AbstractVector{T}, spacing::AbstractGridSpacing{T}
) where {T<:AbstractFloat}
    n = length(d) - 1
    d[end] = _get_h(spacing, n) * bc.val
    return nothing
end

# First element - Generic PolyFit{D}: materialize to Deriv1, then delegate
# Supports all polynomial degrees: LinearFit (D=1), QuadraticFit (D=2), CubicFit (D=3), etc.
@inline function _compute_rhs_first!(
    d::AbstractVector{T}, bc::PolyFit{D, T}, y::AbstractVector{T}, x::AbstractVector{T}, spacing::AbstractGridSpacing{T}
) where {D, T<:AbstractFloat}
    # Materialize PolyFit{D} → Deriv1 using estimated derivative
    concrete_bc = materialize_bc(bc, x, y, Val(:left))
    # Delegate to existing Deriv1 code path
    _compute_rhs_first!(d, concrete_bc, y, x, spacing)
    return nothing
end

# Last element - Generic PolyFit{D}: materialize to Deriv1, then delegate
@inline function _compute_rhs_last!(
    d::AbstractVector{T}, bc::PolyFit{D, T}, y::AbstractVector{T}, x::AbstractVector{T}, spacing::AbstractGridSpacing{T}
) where {D, T<:AbstractFloat}
    # Materialize PolyFit{D} → Deriv1 using estimated derivative
    concrete_bc = materialize_bc(bc, x, y, Val(:right))
    # Delegate to existing Deriv1 code path
    _compute_rhs_last!(d, concrete_bc, y, x, spacing)
    return nothing
end

"""
Compute RHS vector for generic derivative BC system in-place.

The `x` parameter is needed for PolyFit{D} BCs (LinearFit, QuadraticFit, CubicFit, etc.)
which compute endpoint derivatives from data. For other BC types (Deriv1, Deriv2, Deriv3),
`x` is passed but ignored.
"""
@inline function compute_rhs!(
    d::AbstractVector{T}, y::AbstractVector{T}, x::AbstractVector{T},
    spacing::AbstractGridSpacing{T}, bc_config::BCPair{T,L,R}
) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    n = length(y) - 1
    _compute_rhs_first!(d, bc_config.left, y, x, spacing)
    # Use spacing accessors
    @inbounds for i in 2:n
        d[i] = 6 * ((y[i+1] - y[i]) / _get_h(spacing, i) - (y[i] - y[i-1]) / _get_h(spacing, i-1))
    end
    _compute_rhs_last!(d, bc_config.right, y, x, spacing)
    return nothing
end

# ----------------------------------------
# Periodic RHS function
# ----------------------------------------

"Compute RHS vector d for periodic cubic spline system in-place."
@inline function compute_rhs_periodic!(d::AbstractVector{T}, y::AbstractVector{T}, spacing::AbstractGridSpacing{T}) where {T}
    n = length(y) - 1

    # Use spacing accessors
    @inbounds d[1] = 6 * (y[2] - y[1]) / _get_h(spacing, 1) - 6 * (y[1] - y[end-1]) / _get_h(spacing, n)

    @inbounds for i in 2:n-1
        d[i] = 6 * (y[i+1] - y[i]) / _get_h(spacing, i) - 6 * (y[i] - y[i-1]) / _get_h(spacing, i-1)
    end

    @inbounds d[n] = 6 * (y[end] - y[end-1]) / _get_h(spacing, n) - 6 * (y[end-1] - y[end-2]) / _get_h(spacing, n-1)

    return nothing
end

# ========================================
# System Solvers
# ========================================
# Scalar Thomas solver moved to: core/thomas_lu_solver.jl
# - _ldiv_tridiagonal_nopiv!

"Solve periodic cyclic tridiagonal system using Sherman-Morrison formula."
@inline function _solve_cubic_system_periodic!(
    z_workspace::AbstractVector{T},
    y_temp::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T},S},
    y::AbstractVector{T}
) where {T<:AbstractFloat, X, F, S<:AbstractGridSpacing{T}}
    n = length(y) - 1

    compute_rhs_periodic!(y_temp, y, cache.spacing)
    _ldiv_tridiagonal_nopiv!(y_temp, cache.thomas)

    α = _get_h(cache.spacing, n)
    q = cache.bc_config.q

    vTy = α * (y_temp[1] + y_temp[n])
    vTq = α * (q[1] + q[n])

    denom = one(T) + vTq
    # Defensive check: unreachable under valid inputs (denom ≥ √3 for SPD systems),
    # but guards against corrupted data (NaN/Inf from invalid grid spacing).
    # The isfinite check catches NaN propagation since abs(NaN) < tol is always false.
    tol = sqrt(eps(T))
    if !isfinite(denom) || abs(denom) < tol
        throw(DomainError(denom,
            "Sherman-Morrison formula failed: denominator (1 + v'q) ≈ 0 or non-finite.\n" *
            "  denom = $denom (tol = $tol), α = $α, q[1] = $(q[1]), q[n] = $(q[n])\n" *
            "  This usually indicates corrupted input data (NaN/Inf) or degenerate grid."))
    end
    factor = vTy / denom

    @inbounds for i in 1:n
        z_workspace[i] = y_temp[i] - factor * q[i]
    end

    z_workspace[n+1] = z_workspace[1]

    return z_workspace
end

# ========================================
# Unified System Solver Entry Point
# ========================================
#
# All solvers now require explicit output argument (out_z) and use @with_pool
# for thread-safe workspace allocation. The 3-arg versions (using cache workspaces)
# have been removed as part of thread-safety refactoring.

"""
Solve cubic spline system (BCPair) with explicit output and pool-based workspace.
Thread-safe: workspaces allocated from task-local pool.
"""
@inline function _solve_system!(
    out_z::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BCPair{T,L,R},S},
    y::AbstractVector{T},
    bc_pair::BCPair{T,L,R}
) where {T<:AbstractFloat, X, F, L<:PointBC{T}, R<:PointBC{T}, S<:AbstractGridSpacing{T}}
    compute_rhs!(out_z, y, cache.x, cache.spacing, bc_pair)
    _ldiv_tridiagonal_nopiv!(out_z, cache.thomas)
    return out_z
end

"""
Solve cubic spline system (Periodic BC) with explicit output and pool-based workspace.
Thread-safe: workspaces allocated from task-local pool.
"""
@inline @with_pool pool function _solve_system!(
    out_z::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T},S},
    y::AbstractVector{T},
    ::PeriodicData{T}  # Unused, for API consistency with BCPair version
) where {T<:AbstractFloat, X, F, S<:AbstractGridSpacing{T}}
    n = length(y) - 1

    # Periodic workspaces need n elements (NOT length(y)!)
    y_temp = acquire!(pool, T, n)

    _solve_cubic_system_periodic!(out_z, y_temp, cache, y)
    return out_z
end

# Batch Thomas solvers moved to: core/thomas_lu_solver.jl
# - _ldiv_along_dim_vectorized!
# - _ldiv_along_dim!
