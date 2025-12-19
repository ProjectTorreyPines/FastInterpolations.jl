"""
    cubic_interp.jl

Zero-allocation cubic spline interpolation with reusable LU factorization.

# Design Philosophy

Similar to linear_interp.jl, this module provides:
1. Hot-path functions for zero-allocation evaluation
2. LU decomposition caching to avoid repeated factorizations
3. Support for varying y values with same x grid
4. Support for varying query points (x_query)

# Use Cases

**Scenario 1: One-shot interpolation**
```julia
result = cubic_interp(x, y, x_query)  # Allocating version
```

**Scenario 2: Fixed grid, multiple y vectors (RECOMMENDED)**
```julia
cache = CubicSplineCache(x)
for y_i in [y1, y2, ..., y9]
    result = cubic_interp(cache, y_i, x_query)
end
```
This saves 91% allocations and 87% memory when interpolating multiple fields.

# Mathematical Background

Natural cubic spline solves the tridiagonal system:
    A * z = d

where:
- A: tridiagonal matrix depending ONLY on x (grid geometry)
- d: RHS vector depending on y (function values)
- z: second derivative coefficients at knots

Key optimization: A can be LU-factorized once and reused for different y.
"""

# LinearAlgebra imported in main module

# ========================================
# Periodic BC Data Structure
# ========================================

"""
    PeriodicData{T}

Pre-computed data for Sherman-Morrison periodic spline solver.

# Fields
- `q::Vector{T}`: Pre-computed A'^{-1} * u vector for Sherman-Morrison formula
- `y_temp::Vector{T}`: Workspace for intermediate solution (for zero-allocation ldiv!)
- `period::T`: Period T = x[end] - x[1]

# Notes
For cyclic tridiagonal system A_cyclic = A' + u * v^T, Sherman-Morrison gives:
    z = y - ((v^T * y) / (1 + v^T * q)) * q
where q = A'^{-1} * u is pre-computed and reused for different y vectors.
"""
struct PeriodicData{T<:AbstractFloat}
    q::Vector{T}       # Pre-computed A'^{-1} * u
    y_temp::Vector{T}  # Workspace for ldiv! (zero-allocation solver)
    period::T          # x[end] - x[1]
end

"""
    CubicSplineCache{T,X,F,BC}

Cache structure for cubic spline interpolation with reusable LU factorization.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `X`: Grid type (Vector{T} or AbstractRange{T})
- `F`: LU factorization type
- `BC`: Boundary condition data type (Nothing for natural, PeriodicData{T} for periodic)

# Fields
- `x::X`: Grid points (immutable after construction, can be Range or Vector)
- `h::Vector{T}`: Grid spacing h[i] = x[i+1] - x[i]
- `lu_factor::F`: LU factorization of tridiagonal matrix A
- `d_workspace::Vector{T}`: Workspace for RHS vector computation
- `z_workspace::Vector{T}`: Workspace for solution vector
- `bc_data::BC`: Boundary condition data (Nothing for natural, PeriodicData for periodic)

# Notes
The LU factorization depends ONLY on x geometry and can be reused for:
- Different y vectors (varying function values)
- Different x_query vectors (varying query points)

When x is an AbstractRange, O(1) index lookup is used instead of O(log n) binary search.

# Boundary Conditions
- `bc=:natural` (default): Natural spline with z[1] = z[n+1] = 0
- `bc=:periodic`: Periodic spline with C2 continuity at boundaries
"""
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F,BC}
    x::X
    h::Vector{T}
    lu_factor::F
    d_workspace::Vector{T}
    z_workspace::Vector{T}
    bc_data::BC
end

"""
    CubicSplineCache(x::AbstractVector{T}; bc::Symbol=:natural) where {T<:AbstractFloat}

Construct a cubic spline cache for grid points `x`.

Pre-computes and factorizes the tridiagonal matrix that depends only on x geometry.
This factorization can be reused for interpolating different y vectors.

When `x` is an AbstractRange, the Range structure is preserved for O(1) index lookup
during evaluation. When `x` is a Vector, O(log n) binary search is used.

# Arguments
- `x::AbstractVector{T}`: Grid points (must be sorted, length >= 3)
- `bc::Symbol=:natural`: Boundary condition (`:natural` or `:periodic`)

# Returns
- `CubicSplineCache{T,X,F,BC}`: Reusable cache structure

# Example
```julia
x = range(0.0, 1.0, 51)  # Range preserved for O(1) lookup
cache = CubicSplineCache(x)                    # Natural BC (default)
cache_periodic = CubicSplineCache(x; bc=:periodic)  # Periodic BC

# Reuse for multiple y vectors
y1 = sin.(x)
y2 = cos.(x)
result1 = cubic_interp(cache, y1, [0.25, 0.75])
result2 = cubic_interp(cache, y2, [0.25, 0.75])
```
"""
function CubicSplineCache(x::AbstractVector{T}; bc::Symbol=:natural) where {T<:AbstractFloat}
    bc in (:natural, :periodic) || throw(ArgumentError("bc must be :natural or :periodic, got :$bc"))

    if bc == :periodic
        return _build_periodic_cache(x)
    else
        return _build_natural_cache(x)
    end
end

"""
    _build_natural_cache(x::AbstractVector{T}) where {T<:AbstractFloat}

Build cache for natural cubic spline (z[1] = z[n+1] = 0).
Internal function called by CubicSplineCache constructor.
"""
function _build_natural_cache(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x) - 1

    # Compute grid spacing h[i] = x[i+1] - x[i]
    # h is padded: [0, h1, h2, ..., hn, 0]
    h = Vector{T}(undef, n + 2)
    h[1] = zero(T)
    h[end] = zero(T)
    @inbounds for i in 1:n
        h[i+1] = x[i+1] - x[i]
    end

    # Build tridiagonal matrix A
    # Natural cubic spline boundary conditions: z[1] = z[n+1] = 0
    # Interior points: h[i]*z[i-1] + 2(h[i]+h[i+1])*z[i] + h[i+1]*z[i+1] = d[i]

    dl = Vector{T}(undef, n)      # Lower diagonal
    d_diag = Vector{T}(undef, n+1) # Main diagonal
    du = Vector{T}(undef, n)      # Upper diagonal

    # First and last rows: z[1] = 0 and z[n+1] = 0 (natural boundary)
    d_diag[1] = one(T)
    du[1] = zero(T)

    # Interior rows
    @inbounds for i in 2:n
        dl[i-1] = h[i]
        d_diag[i] = 2 * (h[i] + h[i+1])
        du[i] = h[i+1]
    end

    # Last row
    dl[n] = zero(T)
    d_diag[n+1] = one(T)

    tA = Tridiagonal(dl, d_diag, du)
    lu_factor = lu(tA)

    # Allocate workspaces
    d_workspace = Vector{T}(undef, n + 1)
    z_workspace = Vector{T}(undef, n + 1)

    return CubicSplineCache(x, h[1:n+1], lu_factor, d_workspace, z_workspace, nothing)
end

"""
    _build_periodic_cache(x::AbstractVector{T}) where {T<:AbstractFloat}

Build cache for periodic cubic spline using Sherman-Morrison formula.

For cyclic tridiagonal system A_cyc = A' + u * v^T, pre-computes:
- LU factorization of modified tridiagonal A'
- Vector q = A'^{-1} * u for Sherman-Morrison formula

# Endpoint Convention (끝점 포함)
User provides N points where y[1] ≈ y[N]. Period T = x[N] - x[1].
Internally uses N-1 intervals, with z[N] = z[1] for periodicity.
"""
function _build_periodic_cache(x::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(x)
    n = N - 1  # Number of intervals (N-1 unique intervals)

    length(x) >= 4 || throw(ArgumentError("Periodic spline requires at least 4 points"))

    # Period
    period = x[N] - x[1]

    # Compute grid spacing h[i] = x[i+1] - x[i]
    # h is padded: [0, h1, h2, ..., hn, 0] to match natural spline convention
    h = Vector{T}(undef, n + 2)
    h[1] = zero(T)
    h[end] = zero(T)
    @inbounds for i in 1:n
        h[i+1] = x[i+1] - x[i]
    end

    # Build modified tridiagonal matrix A' for Sherman-Morrison
    # Original cyclic matrix has corners h[n] at positions (1,n) and (n,1)
    # A' = A_cyc - u * v^T where u = [1,0,...,0,1]^T, v = [α,0,...,0,α]^T, α = h[n]
    # A'[1,1] = A[1,1] - α, A'[n,n] = A[n,n] - α

    α = h[n+1]  # h[n] in 1-indexed h = h[n+1] in padded h

    dl = Vector{T}(undef, n - 1)     # Lower diagonal (n-1 elements)
    d_diag = Vector{T}(undef, n)     # Main diagonal (n elements)
    du = Vector{T}(undef, n - 1)     # Upper diagonal (n-1 elements)

    # Modified diagonal entries
    # A'[1,1] = 2(h[n] + h[1]) - α = 2(h[n] + h[1]) - h[n] = h[n] + 2*h[1]
    d_diag[1] = h[n+1] + 2 * h[2]  # h[n] + 2*h[1] in original indexing

    # Interior rows (i = 2..n-1)
    @inbounds for i in 2:n-1
        dl[i-1] = h[i]               # h[i-1] in original = h[(i-1)+1] in padded
        d_diag[i] = 2 * (h[i] + h[i+1])
        du[i-1] = h[i+1]             # h[i] in original = h[i+1] in padded
    end

    # Last row modifications
    # A'[n,n] = 2(h[n-1] + h[n]) - α = h[n-1] + h[n] + h[n-1] = h[n-1] + 2*h[n-1]? No...
    # A'[n,n] = 2(h[n-1] + h[n]) - h[n] = 2*h[n-1] + h[n]
    dl[n-1] = h[n]                   # h[n-1] in original
    d_diag[n] = 2 * h[n] + h[n+1]    # 2*h[n-1] + h[n] in original

    # Upper diagonal last element
    if n > 1
        du[n-1] = h[n]               # h[n-1] in original
    end

    tA_prime = Tridiagonal(dl, d_diag, du)
    lu_factor = lu(tA_prime)

    # Pre-compute q = A'^{-1} * u where u = [1, 0, ..., 0, 1]^T
    u = zeros(T, n)
    u[1] = one(T)
    u[n] = one(T)
    q = lu_factor \ u

    # Allocate workspaces (n+1 for z to include z[n+1] = z[1])
    d_workspace = Vector{T}(undef, n)
    z_workspace = Vector{T}(undef, n + 1)
    y_temp_workspace = Vector{T}(undef, n)  # For zero-allocation ldiv!

    bc_data = PeriodicData(q, y_temp_workspace, period)

    return CubicSplineCache(x, h[1:n+1], lu_factor, d_workspace, z_workspace, bc_data)
end

"""
    compute_rhs!(d::Vector{T}, y::AbstractVector{T}, h::Vector{T}) where {T}

Compute RHS vector d for cubic spline system in-place.

For natural cubic spline:
- d[1] = 0 (boundary condition)
- d[i] = 6(y[i+1]-y[i])/h[i+1] - 6(y[i]-y[i-1])/h[i] for i=2..n
- d[n+1] = 0 (boundary condition)

# Arguments
- `d::Vector{T}`: Output workspace (modified in-place)
- `y::AbstractVector{T}`: Function values at grid points
- `h::Vector{T}`: Grid spacing
"""
@inline function compute_rhs!(d::Vector{T}, y::AbstractVector{T}, h::Vector{T}) where {T}
    n = length(y) - 1

    # Boundary conditions
    d[1] = zero(T)
    d[n+1] = zero(T)

    # Interior points
    @inbounds for i in 2:n
        d[i] = 6 * (y[i+1] - y[i]) / h[i+1] - 6 * (y[i] - y[i-1]) / h[i]
    end

    return nothing
end

"""
    compute_rhs_periodic!(d::Vector{T}, y::AbstractVector{T}, h::Vector{T}) where {T}

Compute RHS vector d for periodic cubic spline system in-place.

For periodic cubic spline with N points (y[1] ≈ y[N]):
- Uses N-1 intervals
- d[i] = 6(y[i+1]-y[i])/h[i+1] - 6(y[i]-y[i-1])/h[i] for interior points
- Wraps around at boundaries: y[0] = y[N-1], y[N] = y[1]

# Arguments
- `d::Vector{T}`: Output workspace of length n = N-1 (modified in-place)
- `y::AbstractVector{T}`: Function values at grid points (length N)
- `h::Vector{T}`: Grid spacing (padded format [0, h1, ..., hn, 0])
"""
@inline function compute_rhs_periodic!(d::Vector{T}, y::AbstractVector{T}, h::Vector{T}) where {T}
    N = length(y)
    n = N - 1  # Number of intervals

    # First point (wraps: y[0] = y[N-1] conceptually, but since y[N] ≈ y[1], use y[N-1])
    # d[1] = 6(y[2]-y[1])/h[2] - 6(y[1]-y[N-1])/h[N] (h[N] = h[n+1] in padded)
    # But h[n+1] = h[N] in original = h[n] in 0-indexed... let me use h[n+1] directly
    @inbounds d[1] = 6 * (y[2] - y[1]) / h[2] - 6 * (y[1] - y[N-1]) / h[n+1]

    # Interior points (i = 2..n-1)
    @inbounds for i in 2:n-1
        d[i] = 6 * (y[i+1] - y[i]) / h[i+1] - 6 * (y[i] - y[i-1]) / h[i]
    end

    # Last point (wraps: y[n+1] = y[1])
    # d[n] = 6(y[1]-y[n])/h[n+1] - 6(y[n]-y[n-1])/h[n]
    # But y[n] = y[N-1], y[n+1] = y[N] ≈ y[1]
    @inbounds d[n] = 6 * (y[N] - y[N-1]) / h[n+1] - 6 * (y[N-1] - y[N-2]) / h[n]

    return nothing
end

"""
    _solve_cubic_system_periodic!(z_workspace, d_workspace, cache, y)

Solve periodic cyclic tridiagonal system using Sherman-Morrison formula.

# Sherman-Morrison Formula
For A_cyc = A' + u * v^T:
    z = A'^{-1}d - (v^T * A'^{-1}d) / (1 + v^T * q) * q
where q = A'^{-1}u is pre-computed in cache.bc_data.

# Returns
Reference to z_workspace containing [z[1], ..., z[n], z[n+1]] where z[n+1] = z[1].
"""
@inline function _solve_cubic_system_periodic!(
    z_workspace::Vector{T},
    d_workspace::Vector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T}
) where {T<:AbstractFloat, X, F}
    N = length(y)
    n = N - 1

    # Step 1: Compute RHS vector d (length n)
    compute_rhs_periodic!(d_workspace, y, cache.h)

    # Step 2: Solve A' * y_temp = d using cached LU factorization (ZERO-ALLOCATION)
    y_temp = cache.bc_data.y_temp
    ldiv!(y_temp, cache.lu_factor, d_workspace)

    # Step 3: Apply Sherman-Morrison correction
    # z = y_temp - (v^T * y_temp) / (1 + v^T * q) * q
    # v = [α, 0, ..., 0, α]^T where α = h[n] = h[n+1] in padded h
    α = cache.h[n+1]
    q = cache.bc_data.q

    vTy = α * (y_temp[1] + y_temp[n])
    vTq = α * (q[1] + q[n])

    factor = vTy / (one(T) + vTq)

    # Store result in z_workspace[1:n]
    @inbounds for i in 1:n
        z_workspace[i] = y_temp[i] - factor * q[i]
    end

    # z[n+1] = z[1] for periodicity (wrap-around)
    z_workspace[n+1] = z_workspace[1]

    return z_workspace
end

"""
    _eval_cubic_at_point_periodic(x, y, h, z, xi, period)

Evaluate periodic cubic spline at a single point.

Wraps query point to domain [x[1], x[1] + period) before evaluation.
"""
@inline function _eval_cubic_at_point_periodic(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    period::T
) where {T<:AbstractFloat}
    # Wrap xi to domain [x[1], x[1] + period)
    xi_wrapped = _wrap_to_domain(xi, first(x), period)

    # Find interval and evaluate (same as natural spline)
    idx, x0, x1 = _find_interval_with_bounds(x, xi_wrapped)

    # Cubic spline formula using pre-computed z coefficients
    dt1 = xi_wrapped - x0
    dt2 = x1 - xi_wrapped
    h_i = h[idx+1]

    @inbounds begin
        I = (z[idx] * dt2^3 + z[idx+1] * dt1^3) / (6 * h_i)
        C = (y[idx+1] / h_i - z[idx+1] * h_i / 6) * dt1
        D = (y[idx] / h_i - z[idx] * h_i / 6) * dt2
    end

    return I + C + D
end

"""
    cubic_interp!(output::AbstractVector{T}, cache::CubicSplineCache{T},
                  y::AbstractVector{T}, x_query::AbstractVector{T}; extrapolation=:none) where {T<:AbstractFloat}

In-place cubic spline interpolation using cached LU factorization.

Solves the tridiagonal system ONCE, then evaluates at all query points.

# Arguments
- `output::AbstractVector{T}`: Pre-allocated output buffer (modified in-place)
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points (length must match cache.x)
- `x_query::AbstractVector{T}`: Query points where interpolation is evaluated
- `extrapolation::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`)

# Notes
- Zero allocations after cache construction
- Reuses LU factorization from cache
- Thread-safe if different caches/workspaces are used per thread
- Pattern: Solve system once -> evaluate at all query points
"""
function cubic_interp!(output::AbstractVector{T}, cache::CubicSplineCache{T,X,F,Nothing},
                       y::AbstractVector{T}, x_query::AbstractVector{T}; extrapolation::Symbol=:none) where {T<:AbstractFloat, X, F}
    @assert length(y) == length(cache.x) "y length must match cache grid"
    @assert length(output) == length(x_query) "output length must match x_query"

    # Direct branching to Val literals for type stability
    extrapolation === :none      && return _cubic_interp_impl!(output, cache, y, x_query, Val(:none))
    extrapolation === :constant  && return _cubic_interp_impl!(output, cache, y, x_query, Val(:constant))
    extrapolation === :extension && return _cubic_interp_impl!(output, cache, y, x_query, Val(:extension))
    throw(ArgumentError("extrapolation must be :none, :constant, or :extension, got :$extrapolation"))
end

# Internal implementation with Val dispatch (type-stable)
@inline function _cubic_interp_impl!(output::AbstractVector{T}, cache::CubicSplineCache{T,X,F,Nothing},
                                     y::AbstractVector{T}, x_query::AbstractVector{T}, extrap::Val) where {T<:AbstractFloat, X, F}
    # Vector-level domain check (skipped for extension/constant via no-op dispatch)
    _check_domain(cache.x, x_query, extrap)

    # Step 1: Solve for z coefficients (solves system ONCE for all query points)
    z = _solve_cubic_system!(cache.z_workspace, cache.d_workspace, cache, y)

    # Step 2: Evaluate at all query points (@inbounds skips scalar _check_domain)
    @inbounds for (k, xq) in enumerate(x_query)
        output[k] = _eval_cubic_with_extrap(cache.x, y, cache.h, z, xq, extrap)
    end

    return output
end

# Periodic BC dispatch (extrapolation ignored - coordinates are wrapped)
function cubic_interp!(output::AbstractVector{T}, cache::CubicSplineCache{T,X,F,PeriodicData{T}},
                       y::AbstractVector{T}, x_query::AbstractVector{T}; extrapolation::Symbol=:none) where {T<:AbstractFloat, X, F}
    @assert length(y) == length(cache.x) "y length must match cache grid"
    @assert length(output) == length(x_query) "output length must match x_query"

    # Step 1: Solve for z coefficients using Sherman-Morrison
    z = _solve_cubic_system_periodic!(cache.z_workspace, cache.d_workspace, cache, y)

    # Step 2: Evaluate at all query points with coordinate wrapping (extrapolation ignored)
    period = cache.bc_data.period
    @inbounds for (k, xq) in enumerate(x_query)
        output[k] = _eval_cubic_at_point_periodic(cache.x, y, cache.h, z, xq, period)
    end

    return output
end

"""
    cubic_interp(cache::CubicSplineCache{T}, y::AbstractVector{T},
                 x_query::AbstractVector{T}; extrapolation=:none) where {T<:AbstractFloat}

Allocating version of cubic spline interpolation using cached LU factorization.

# Arguments
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points
- `extrapolation::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`)

# Returns
- `Vector{T}`: Interpolated values at x_query points

# Example
```julia
cache = CubicSplineCache(collect(range(0.0, 1.0, 51)))
y = sin.(cache.x)
result = cubic_interp(cache, y, [0.25, 0.5, 0.75])
```
"""
function cubic_interp(cache::CubicSplineCache{T}, y::AbstractVector{T},
                      x_query::AbstractVector{T}; extrapolation::Symbol=:none) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(x_query))
    cubic_interp!(output, cache, y, x_query; extrapolation=extrapolation)
    return output
end

"""
    cubic_interp(x::AbstractVector{T}, y::AbstractVector{T},
                 x_query::AbstractVector{T}; bc=:natural, extrapolation=:none, autocache=true) where {T<:AbstractFloat}

Cubic spline interpolation with optional automatic caching.

With `autocache=true` (default), automatically reuses cached LU factorization for
previously seen x-grids. First call creates cache (~35 allocations), subsequent calls
with same x-grid reuse it (~2 allocations).

With `autocache=false`, constructs cache, computes interpolation, and discards cache.

# Arguments
- `x::AbstractVector{T}`: Grid points
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points
- `bc::Symbol=:natural`: Boundary condition (`:natural` or `:periodic`)
- `extrapolation::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`)
- `autocache::Bool=true`: Enable transparent cache reuse (default: true)

# Returns
- `Vector{T}`: Interpolated values at x_query points

# Extrapolation Modes
- `:none` (default): Throws DomainError if query point is outside domain
- `:constant`: Returns boundary values (y[1] or y[end]) outside domain
- `:extension`: Extends boundary polynomial outside domain
- For `bc=:periodic`: extrapolation is ignored (coordinates are always wrapped)

# Performance
With autocache=true, repeated calls with same x-grid reuse cached LU factorization:
- First call: ~35 allocations (cache creation)
- Subsequent calls: ~2 allocations (cache reuse)

# Example
```julia
result = cubic_interp(x, y, x_query)              # Auto-cached (default)
result = cubic_interp(x, y, x_query; autocache=false)  # One-shot, no caching
result = cubic_interp(x, y, x_query; extrapolation=:extension)  # Extend beyond domain
```
"""
function cubic_interp(x::AbstractVector{T}, y::AbstractVector{T},
                      x_query::AbstractVector{T}; bc::Symbol=:natural, extrapolation::Symbol=:none, autocache::Bool=true) where {T<:AbstractFloat}
    bc in (:natural, :periodic) || throw(ArgumentError("bc must be :natural or :periodic, got :$bc"))
    extrapolation in (:none, :constant, :extension) || throw(ArgumentError("extrapolation must be :none, :constant, or :extension, got :$extrapolation"))

    if bc == :periodic
        # Validate periodic endpoints (once, zero runtime overhead)
        _check_periodic_endpoints(y)
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
        return cubic_interp(cache, y, x_query; extrapolation=extrapolation)
    elseif autocache
        cache = get_cubic_cache(x, Val(:natural))
        return cubic_interp(cache, y, x_query; extrapolation=extrapolation)
    else
        cache = CubicSplineCache(x)
        return cubic_interp(cache, y, x_query; extrapolation=extrapolation)
    end
end

"""
    cubic_interp!(output::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
                  x_query::AbstractVector{T}; bc=:natural, extrapolation=:none, autocache=true) where {T<:AbstractFloat}

In-place cubic spline interpolation with optional automatic caching.

Like the allocating version, this supports auto-cache to transparently reuse LU factorization
for the same x-grid across multiple calls. This gives you **zero-allocation + auto-cache**!

# Arguments
- `output::AbstractVector{T}`: Pre-allocated output buffer (modified in-place)
- `x::AbstractVector{T}`: Grid points
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points
- `bc::Symbol=:natural`: Boundary condition (`:natural` or `:periodic`)
- `extrapolation::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`)
- `autocache::Bool=true`: Enable transparent cache reuse (default: true)

# Example
```julia
output = Vector{Float64}(undef, length(x_query))
cubic_interp!(output, x, y, x_query)  # Auto-cached (default)
cubic_interp!(output, x, y, x_query; extrapolation=:extension)  # Extend beyond domain
```
"""
function cubic_interp!(output::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
                       x_query::AbstractVector{T}; bc::Symbol=:natural, extrapolation::Symbol=:none, autocache::Bool=true) where {T<:AbstractFloat}
    bc in (:natural, :periodic) || throw(ArgumentError("bc must be :natural or :periodic, got :$bc"))
    extrapolation in (:none, :constant, :extension) || throw(ArgumentError("extrapolation must be :none, :constant, or :extension, got :$extrapolation"))

    if bc == :periodic
        _check_periodic_endpoints(y)
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
        return cubic_interp!(output, cache, y, x_query; extrapolation=extrapolation)
    elseif autocache
        cache = get_cubic_cache(x, Val(:natural))
        return cubic_interp!(output, cache, y, x_query; extrapolation=extrapolation)
    else
        cache = CubicSplineCache(x)
        return cubic_interp!(output, cache, y, x_query; extrapolation=extrapolation)
    end
end

"""
    _eval_cubic_at_point(x, y, h, z, xi) -> value

True zero-allocation cubic spline evaluation at a single point.

Uses pre-computed second derivative coefficients z to evaluate the cubic polynomial.
This is the hot path for broadcast fusion - must be allocation-free and inlined!

# Arguments
- `x::AbstractVector{T}`: Grid points
- `y::AbstractVector{T}`: Function values at grid points
- `h::AbstractVector{T}`: Grid spacing (from cache)
- `z::AbstractVector{T}`: Pre-computed second derivative coefficients
- `xi::T`: Query point (scalar)

# Returns
- `T`: Interpolated value (single scalar, zero allocation)

# Implementation Notes
- No array allocations, no system solves
- Just arithmetic operations on pre-computed data
- Inlined for perfect broadcast fusion
"""
@inline function _eval_cubic_at_point(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T
) where {T<:AbstractFloat}
    idx, x0, x1 = _find_interval_with_bounds(x, xi)

    # Cubic spline formula using pre-computed z coefficients
    dt1 = xi - x0
    dt2 = x1 - xi
    h_i = h[idx+1]

    @inbounds begin
        I = (z[idx] * dt2^3 + z[idx+1] * dt1^3) / (6 * h_i)
        C = (y[idx+1] / h_i - z[idx+1] * h_i / 6) * dt1
        D = (y[idx] / h_i - z[idx] * h_i / 6) * dt2
    end

    return I + C + D
end

# ========================================
# Extrapolation-aware evaluation functions
# ========================================

"""
    _eval_cubic_with_extrap(x, y, h, z, xi, ::Val{:none})

Evaluate cubic spline with no extrapolation - throws DomainError if outside domain.
Uses shared `_check_domain` from utils.jl.
"""
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:none}
) where {T<:AbstractFloat}
    _check_domain(x, xi, Val(:none))
    return _eval_cubic_at_point(x, y, h, z, xi)
end

"""
    _eval_cubic_with_extrap(x, y, h, z, xi, ::Val{:constant})

Evaluate cubic spline with constant extrapolation - returns boundary values outside domain.
"""
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:constant}
) where {T<:AbstractFloat}
    xi < first(x) && return @inbounds y[1]
    xi > last(x) && return @inbounds y[end]
    return _eval_cubic_at_point(x, y, h, z, xi)
end

"""
    _eval_cubic_with_extrap(x, y, h, z, xi, ::Val{:extension})

Evaluate cubic spline with extension extrapolation - extends boundary polynomial outside domain.
This is the default behavior (same as _eval_cubic_at_point).
"""
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:extension}
) where {T<:AbstractFloat}
    return _eval_cubic_at_point(x, y, h, z, xi)
end

"""
    _solve_cubic_system!(z_workspace, d_workspace, cache, y)

Solve tridiagonal system A * z = d for second derivative coefficients (natural BC).

Uses pre-computed LU factorization from cache. Modifies workspaces in-place.

# Arguments
- `z_workspace::Vector{T}`: Output workspace for z coefficients (modified in-place)
- `d_workspace::Vector{T}`: Workspace for RHS vector (modified in-place)
- `cache::CubicSplineCache{T,...,Nothing}`: Pre-computed cache with LU factorization (natural BC)
- `y::AbstractVector{T}`: Function values at grid points

# Returns
- `z_workspace`: Reference to modified z_workspace (for convenience)
"""
@inline function _solve_cubic_system!(
    z_workspace::Vector{T},
    d_workspace::Vector{T},
    cache::CubicSplineCache{T,X,F,Nothing},
    y::AbstractVector{T}
) where {T<:AbstractFloat, X, F}
    # Step 1: Compute RHS vector d from y values
    compute_rhs!(d_workspace, y, cache.h)

    # Step 2: Solve A * z = d using cached LU factorization
    ldiv!(z_workspace, cache.lu_factor, d_workspace)

    return z_workspace
end

"""
    cubic_interp_scalar(cache::CubicSplineCache{T}, y::AbstractVector{T},
                        x_query::T; extrapolation=:none) where {T<:AbstractFloat}

Scalar cubic spline evaluation (solves system once, evaluates once).

For repeated evaluations at different query points with same y, use CubicInterpolant
instead, which pre-computes z coefficients once and reuses them for all evaluations.

# Arguments
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::T`: Single query point (scalar)
- `extrapolation::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`)

# Returns
- `T`: Interpolated value at x_query

# Note
This function solves the tridiagonal system for each call. If you need to evaluate
at multiple points with the same y, create a CubicInterpolant instead:
```julia
itp = cubic_interp(x, y)  # Solve once
vals = itp.(query_points)  # Reuse z for all points
```
"""
@inline function cubic_interp_scalar(cache::CubicSplineCache{T,X,F,Nothing}, y::AbstractVector{T},
                                      x_query::T; extrapolation::Symbol=:none) where {T<:AbstractFloat, X, F}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    # Solve for z coefficients (reuses cache workspaces)
    z = _solve_cubic_system!(cache.z_workspace, cache.d_workspace, cache, y)

    # Direct branching to Val literals for type stability
    extrapolation === :none      && return _eval_cubic_with_extrap(cache.x, y, cache.h, z, x_query, Val(:none))
    extrapolation === :constant  && return _eval_cubic_with_extrap(cache.x, y, cache.h, z, x_query, Val(:constant))
    extrapolation === :extension && return _eval_cubic_with_extrap(cache.x, y, cache.h, z, x_query, Val(:extension))
    throw(ArgumentError("extrapolation must be :none, :constant, or :extension, got :$extrapolation"))
end

# Periodic BC dispatch for scalar evaluation (extrapolation ignored)
@inline function cubic_interp_scalar(cache::CubicSplineCache{T,X,F,PeriodicData{T}}, y::AbstractVector{T},
                                      x_query::T; extrapolation::Symbol=:none) where {T<:AbstractFloat, X, F}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    # Solve for z coefficients using Sherman-Morrison
    z = _solve_cubic_system_periodic!(cache.z_workspace, cache.d_workspace, cache, y)

    # Evaluate at query point with coordinate wrapping (extrapolation ignored)
    return _eval_cubic_at_point_periodic(cache.x, y, cache.h, z, x_query, cache.bc_data.period)
end

# Scalar query point convenience methods
cubic_interp!(output::AbstractVector{T}, cache::CubicSplineCache{T},
              y::AbstractVector{T}, x_query::T; extrapolation::Symbol=:none) where {T<:AbstractFloat} =
    cubic_interp!(output, cache, y, [x_query]; extrapolation=extrapolation)

cubic_interp!(output::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
              x_query::T; bc::Symbol=:natural, extrapolation::Symbol=:none, autocache::Bool=true) where {T<:AbstractFloat} =
    cubic_interp!(output, x, y, [x_query]; bc=bc, extrapolation=extrapolation, autocache=autocache)

# CRITICAL: Zero-allocation scalar path for broadcast fusion
cubic_interp(cache::CubicSplineCache{T}, y::AbstractVector{T},
             x_query::T; extrapolation::Symbol=:none) where {T<:AbstractFloat} =
    cubic_interp_scalar(cache, y, x_query; extrapolation=extrapolation)

# Scalar query with autocache option
function cubic_interp(x::AbstractVector{T}, y::AbstractVector{T},
                      x_query::T; bc::Symbol=:natural, extrapolation::Symbol=:none, autocache::Bool=true) where {T<:AbstractFloat}
    bc in (:natural, :periodic) || throw(ArgumentError("bc must be :natural or :periodic, got :$bc"))
    extrapolation in (:none, :constant, :extension) || throw(ArgumentError("extrapolation must be :none, :constant, or :extension, got :$extrapolation"))

    if bc == :periodic
        _check_periodic_endpoints(y)
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
        return cubic_interp_scalar(cache, y, x_query; extrapolation=extrapolation)
    elseif autocache
        cache::CubicSplineCache{T} = get_cubic_cache(x, Val(:natural))
        return cubic_interp_scalar(cache, y, x_query; extrapolation=extrapolation)
    else
        cache = CubicSplineCache(x)
        return cubic_interp_scalar(cache, y, x_query; extrapolation=extrapolation)
    end
end

# ========================================
# Callable Interpolator for Broadcast Fusion
# ========================================

# Helper for BC-aware evaluation (used by CubicInterpolant)
# Natural BC with extrapolation
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,Nothing},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    extrap::Val
) where {T<:AbstractFloat, X, F}
    _eval_cubic_with_extrap(cache.x, y, h, z, xi, extrap)
end

# Periodic BC ignores extrapolation (coordinates are wrapped)
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val  # extrapolation ignored for periodic
) where {T<:AbstractFloat, X, F}
    _eval_cubic_at_point_periodic(cache.x, y, h, z, xi, cache.bc_data.period)
end

"""
    CubicInterpolant{T,C,Y,Z,E}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `cubic_interp(x, y)` (2-argument form).

# Fields
- `cache::C`: Pre-computed CubicSplineCache (LU factorization)
- `y::Y`: y-values (function values at grid points)
- `z::Z`: Pre-computed second derivative coefficients (solves system once!)
- `extrap::E`: Extrapolation mode (Val{:none}, Val{:constant}, Val{:extension})

# Usage
```julia
# Create interpolator (reusable, pre-computes z coefficients)
itp = cubic_interp(x, y)

# Use in broadcast (fused, zero-allocation per call!)
result = @. coef * itp(rho) * other_terms

# Reuse interpolator multiple times
vals1 = itp.(query_points1)
vals2 = @. compute(itp(query_points2))

# Scalar call (zero-allocation!)
val = itp(0.5)
```

# Extrapolation Modes
- `:none` (default): Throws DomainError if query point is outside domain
- `:constant`: Returns boundary values (y[1] or y[end]) outside domain
- `:extension`: Extends boundary polynomial outside domain

# Performance Notes
- System solved ONCE at construction -> z coefficients pre-computed
- Each scalar call just evaluates cubic polynomial (zero-allocation!)
- Broadcast operations are perfectly fused (no intermediate arrays)
- Optimal for multiple evaluations with same x-grid and y-values
"""
struct CubicInterpolant{T<:AbstractFloat,C<:CubicSplineCache{T},Y<:AbstractVector{T},Z<:AbstractVector{T},E<:Val}
    cache::C
    y::Y
    z::Z  # Pre-computed second derivative coefficients
    extrap::E  # Extrapolation mode

    function CubicInterpolant(
        cache::C,
        y::Y,
        z::Z,
        extrap::E
    ) where {T<:AbstractFloat, C<:CubicSplineCache{T}, Y<:AbstractVector{T}, Z<:AbstractVector{T}, E<:Val}
        @assert length(cache.x) == length(y) "cache grid and y must have same length"
        @assert length(cache.x) == length(z) "z coefficients must match grid length"
        new{T,C,Y,Z,E}(cache, y, z, extrap)
    end
end

# Scalar call - hot path (inlined for broadcast fusion)
# CRITICAL: Uses pre-computed z coefficients -> TRUE zero-allocation!
# Uses _eval_with_bc for BC-aware dispatch (natural vs periodic)
@inline function (itp::CubicInterpolant{T})(xi::T) where {T<:AbstractFloat}
    _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xi, itp.extrap)
end

# Real scalar wrapper for convenience
@inline function (itp::CubicInterpolant{T})(xi::S) where {T<:AbstractFloat, S<:Real}
    _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, T(xi), itp.extrap)
end

# Vector call - uses pre-computed z coefficients (no redundant system solve!)
# Domain check dispatches on extrap: :none checks, others are no-op
# Periodic BC wraps coordinates in _eval_with_bc, so no domain error possible
function (itp::CubicInterpolant{T})(xi::AbstractVector{S}) where {T<:AbstractFloat, S<:Real}
    xi_typed = S === T ? xi : T.(xi)
    _check_domain(itp.cache.x, xi_typed, itp.extrap)
    output = Vector{T}(undef, length(xi_typed))
    @inbounds for (k, xq) in enumerate(xi_typed)
        output[k] = _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xq, itp.extrap)
    end
    return output
end

# Optimized path when xi element type matches T (zero conversion)
function (itp::CubicInterpolant{T})(xi::AbstractVector{T}) where {T<:AbstractFloat}
    _check_domain(itp.cache.x, xi, itp.extrap)
    output = Vector{T}(undef, length(xi))
    @inbounds for (k, xq) in enumerate(xi)
        output[k] = _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xq, itp.extrap)
    end
    return output
end

# In-place vector call - zero allocation
function (itp::CubicInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}) where {T<:AbstractFloat}
    @assert length(output) == length(xi) "output length must match xi length"
    _check_domain(itp.cache.x, xi, itp.extrap)
    @inbounds for (k, xq) in enumerate(xi)
        output[k] = _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xq, itp.extrap)
    end
    return output
end

# In-place with type conversion
function (itp::CubicInterpolant{T})(output::AbstractVector, xi::AbstractVector{S}) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = T.(xi)
    _check_domain(itp.cache.x, xi_typed, itp.extrap)
    @inbounds for (k, xq) in enumerate(xi_typed)
        output[k] = _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xq, itp.extrap)
    end
    return output
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    cubic_interp(x, y; bc=:natural, extrapolation=:none, autocache=true) -> CubicInterpolant

Create a callable interpolant for broadcast fusion and reuse.

Pre-computes second derivative coefficients z ONCE at construction time,
enabling true zero-allocation scalar evaluations in broadcast operations.

# Arguments
- `x::AbstractVector`: Grid points (must be sorted, length >= 3)
- `y::AbstractVector`: Function values at grid points
- `bc::Symbol=:natural`: Boundary condition (`:natural` or `:periodic`)
- `extrapolation::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`)
- `autocache::Bool=true`: Use automatic cache lookup (default: true)

# Returns
`CubicInterpolant` object that can be:
- Called with scalar: `itp(0.5)` (zero-allocation!)
- Broadcasted: `itp.(rho)` or `@. coef * itp(rho)` (zero-allocation per call!)
- Reused multiple times without re-creating

# Extrapolation Modes
- `:none` (default): Throws DomainError if query point is outside domain
- `:constant`: Returns boundary values (y[1] or y[end]) outside domain
- `:extension`: Extends boundary polynomial outside domain
- For `bc=:periodic`: extrapolation is ignored (coordinates are always wrapped)

# Example
```julia
itp = cubic_interp(x, y)           # Pre-computes z coefficients
val = itp(0.5)                      # Scalar (zero-allocation)
vals = itp.(query_points)           # Broadcast
result = @. coef * itp(rho) * ne    # Fused broadcast
```

# Performance Notes
- Construction: Solves system ONCE, stores z coefficients
- Scalar calls: Zero-allocation (just arithmetic with pre-computed z)
- Best for repeated evaluations with same (x, y)
"""
function cubic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    bc::Symbol=:natural,
    extrapolation::Symbol=:none,
    autocache::Bool=true
) where {T<:AbstractFloat}
    bc in (:natural, :periodic) || throw(ArgumentError("bc must be :natural or :periodic, got :$bc"))
    extrapolation in (:none, :constant, :extension) || throw(ArgumentError("extrapolation must be :none, :constant, or :extension, got :$extrapolation"))

    if bc == :periodic
        _check_periodic_endpoints(y)
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
    else
        cache = autocache ? get_cubic_cache(x, Val(:natural)) : CubicSplineCache(x)
    end

    # Pre-compute z coefficients (solve system once, then copy for storage)
    # Dispatches based on BC type (natural vs periodic)
    _solve_for_interpolant!(cache, y)
    z = copy(cache.z_workspace)  # Allocate separate storage for callable

    # Direct branching to Val literals for type stability
    extrapolation === :none      && return CubicInterpolant(cache, y, z, Val(:none))
    extrapolation === :constant  && return CubicInterpolant(cache, y, z, Val(:constant))
    extrapolation === :extension && return CubicInterpolant(cache, y, z, Val(:extension))
    error("unreachable")  # validation already done above
end

# Helper to solve z coefficients with BC dispatch (used by CubicInterpolant construction)
@inline function _solve_for_interpolant!(cache::CubicSplineCache{T,X,F,Nothing}, y::AbstractVector{T}) where {T, X, F}
    _solve_cubic_system!(cache.z_workspace, cache.d_workspace, cache, y)
end

@inline function _solve_for_interpolant!(cache::CubicSplineCache{T,X,F,PeriodicData{T}}, y::AbstractVector{T}) where {T, X, F}
    _solve_cubic_system_periodic!(cache.z_workspace, cache.d_workspace, cache, y)
end

"""
    cubic_interp(cache::CubicSplineCache{T}, y::AbstractVector{T}; extrapolation=:none) -> CubicInterpolant

Create a callable interpolant from a pre-built cache.

Useful when you want to create a periodic cache explicitly and reuse it with different y values.

# Arguments
- `cache::CubicSplineCache{T}`: Pre-computed cache (can be natural or periodic)
- `y::AbstractVector{T}`: Function values at grid points
- `extrapolation::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`)

# Returns
`CubicInterpolant` object that can be called with scalar or vector arguments.

# Example
```julia
cache = CubicSplineCache(x; bc=:periodic)
itp = cubic_interp(cache, y)
val = itp(0.5)  # Zero-allocation scalar call
```
"""
function cubic_interp(
    cache::CubicSplineCache{T},
    y::AbstractVector{T};
    extrapolation::Symbol=:none
) where {T<:AbstractFloat}
    extrapolation in (:none, :constant, :extension) || throw(ArgumentError("extrapolation must be :none, :constant, or :extension, got :$extrapolation"))
    _solve_for_interpolant!(cache, y)
    z = copy(cache.z_workspace)
    # Direct branching to Val literals for type stability
    extrapolation === :none      && return CubicInterpolant(cache, y, z, Val(:none))
    extrapolation === :constant  && return CubicInterpolant(cache, y, z, Val(:constant))
    extrapolation === :extension && return CubicInterpolant(cache, y, z, Val(:extension))
    error("unreachable")  # validation already done above
end

# Real wrapper for 2-argument form
# Uses _to_float from utils.jl to preserve Range structure for O(1) lookup
function cubic_interp(
    x::X,
    y::Y;
    bc::Symbol=:natural,
    extrapolation::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}}
    bc in (:natural, :periodic) || throw(ArgumentError("bc must be :natural or :periodic, got :$bc"))
    extrapolation in (:none, :constant, :extension) || throw(ArgumentError("extrapolation must be :none, :constant, or :extension, got :$extrapolation"))
    T = promote_type(TX, TY)
    FT = float(T)
    x_float = _to_float(x, FT)  # Preserves Range structure
    y_float = FT.(y)
    if bc == :periodic
        _check_periodic_endpoints(y_float)
        cache = autocache ? get_cubic_cache(x_float, Val(:periodic)) : CubicSplineCache(x_float; bc=:periodic)
    else
        cache = autocache ? get_cubic_cache(x_float, Val(:natural)) : CubicSplineCache(x_float)
    end

    # Pre-compute z coefficients (solve system once, then copy for storage)
    _solve_for_interpolant!(cache, y_float)
    z = copy(cache.z_workspace)  # Allocate separate storage for callable

    # Direct branching to Val literals for type stability
    extrapolation === :none      && return CubicInterpolant(cache, y_float, z, Val(:none))
    extrapolation === :constant  && return CubicInterpolant(cache, y_float, z, Val(:constant))
    extrapolation === :extension && return CubicInterpolant(cache, y_float, z, Val(:extension))
    error("unreachable")  # validation already done above
end

# ============================================================================
#                        GENERIC WRAPPERS - CONVENIENCE
#              Auto-promote Real types to Float (type conversion)
#                     Integer inputs -> Float outputs
# ============================================================================

# ========================================
# 3-argument form: Real type wrappers
# ========================================

# Allocating version - vector query
# Uses _to_float to preserve Range structure for O(1) lookup
function cubic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::AbstractVector{TQ};
    bc::Symbol=:natural,
    extrapolation::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp(_to_float(x, FT), FT.(y), FT.(x_query); bc=bc, extrapolation=extrapolation, autocache=autocache)
end

# Allocating version - scalar query
function cubic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::TQ;
    bc::Symbol=:natural,
    extrapolation::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp(_to_float(x, FT), FT.(y), FT(x_query); bc=bc, extrapolation=extrapolation, autocache=autocache)
end

# In-place version - vector query
function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::AbstractVector{TQ};
    bc::Symbol=:natural,
    extrapolation::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp!(output, _to_float(x, FT), FT.(y), FT.(x_query); bc=bc, extrapolation=extrapolation, autocache=autocache)
end

# In-place version - scalar query
function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::TQ;
    bc::Symbol=:natural,
    extrapolation::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp!(output, _to_float(x, FT), FT.(y), FT(x_query); bc=bc, extrapolation=extrapolation, autocache=autocache)
end
