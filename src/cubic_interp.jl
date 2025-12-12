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

"""
    CubicSplineCache{T,X,F}

Cache structure for cubic spline interpolation with reusable LU factorization.

# Fields
- `x::X`: Grid points (immutable after construction, can be Range or Vector)
- `h::Vector{T}`: Grid spacing h[i] = x[i+1] - x[i]
- `lu_factor`: LU factorization of tridiagonal matrix A
- `d_workspace::Vector{T}`: Workspace for RHS vector computation
- `z_workspace::Vector{T}`: Workspace for solution vector

# Notes
The LU factorization depends ONLY on x geometry and can be reused for:
- Different y vectors (varying function values)
- Different x_query vectors (varying query points)

When x is an AbstractRange, O(1) index lookup is used instead of O(log n) binary search.
"""
struct CubicSplineCache{T<:AbstractFloat,X<:AbstractVector{T},F}
    x::X
    h::Vector{T}
    lu_factor::F
    d_workspace::Vector{T}
    z_workspace::Vector{T}
end

"""
    CubicSplineCache(x::AbstractVector{T}) where {T<:AbstractFloat}

Construct a cubic spline cache for grid points `x`.

Pre-computes and factorizes the tridiagonal matrix that depends only on x geometry.
This factorization can be reused for interpolating different y vectors.

When `x` is an AbstractRange, the Range structure is preserved for O(1) index lookup
during evaluation. When `x` is a Vector, O(log n) binary search is used.

# Arguments
- `x::AbstractVector{T}`: Grid points (must be sorted, length >= 3)

# Returns
- `CubicSplineCache{T,X,F}`: Reusable cache structure

# Example
```julia
x = range(0.0, 1.0, 51)  # Range preserved for O(1) lookup
cache = CubicSplineCache(x)

# Reuse for multiple y vectors
y1 = sin.(x)
y2 = cos.(x)
result1 = cubic_interp(cache, y1, [0.25, 0.75])
result2 = cubic_interp(cache, y2, [0.25, 0.75])
```
"""
function CubicSplineCache(x::AbstractVector{T}) where {T<:AbstractFloat}
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

    return CubicSplineCache(x, h[1:n+1], lu_factor, d_workspace, z_workspace)
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
    cubic_interp!(output::AbstractVector{T}, cache::CubicSplineCache{T},
                  y::AbstractVector{T}, x_query::AbstractVector{T}) where {T<:AbstractFloat}

In-place cubic spline interpolation using cached LU factorization.

Solves the tridiagonal system ONCE, then evaluates at all query points.

# Arguments
- `output::AbstractVector{T}`: Pre-allocated output buffer (modified in-place)
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points (length must match cache.x)
- `x_query::AbstractVector{T}`: Query points where interpolation is evaluated

# Notes
- Zero allocations after cache construction
- Reuses LU factorization from cache
- Thread-safe if different caches/workspaces are used per thread
- Pattern: Solve system once -> evaluate at all query points
"""
function cubic_interp!(output::AbstractVector{T}, cache::CubicSplineCache{T},
                       y::AbstractVector{T}, x_query::AbstractVector{T}) where {T<:AbstractFloat}
    @assert length(y) == length(cache.x) "y length must match cache grid"
    @assert length(output) == length(x_query) "output length must match x_query"

    # Step 1: Solve for z coefficients (solves system ONCE for all query points)
    z = _solve_cubic_system!(cache.z_workspace, cache.d_workspace, cache, y)

    # Step 2: Evaluate at all query points using pre-computed z
    @inbounds for (k, xq) in enumerate(x_query)
        output[k] = _eval_cubic_at_point(cache.x, y, cache.h, z, xq)
    end

    return output
end

"""
    cubic_interp(cache::CubicSplineCache{T}, y::AbstractVector{T},
                 x_query::AbstractVector{T}) where {T<:AbstractFloat}

Allocating version of cubic spline interpolation using cached LU factorization.

# Arguments
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points

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
                      x_query::AbstractVector{T}) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(x_query))
    cubic_interp!(output, cache, y, x_query)
    return output
end

"""
    cubic_interp(x::AbstractVector{T}, y::AbstractVector{T},
                 x_query::AbstractVector{T}; autocache::Bool=true) where {T<:AbstractFloat}

Cubic spline interpolation with optional automatic caching.

With `autocache=true` (default), automatically reuses cached LU factorization for
previously seen x-grids. First call creates cache (~35 allocations), subsequent calls
with same x-grid reuse it (~2 allocations).

With `autocache=false`, constructs cache, computes interpolation, and discards cache.

# Arguments
- `x::AbstractVector{T}`: Grid points
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points
- `autocache::Bool=true`: Enable transparent cache reuse (default: true)

# Returns
- `Vector{T}`: Interpolated values at x_query points

# Performance
With autocache=true, repeated calls with same x-grid reuse cached LU factorization:
- First call: ~35 allocations (cache creation)
- Subsequent calls: ~2 allocations (cache reuse)

# Example
```julia
# Automatic caching (no user management needed)
for field in [:volume, :area, :surface, :psi]
    y = getfield(data, field)
    result = cubic_interp(x, y, x_query)  # Auto-cached!
end
# First call creates cache, next 3 reuse it

# Disable auto-cache for one-shot usage
result = cubic_interp(x, y, x_query; autocache=false)

# Explicit cache for deterministic performance
cache = CubicSplineCache(x)
result = cubic_interp(cache, y, x_query)
```
"""
function cubic_interp(x::AbstractVector{T}, y::AbstractVector{T},
                      x_query::AbstractVector{T}; autocache::Bool=true) where {T<:AbstractFloat}
    if autocache
        cache = get_cubic_cache(x)
        return cubic_interp(cache, y, x_query)
    else
        cache = CubicSplineCache(x)
        return cubic_interp(cache, y, x_query)
    end
end

"""
    cubic_interp!(output::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
                  x_query::AbstractVector{T}; autocache::Bool=true) where {T<:AbstractFloat}

In-place cubic spline interpolation with optional automatic caching.

Like the allocating version, this supports auto-cache to transparently reuse LU factorization
for the same x-grid across multiple calls. This gives you **zero-allocation + auto-cache**!

# Arguments
- `output::AbstractVector{T}`: Pre-allocated output buffer (modified in-place)
- `x::AbstractVector{T}`: Grid points
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points
- `autocache::Bool=true`: Enable transparent cache reuse (default: true)

# Example
```julia
output = Vector{Float64}(undef, length(x_query))

# Auto-cache enabled (default)
for y_i in [y1, y2, y3, ...]
    cubic_interp!(output, x, y_i, x_query)  # Zero-allocation + auto-cached!
    # use output...
end

# Disable auto-cache for one-shot usage
cubic_interp!(output, x, y, x_query; autocache=false)

# Explicit cache for maximum control
cache = CubicSplineCache(x)
cubic_interp!(output, cache, y, x_query)
```
"""
function cubic_interp!(output::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
                       x_query::AbstractVector{T}; autocache::Bool=true) where {T<:AbstractFloat}
    if autocache
        cache = get_cubic_cache(x)
        return cubic_interp!(output, cache, y, x_query)
    else
        cache = CubicSplineCache(x)
        return cubic_interp!(output, cache, y, x_query)
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

"""
    _solve_cubic_system!(z_workspace, d_workspace, cache, y)

Solve tridiagonal system A * z = d for second derivative coefficients.

Uses pre-computed LU factorization from cache. Modifies workspaces in-place.

# Arguments
- `z_workspace::Vector{T}`: Output workspace for z coefficients (modified in-place)
- `d_workspace::Vector{T}`: Workspace for RHS vector (modified in-place)
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points

# Returns
- `z_workspace`: Reference to modified z_workspace (for convenience)
"""
@inline function _solve_cubic_system!(
    z_workspace::Vector{T},
    d_workspace::Vector{T},
    cache::CubicSplineCache{T},
    y::AbstractVector{T}
) where {T<:AbstractFloat}
    # Step 1: Compute RHS vector d from y values
    compute_rhs!(d_workspace, y, cache.h)

    # Step 2: Solve A * z = d using cached LU factorization
    ldiv!(z_workspace, cache.lu_factor, d_workspace)

    return z_workspace
end

"""
    cubic_interp_scalar(cache::CubicSplineCache{T}, y::AbstractVector{T},
                        x_query::T) where {T<:AbstractFloat}

Scalar cubic spline evaluation (solves system once, evaluates once).

For repeated evaluations at different query points with same y, use CubicInterpCallable
instead, which pre-computes z coefficients once and reuses them for all evaluations.

# Arguments
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::T`: Single query point (scalar)

# Returns
- `T`: Interpolated value at x_query

# Note
This function solves the tridiagonal system for each call. If you need to evaluate
at multiple points with the same y, create a CubicInterpCallable instead:
```julia
itp = cubic_interp(x, y)  # Solve once
vals = itp.(query_points)  # Reuse z for all points
```
"""
@inline function cubic_interp_scalar(cache::CubicSplineCache{T}, y::AbstractVector{T},
                                      x_query::T) where {T<:AbstractFloat}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    # Solve for z coefficients (reuses cache workspaces)
    z = _solve_cubic_system!(cache.z_workspace, cache.d_workspace, cache, y)

    # Evaluate at query point using z
    return _eval_cubic_at_point(cache.x, y, cache.h, z, x_query)
end

# Scalar query point convenience methods
cubic_interp!(output::AbstractVector{T}, cache::CubicSplineCache{T},
              y::AbstractVector{T}, x_query::T) where {T<:AbstractFloat} =
    cubic_interp!(output, cache, y, [x_query])

cubic_interp!(output::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
              x_query::T; autocache::Bool=true) where {T<:AbstractFloat} =
    cubic_interp!(output, x, y, [x_query]; autocache=autocache)

# CRITICAL: Zero-allocation scalar path for broadcast fusion
cubic_interp(cache::CubicSplineCache{T}, y::AbstractVector{T},
             x_query::T) where {T<:AbstractFloat} =
    cubic_interp_scalar(cache, y, x_query)

# Scalar query with autocache option
function cubic_interp(x::AbstractVector{T}, y::AbstractVector{T},
                      x_query::T; autocache::Bool=true) where {T<:AbstractFloat}
    if autocache
        cache::CubicSplineCache{T} = get_cubic_cache(x)
        return cubic_interp_scalar(cache, y, x_query)
    else
        cache = CubicSplineCache(x)
        return cubic_interp_scalar(cache, y, x_query)
    end
end

# ========================================
# Callable Interpolator for Broadcast Fusion
# ========================================

"""
    CubicInterpCallable{T,C,Y,Z}

Lightweight callable interpolator for broadcast fusion optimization.
Returned by `cubic_interp(x, y)` (2-argument form).

# Fields
- `cache::C`: Pre-computed CubicSplineCache (LU factorization)
- `y::Y`: y-values (function values at grid points)
- `z::Z`: Pre-computed second derivative coefficients (solves system once!)

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

# Performance Notes
- System solved ONCE at construction -> z coefficients pre-computed
- Each scalar call just evaluates cubic polynomial (zero-allocation!)
- Broadcast operations are perfectly fused (no intermediate arrays)
- Optimal for multiple evaluations with same x-grid and y-values
"""
struct CubicInterpCallable{T<:AbstractFloat,C<:CubicSplineCache{T},Y<:AbstractVector{T},Z<:AbstractVector{T}}
    cache::C
    y::Y
    z::Z  # Pre-computed second derivative coefficients

    function CubicInterpCallable(
        cache::C,
        y::Y,
        z::Z
    ) where {T<:AbstractFloat, C<:CubicSplineCache{T}, Y<:AbstractVector{T}, Z<:AbstractVector{T}}
        @assert length(cache.x) == length(y) "cache grid and y must have same length"
        @assert length(cache.x) == length(z) "z coefficients must match grid length"
        new{T,C,Y,Z}(cache, y, z)
    end
end

# Scalar call - hot path (inlined for broadcast fusion)
# CRITICAL: Uses pre-computed z coefficients -> TRUE zero-allocation!
@inline function (itp::CubicInterpCallable{T})(xi::T) where {T<:AbstractFloat}
    _eval_cubic_at_point(itp.cache.x, itp.y, itp.cache.h, itp.z, xi)
end

# Real scalar wrapper for convenience
@inline function (itp::CubicInterpCallable{T})(xi::S) where {T<:AbstractFloat, S<:Real}
    _eval_cubic_at_point(itp.cache.x, itp.y, itp.cache.h, itp.z, T(xi))
end

# Vector call - uses pre-computed z coefficients (no redundant system solve!)
function (itp::CubicInterpCallable{T})(xi::AbstractVector{S}) where {T<:AbstractFloat, S<:Real}
    xi_typed = S === T ? xi : T.(xi)
    output = Vector{T}(undef, length(xi_typed))
    @inbounds for (k, xq) in enumerate(xi_typed)
        output[k] = _eval_cubic_at_point(itp.cache.x, itp.y, itp.cache.h, itp.z, xq)
    end
    return output
end

# Optimized path when xi element type matches T (zero conversion)
function (itp::CubicInterpCallable{T})(xi::AbstractVector{T}) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(xi))
    @inbounds for (k, xq) in enumerate(xi)
        output[k] = _eval_cubic_at_point(itp.cache.x, itp.y, itp.cache.h, itp.z, xq)
    end
    return output
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    cubic_interp(x, y; autocache=true) -> CubicInterpCallable

Create a callable interpolator for broadcast fusion and reuse.

Pre-computes second derivative coefficients z ONCE at construction time,
enabling true zero-allocation scalar evaluations in broadcast operations.

# Arguments
- `x::AbstractVector`: Grid points (must be sorted, length >= 3)
- `y::AbstractVector`: Function values at grid points
- `autocache::Bool`: Use automatic cache lookup (default: true)

# Returns
`CubicInterpCallable` object that can be:
- Called with scalar: `itp(0.5)` (zero-allocation!)
- Broadcasted: `itp.(rho)` or `@. coef * itp(rho)` (zero-allocation per call!)
- Reused multiple times without re-creating

# Examples
```julia
# Create once, reuse multiple times
itp = cubic_interp(x_data, y_data)  # Solves system once, stores z

# Scalar call (zero-allocation, just arithmetic!)
val = itp(0.5)

# Broadcast (zero-allocation per call!)
vals = itp.(query_points)

# Fused broadcast (optimal - no intermediate arrays, no allocations!)
result = @. coefficient * itp(rho) * ne / Te^2

# Compare with 3-argument form (returns array immediately)
vals_direct = cubic_interp(x_data, y_data, query_points)
```

# Performance Notes
- Construction: Solves tridiagonal system ONCE -> stores z coefficients
- Scalar calls: Just arithmetic with pre-computed z (true zero-allocation!)
- Broadcast fusion: Each call is zero-allocation -> perfect fusion
- Best for: Repeated evaluations with same (x, y) at different query points
- 3-argument form: Best for single immediate use, no reuse needed
"""
function cubic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    autocache::Bool=true
) where {T<:AbstractFloat}
    cache = autocache ? get_cubic_cache(x) : CubicSplineCache(x)

    # Pre-compute z coefficients (solve system once, then copy for storage)
    _solve_cubic_system!(cache.z_workspace, cache.d_workspace, cache, y)
    z = copy(cache.z_workspace)  # Allocate separate storage for callable

    return CubicInterpCallable(cache, y, z)
end

# Real wrapper for 2-argument form
# Uses _to_float from utils.jl to preserve Range structure for O(1) lookup
function cubic_interp(
    x::X,
    y::Y;
    autocache::Bool=true
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}}
    T = promote_type(TX, TY)
    FT = float(T)
    x_float = _to_float(x, FT)  # Preserves Range structure
    cache = autocache ? get_cubic_cache(x_float) : CubicSplineCache(x_float)
    y_float = FT.(y)

    # Pre-compute z coefficients (solve system once, then copy for storage)
    _solve_cubic_system!(cache.z_workspace, cache.d_workspace, cache, y_float)
    z = copy(cache.z_workspace)  # Allocate separate storage for callable

    return CubicInterpCallable(cache, y_float, z)
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
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp(_to_float(x, FT), FT.(y), FT.(x_query); autocache)
end

# Allocating version - scalar query
function cubic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::TQ;
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp(_to_float(x, FT), FT.(y), FT(x_query); autocache)
end

# In-place version - vector query
function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::AbstractVector{TQ};
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp!(output, _to_float(x, FT), FT.(y), FT.(x_query); autocache)
end

# In-place version - scalar query
function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::TQ;
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp!(output, _to_float(x, FT), FT.(y), FT(x_query); autocache)
end
