"""
    cubic_interp.jl

Zero-allocation cubic spline interpolation with reusable LU factorization.

# Design Philosophy

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

# Type definitions in cubic_types.jl
# System solvers in cubic_solver.jl
# Evaluation functions in cubic_eval.jl

# ========================================
# CubicSplineCache Constructor
# ========================================

"""
    CubicSplineCache(x::AbstractVector{T}; bc::Symbol=:natural) where {T<:AbstractFloat}

Construct a cubic spline cache for grid points `x`.

Pre-computes and factorizes the tridiagonal matrix that depends only on x geometry.
This factorization can be reused for interpolating different y vectors.

# Arguments
- `x::AbstractVector{T}`: Grid points (must be sorted, length >= 3)
- `bc::Symbol=:natural`: Boundary condition (`:natural` or `:periodic`)

# Example
```julia
x = range(0.0, 1.0, 51)
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
    _validate_bc(bc)

    if bc == :periodic
        return _build_periodic_cache(x)
    else
        return _build_natural_cache(x)
    end
end

# ========================================
# In-Place Vector API
# ========================================

"""
    cubic_interp!(output, cache, y, x_query; extrap=:none)

In-place cubic spline interpolation using cached LU factorization.

Solves the tridiagonal system ONCE, then evaluates at all query points.

# Arguments
- `output::AbstractVector{T}`: Pre-allocated output buffer (modified in-place)
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points
- `extrap::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`, `:wrap`)
"""
@inline function cubic_interp!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BC},
    y::AbstractVector{T},
    x_query::AbstractVector{T};
    extrap::Symbol=:none
) where {T<:AbstractFloat, X, F, BC}
    @assert length(y) == length(cache.x) "y length must match cache grid"
    @assert length(output) == length(x_query) "output length must match x_query"

    z = _solve_system!(cache, y)

    @_dispatch_extrap extrap => ev begin
        _cubic_vector_loop!(output, cache, y, z, x_query, ev)
    end

    return output
end

"""
    cubic_interp!(output, x, y, x_query; bc=:natural, extrap=:none, autocache=true)

In-place cubic spline interpolation with optional automatic caching.
"""
@inline function cubic_interp!(
    output::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::AbstractVector{T};
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {T<:AbstractFloat}
    _validate_bc(bc)
    _validate_extrap(extrap)

    if bc == :periodic
        _check_periodic_endpoints(y)
        extrap = :wrap  # override extrap for periodic
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
        return cubic_interp!(output, cache, y, x_query; extrap=extrap)
    elseif autocache
        cache = get_cubic_cache(x, Val(:natural))
        return cubic_interp!(output, cache, y, x_query; extrap=extrap)
    else
        cache = CubicSplineCache(x)
        return cubic_interp!(output, cache, y, x_query; extrap=extrap)
    end
end

# Scalar query - zero allocation
@inline function cubic_interp!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BC},
    y::AbstractVector{T},
    x_query::T;
    extrap::Symbol=:none
) where {T<:AbstractFloat, X, F, BC}
    @assert length(output) >= 1 "output must have at least 1 element"
    output[1] = cubic_interp_scalar(cache, y, x_query; extrap=extrap)
    return output
end

@inline function cubic_interp!(
    output::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::T;
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {T<:AbstractFloat}
    @assert length(output) >= 1 "output must have at least 1 element"
    _validate_bc(bc)
    _validate_extrap(extrap)

    if bc == :periodic
        _check_periodic_endpoints(y)
        extrap = :wrap  # override extrap for periodic
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
    elseif autocache
        cache = get_cubic_cache(x, Val(:natural))
    else
        cache = CubicSplineCache(x)
    end

    output[1] = cubic_interp_scalar(cache, y, x_query; extrap=extrap)
    return output
end

# ========================================
# Allocating Vector API
# ========================================

"""
    cubic_interp(cache, y, x_query; extrap=:none) -> Vector{T}

Allocating version of cubic spline interpolation using cached LU factorization.

# Example
```julia
cache = CubicSplineCache(collect(range(0.0, 1.0, 51)))
y = sin.(cache.x)
result = cubic_interp(cache, y, [0.25, 0.5, 0.75])
```
"""
function cubic_interp(
    cache::CubicSplineCache{T},
    y::AbstractVector{T},
    x_query::AbstractVector{T};
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(x_query))
    cubic_interp!(output, cache, y, x_query; extrap=extrap)
    return output
end

"""
    cubic_interp(x, y, x_query; bc=:natural, extrap=:none, autocache=true) -> Vector{T}

Cubic spline interpolation with optional automatic caching.

# Extrapolation Modes
- `:none` (default): Throws DomainError if query point is outside domain
- `:constant`: Returns boundary values outside domain
- `:extension`: Extends boundary polynomial outside domain
- `:wrap`: Wraps coordinates to domain (for sawtooth/triangle patterns)
- For `bc=:periodic`: extrapolation is ignored (coordinates are always wrapped)

# Example
```julia
result = cubic_interp(x, y, x_query)              # Auto-cached (default)
result = cubic_interp(x, y, x_query; autocache=false)  # One-shot
result = cubic_interp(x, y, x_query; extrap=:extension)  # Extend beyond domain
```
"""
function cubic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::AbstractVector{T};
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {T<:AbstractFloat}
    _validate_bc(bc)
    _validate_extrap(extrap)

    if bc == :periodic
        _check_periodic_endpoints(y)
        extrap = :wrap  # override extrap for periodic
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
        return cubic_interp(cache, y, x_query; extrap=extrap)
    elseif autocache
        cache = get_cubic_cache(x, Val(:natural))
        return cubic_interp(cache, y, x_query; extrap=extrap)
    else
        cache = CubicSplineCache(x)
        return cubic_interp(cache, y, x_query; extrap=extrap)
    end
end

# Scalar query - zero allocation
cubic_interp(cache::CubicSplineCache{T}, y::AbstractVector{T},
             x_query::T; extrap::Symbol=:none) where {T<:AbstractFloat} =
    cubic_interp_scalar(cache, y, x_query; extrap=extrap)

function cubic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::T;
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {T<:AbstractFloat}
    _validate_bc(bc)
    _validate_extrap(extrap)

    if bc == :periodic
        _check_periodic_endpoints(y)
        extrap = :wrap  # override extrap for periodic
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
        return cubic_interp_scalar(cache, y, x_query; extrap=extrap)
    elseif autocache
        cache::CubicSplineCache{T} = get_cubic_cache(x, Val(:natural))
        return cubic_interp_scalar(cache, y, x_query; extrap=extrap)
    else
        cache = CubicSplineCache(x)
        return cubic_interp_scalar(cache, y, x_query; extrap=extrap)
    end
end

# ========================================
# CubicInterpolant Callable Methods
# ========================================

# CubicInterpolant struct defined in cubic_types.jl

# Scalar call - hot path (zero-allocation)
@inline function (itp::CubicInterpolant{T})(xi::T) where {T<:AbstractFloat}
    @boundscheck _check_domain(itp.cache, xi, itp.extrap)
    _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xi, itp.extrap)
end

# Real scalar wrapper
@inline function (itp::CubicInterpolant{T})(xi::S) where {T<:AbstractFloat, S<:Real}
    xi_t = T(xi)
    @boundscheck _check_domain(itp.cache, xi_t, itp.extrap)
    _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xi_t, itp.extrap)
end

# Vector call
function (itp::CubicInterpolant{T})(xi::AbstractVector{S}) where {T<:AbstractFloat, S<:Real}
    xi_typed = S === T ? xi : T.(xi)
    output = Vector{T}(undef, length(xi_typed))
    _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi_typed, itp.extrap)
    return output
end

function (itp::CubicInterpolant{T})(xi::AbstractVector{T}) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(xi))
    _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi, itp.extrap)
    return output
end

# In-place vector call
function (itp::CubicInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}) where {T<:AbstractFloat}
    @assert length(output) == length(xi) "output length must match xi length"
    _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi, itp.extrap)
    return output
end

function (itp::CubicInterpolant{T})(output::AbstractVector, xi::AbstractVector{S}) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = T.(xi)
    _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi_typed, itp.extrap)
    return output
end

# ========================================
# 2-Argument Form: Return CubicInterpolant
# ========================================

"""
    cubic_interp(x, y; bc=:natural, extrap=:none, autocache=true) -> CubicInterpolant

Create a callable interpolant for broadcast fusion and reuse.

Pre-computes second derivative coefficients z ONCE at construction time,
enabling true zero-allocation scalar evaluations in broadcast operations.

# Example
```julia
itp = cubic_interp(x, y)           # Pre-computes z coefficients
val = itp(0.5)                      # Scalar (zero-allocation)
vals = itp.(query_points)           # Broadcast
result = @. coef * itp(rho) * ne    # Fused broadcast
```
"""
function cubic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {T<:AbstractFloat}
    _validate_bc(bc)

    if bc == :periodic
        _check_periodic_endpoints(y)
        extrap = :wrap  # override extrap for periodic
        cache = autocache ? get_cubic_cache(x, Val(:periodic)) : CubicSplineCache(x; bc=:periodic)
    else
        cache = autocache ? get_cubic_cache(x, Val(:natural)) : CubicSplineCache(x)
    end

    _solve_system!(cache, y)
    z = copy(cache.z_workspace)

    @_dispatch_extrap extrap => ev begin
        return CubicInterpolant(cache, y, z, ev)
    end
end

"""
    cubic_interp(cache, y; extrap=:none) -> CubicInterpolant

Create a callable interpolant from a pre-built cache.
"""
function cubic_interp(
    cache::CubicSplineCache{T},
    y::AbstractVector{T};
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    _solve_system!(cache, y)
    z = copy(cache.z_workspace)

    @_dispatch_extrap extrap => ev begin
        return CubicInterpolant(cache, y, z, ev)
    end
end

# Real wrapper for 2-argument form
function cubic_interp(
    x::X,
    y::Y;
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}}
    _validate_bc(bc)

    T = promote_type(TX, TY)
    FT = float(T)
    x_float = _to_float(x, FT)
    y_float = FT.(y)

    if bc == :periodic
        _check_periodic_endpoints(y_float)
        extrap = :wrap  # override extrap for periodic
        cache = autocache ? get_cubic_cache(x_float, Val(:periodic)) : CubicSplineCache(x_float; bc=:periodic)
    else
        cache = autocache ? get_cubic_cache(x_float, Val(:natural)) : CubicSplineCache(x_float)
    end

    _solve_system!(cache, y_float)
    z = copy(cache.z_workspace)

    @_dispatch_extrap extrap => ev begin
        return CubicInterpolant(cache, y_float, z, ev)
    end
end

# ========================================
# Generic Real Type Wrappers
# ========================================

# Allocating - vector query
function cubic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::AbstractVector{TQ};
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp(_to_float(x, FT), FT.(y), FT.(x_query); bc=bc, extrap=extrap, autocache=autocache)
end

# Allocating - scalar query
function cubic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::TQ;
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp(_to_float(x, FT), FT.(y), FT(x_query); bc=bc, extrap=extrap, autocache=autocache)
end

# In-place - vector query
@inline function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::AbstractVector{TQ};
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp!(output, _to_float(x, FT), FT.(y), FT.(x_query); bc=bc, extrap=extrap, autocache=autocache)
end

# In-place - scalar query
@inline function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::TQ;
    bc::Symbol=:natural,
    extrap::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp!(output, _to_float(x, FT), FT.(y), FT(x_query); bc=bc, extrap=extrap, autocache=autocache)
end
