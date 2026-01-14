# ========================================
# Cubic Spline Oneshot API
# ========================================
# Zero-allocation cubic spline interpolation functions.
# CubicSplineCache constructor is in cubic_cache.jl.
# Type definitions in cubic_types.jl.
# System solvers in cubic_solver.jl.
# Evaluation functions in cubic_eval.jl.

# ========================================
# In-Place Vector API
# ========================================

"""
    cubic_interp!(output, cache, y, x_query; extrap=:none, deriv=0)

In-place cubic spline interpolation using cached LU factorization.

Solves the tridiagonal system ONCE, then evaluates at all query points.
Thread-safe: workspaces allocated from task-local pool.

# Arguments
- `output::AbstractVector{T}`: Pre-allocated output buffer (modified in-place)
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points
- `extrap::Symbol=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`, `:wrap`)
- `deriv::Int=0`: Derivative order (0=value, 1=first derivative, 2=second derivative)
"""
@inline @with_pool pool function cubic_interp!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BC},
    y::AbstractVector{T},
    x_query::AbstractVector{T};
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:AbstractFloat, X, F, BC}
    @assert length(y) == length(cache.x) "y length must match cache grid"
    @assert length(output) == length(x_query) "output length must match x_query"

    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            _cubic_vector_loop!(output, cache, y, z, x_query, ev, op)
        end
    end

    return output
end

# ========================================
# Core Fallback Functions (Type-Stable Dispatch)
# ========================================
#
# These are the two core implementations that all Symbol-based APIs
# dispatch to after normalizing BC to concrete types.
#
# Key insight: LU factorization depends only on BC **types** (Deriv1 vs Deriv2),
# not BC **values**. So autocache stores by (x, L_type, R_type),
# and we apply actual BC values at solve time via _solve_system!(cache, y, bc_tuple).

"""
Core implementation for BCPair boundary conditions (vector output).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
"""
@inline @with_pool pool function _cubic_interp_bcpair!(
    output::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::AbstractVector{T},
    bc::BCPair{T,L,R},
    extrap::Symbol,
    autocache::Bool,
    op::O
) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, O<:AbstractEvalOp}
    @assert length(y) == length(x) "y length must match x"
    @assert length(output) == length(x_query) "output length must match x_query"

    cache = _get_cubic_cache(x, bc, autocache)
    z = similar!(pool, y)
    _solve_system!(z, cache, y, bc)

    @_dispatch_extrap extrap => ev begin
        _cubic_vector_loop!(output, cache, y, z, x_query, ev, op)
    end

    return output
end

"""
Core implementation for BCPair boundary conditions (scalar query).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
"""
@inline @with_pool pool function _cubic_interp_bcpair_scalar(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::T,
    bc::BCPair{T,L,R},
    extrap::Symbol,
    autocache::Bool,
    op::O
) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}, O<:AbstractEvalOp}
    # cache = _get_cache_and_solve!(x, y, bc, autocache)
    # z = cache.z_workspace

    tmp_z = similar!(pool, y)
    cache = _get_cubic_cache(x, bc, autocache)
    _solve_system!(tmp_z, cache, y, bc)

    @_dispatch_extrap extrap => ev begin
        _check_domain(cache.x, x_query, ev)
        return _eval_with_bc(cache, y, tmp_z, x_query, ev, op)
    end
end

"""
Core implementation for PeriodicBC boundary conditions (vector output).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
"""
@inline @with_pool pool function _cubic_interp_periodic!(
    output::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::AbstractVector{T},
    autocache::Bool,
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    @assert length(y) == length(x) "y length must match x"
    @assert length(output) == length(x_query) "output length must match x_query"

    _check_periodic_endpoints(y)
    cache = _get_cubic_cache(x, PeriodicBC{T}(), autocache)
    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    # Periodic BC always uses :wrap extrapolation
    @_dispatch_extrap :wrap => ev begin
        _cubic_vector_loop!(output, cache, y, z, x_query, ev, op)
    end

    return output
end

"""
Core implementation for PeriodicBC boundary conditions (scalar query).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
"""
@inline @with_pool pool function _cubic_interp_periodic_scalar(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::T,
    autocache::Bool,
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    _check_periodic_endpoints(y)
    cache = _get_cubic_cache(x, PeriodicBC{T}(), autocache)
    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    # Periodic BC always uses :wrap extrapolation
    @_dispatch_extrap :wrap => ev begin
        _check_domain(cache.x, x_query, ev)
        return _eval_with_bc(cache, y, z, x_query, ev, op)
    end
end

"""
    cubic_interp!(output, x, y, x_query; bc=NaturalBC(), extrap=:none, autocache=true, deriv=0)

In-place cubic spline interpolation with optional automatic caching.
"""
@inline function cubic_interp!(
    output::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::AbstractVector{T};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0
) where {T<:AbstractFloat}
    _validate_extrap(extrap)

    @_dispatch_deriv deriv => op begin
        # Periodic BC
        if _is_periodic_bc(bc)
            return _cubic_interp_periodic!(output, x, y, x_query, autocache, op)
        end

        # Normalize to BCPair and dispatch to core
        bc_pair = _normalize_bc(bc, T)
        return _cubic_interp_bcpair!(output, x, y, x_query, bc_pair, extrap, autocache, op)
    end
end


# Scalar query - zero allocation
@inline function cubic_interp!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BC},
    y::AbstractVector{T},
    x_query::T;
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:AbstractFloat, X, F, BC}
    @assert length(output) >= 1 "output must have at least 1 element"
    output[1] = cubic_interp_scalar(cache, y, x_query; extrap=extrap, deriv=deriv)
    return output
end

@inline function cubic_interp!(
    output::AbstractVector{T},
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::T;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0
) where {T<:AbstractFloat}
    @assert length(output) >= 1 "output must have at least 1 element"

    _validate_extrap(extrap)

    @_dispatch_deriv deriv => op begin
        if _is_periodic_bc(bc)
            output[1] = _cubic_interp_periodic_scalar(x, y, x_query, autocache, op)
        else
            bc_pair = _normalize_bc(bc, T)
            output[1] = _cubic_interp_bcpair_scalar(x, y, x_query, bc_pair, extrap, autocache, op)
        end
    end

    return output
end

# ========================================
# Allocating Vector API
# ========================================

"""
    cubic_interp(cache, y, x_query; extrap=:none, deriv=0) -> Vector{T}

Allocating version of cubic spline interpolation using cached LU factorization.

# Arguments
- `deriv::Int=0`: Derivative order (0=value, 1=first derivative, 2=second derivative)

# Example
```julia
cache = CubicSplineCache(collect(range(0.0, 1.0, 51)))
y = sin.(cache.x)
result = cubic_interp(cache, y, [0.25, 0.5, 0.75])
derivs = cubic_interp(cache, y, [0.25, 0.5, 0.75]; deriv=1)  # First derivative
```
"""
function cubic_interp(
    cache::CubicSplineCache{T},
    y::AbstractVector{T},
    x_query::AbstractVector{T};
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(x_query))
    cubic_interp!(output, cache, y, x_query; extrap=extrap, deriv=deriv)
    return output
end

"""
    cubic_interp(x, y, x_query; bc=NaturalBC(), extrap=:none, autocache=true, deriv=0) -> Vector{T}

Cubic spline interpolation with optional automatic caching.

# Arguments
- `deriv::Int=0`: Derivative order (0=value, 1=first derivative, 2=second derivative)

# Extrapolation Modes
- `:none` (default): Throws DomainError if query point is outside domain
- `:constant`: Returns boundary values outside domain (0 for derivatives)
- `:extension`: Extends boundary polynomial outside domain
- `:wrap`: Wraps coordinates to domain (for sawtooth/triangle patterns)
- For `bc=PeriodicBC()`: extrapolation is ignored (coordinates are always wrapped)

# Example
```julia
result = cubic_interp(x, y, x_query)                     # Auto-cached (default)
derivs = cubic_interp(x, y, x_query; deriv=1)            # First derivative
result = cubic_interp(x, y, x_query; extrap=:extension)  # Extend beyond domain
```
"""
function cubic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::AbstractVector{T};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0
) where {T<:AbstractFloat}
    _validate_extrap(extrap)

    output = Vector{T}(undef, length(x_query))

    @_dispatch_deriv deriv => op begin
        if _is_periodic_bc(bc)
            return _cubic_interp_periodic!(output, x, y, x_query, autocache, op)
        end

        bc_pair = _normalize_bc(bc, T)
        return _cubic_interp_bcpair!(output, x, y, x_query, bc_pair, extrap, autocache, op)
    end
end

# Scalar query - zero allocation
cubic_interp(cache::CubicSplineCache{T}, y::AbstractVector{T},
             x_query::T; extrap::Symbol=:none, deriv::Int=0) where {T<:AbstractFloat} =
    cubic_interp_scalar(cache, y, x_query; extrap=extrap, deriv=deriv)

function cubic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_query::T;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0
) where {T<:AbstractFloat}
    _validate_extrap(extrap)

    @_dispatch_deriv deriv => op begin
        if _is_periodic_bc(bc)
            return _cubic_interp_periodic_scalar(x, y, x_query, autocache, op)
        end

        bc_pair = _normalize_bc(bc, T)
        return _cubic_interp_bcpair_scalar(x, y, x_query, bc_pair, extrap, autocache, op)
    end
end

# ========================================
# Generic Real Type Wrappers
# ========================================
# Note: CubicInterpolant callable methods and 2-arg form are in cubic_interpolant.jl

# Allocating - vector query
function cubic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::AbstractVector{TQ};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp(_to_float(x, FT), FT.(y), FT.(x_query); bc=bc, extrap=extrap, autocache=autocache, deriv=deriv)
end

# Allocating - scalar query
function cubic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::TQ;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp(_to_float(x, FT), FT.(y), FT(x_query); bc=bc, extrap=extrap, autocache=autocache, deriv=deriv)
end

# In-place - vector query
@inline function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::AbstractVector{TQ};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp!(output, _to_float(x, FT), FT.(y), FT.(x_query); bc=bc, extrap=extrap, autocache=autocache, deriv=deriv)
end

# In-place - scalar query
@inline function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{TX},
    y::AbstractVector{TY},
    x_query::TQ;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0
) where {TX<:Real, TY<:Real, TQ<:Real}
    FT = float(promote_type(TX, TY, TQ))
    return cubic_interp!(output, _to_float(x, FT), FT.(y), FT(x_query); bc=bc, extrap=extrap, autocache=autocache, deriv=deriv)
end
