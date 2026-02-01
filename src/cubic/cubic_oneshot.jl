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
    cubic_interp!(output, cache, y, x_query; extrap=:none, deriv=0, search=Binary())

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
- `search::AbstractSearchPolicy=Binary()`: Search algorithm for interval finding
"""
@inline @with_pool pool function cubic_interp!(
    output::AbstractVector{Tv},
    cache::CubicSplineCache{Tg,X,F,BC},
    y::AbstractVector{Tv},
    x_query::AbstractVector{Tg};
    extrap::Symbol=:none,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv, X, F, BC}
    @assert length(y) == length(cache.x) "y length must match cache grid"
    @assert length(output) == length(x_query) "output length must match x_query"

    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            _cubic_vector_loop!(output, cache, y, z, x_query, ev, op, searcher)
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
Core implementation for BCPair boundary conditions (vector query).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.

Type-Free design: handles both concrete (Deriv1{T}) and lazy (PolyFit{D}) types.
- Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair)
- Solve uses original BC for proper RHS materialization (PolyFit materializes via compute_rhs!)
"""
@inline @with_pool pool function _cubic_interp_bcpair!(
    output::AbstractVector{Tv},
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::AbstractVector{Tg},
    bc::BCPair{L,R},
    extrap::Symbol,
    autocache::Bool,
    op::O,
    searcher::S
) where {Tg<:AbstractFloat, Tv, L<:PointBC, R<:PointBC, O<:AbstractEvalOp, S<:Searcher}
    @assert length(y) == length(x) "y length must match x"
    @assert length(output) == length(x_query) "output length must match x_query"

    # Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair internally)
    cache = _get_cubic_cache(x, bc, autocache)
    z = similar!(pool, y)
    # Solve uses original BC for proper RHS materialization
    _solve_system!(z, cache, y, bc)

    @_dispatch_extrap extrap => ev begin
        _cubic_vector_loop!(output, cache, y, z, x_query, ev, op, searcher)
    end

    return output
end

"""
Core implementation for BCPair boundary conditions (scalar query).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.

Type-Free design: handles both concrete (Deriv1{T}) and lazy (PolyFit{D}) types.
"""
@inline @with_pool pool function _cubic_interp_bcpair_scalar(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::Tg,
    bc::BCPair{L,R},
    extrap::Symbol,
    autocache::Bool,
    op::O,
    searcher::S
) where {Tg<:AbstractFloat, Tv, L<:PointBC, R<:PointBC, O<:AbstractEvalOp, S<:Searcher}
    tmp_z = similar!(pool, y)
    # Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair internally)
    cache = _get_cubic_cache(x, bc, autocache)
    # Solve uses original BC for proper RHS materialization
    _solve_system!(tmp_z, cache, y, bc)

    @_dispatch_extrap extrap => ev begin
        _check_domain(cache.x, x_query, ev)
        return _eval_with_bc(cache, y, tmp_z, x_query, ev, op, searcher)
    end
end

"""
Core implementation for PeriodicBC boundary conditions (vector output).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
"""
@inline @with_pool pool function _cubic_interp_periodic!(
    output::AbstractVector{Tv},
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::AbstractVector{Tg},
    autocache::Bool,
    op::O,
    searcher::S
) where {Tg<:AbstractFloat, Tv, O<:AbstractEvalOp, S<:Searcher}
    @assert length(y) == length(x) "y length must match x"
    @assert length(output) == length(x_query) "output length must match x_query"

    _check_periodic_endpoints(y)
    cache = _get_cubic_cache(x, PeriodicBC(), autocache)
    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    # Periodic BC always uses :wrap extrapolation
    @_dispatch_extrap :wrap => ev begin
        _cubic_vector_loop!(output, cache, y, z, x_query, ev, op, searcher)
    end

    return output
end

"""
Core implementation for PeriodicBC boundary conditions (scalar query).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
"""
@inline @with_pool pool function _cubic_interp_periodic_scalar(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::Tg,
    autocache::Bool,
    op::O,
    searcher::S
) where {Tg<:AbstractFloat, Tv, O<:AbstractEvalOp, S<:Searcher}
    _check_periodic_endpoints(y)
    cache = _get_cubic_cache(x, PeriodicBC(), autocache)
    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    # Periodic BC always uses :wrap extrapolation
    @_dispatch_extrap :wrap => ev begin
        _check_domain(cache.x, x_query, ev)
        return _eval_with_bc(cache, y, z, x_query, ev, op, searcher)
    end
end

"""
    cubic_interp!(output, x, y, x_query; bc=NaturalBC(), extrap=:none, autocache=true, deriv=0, search=Binary())

In-place cubic spline interpolation with optional automatic caching.
"""
@inline function cubic_interp!(
    output::AbstractVector{Tv},
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::AbstractVector{Tg};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    _validate_extrap(extrap)

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        # Periodic BC
        if _is_periodic_bc(bc)
            return _cubic_interp_periodic!(output, x, y, x_query, autocache, op, searcher)
        end

        # Normalize to BCPair and dispatch to core
        bc_pair = _normalize_bc(bc, Tv)
        return _cubic_interp_bcpair!(output, x, y, x_query, bc_pair, extrap, autocache, op, searcher)
    end
end


# Scalar query - zero allocation
@inline function cubic_interp!(
    output::AbstractVector{Tv},
    cache::CubicSplineCache{Tg,X,F,BC},
    y::AbstractVector{Tv},
    x_query::Tg;
    extrap::Symbol=:none,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv, X, F, BC}
    @assert length(output) >= 1 "output must have at least 1 element"
    output[1] = cubic_interp_scalar(cache, y, x_query; extrap=extrap, deriv=deriv, search=search)
    return output
end

@inline function cubic_interp!(
    output::AbstractVector{Tv},
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::Tg;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    @assert length(output) >= 1 "output must have at least 1 element"

    _validate_extrap(extrap)

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        if _is_periodic_bc(bc)
            output[1] = _cubic_interp_periodic_scalar(x, y, x_query, autocache, op, searcher)
        else
            bc_pair = _normalize_bc(bc, Tv)
            output[1] = _cubic_interp_bcpair_scalar(x, y, x_query, bc_pair, extrap, autocache, op, searcher)
        end
    end

    return output
end

# ========================================
# Allocating Vector API
# ========================================

"""
    cubic_interp(cache, y, x_query; extrap=:none, deriv=0, search=Binary()) -> Vector{T}

Allocating version of cubic spline interpolation using cached LU factorization.

# Arguments
- `deriv::Int=0`: Derivative order (0=value, 1=first derivative, 2=second derivative)
- `search::AbstractSearchPolicy=Binary()`: Search algorithm for interval finding

# Example
```julia
cache = CubicSplineCache(collect(range(0.0, 1.0, 51)))
y = sin.(cache.x)
result = cubic_interp(cache, y, [0.25, 0.5, 0.75])
derivs = cubic_interp(cache, y, [0.25, 0.5, 0.75]; deriv=1)  # First derivative

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = cubic_interp(cache, y, sorted_queries; search=LinearBinary(linear_window=8))
```
"""
function cubic_interp(
    cache::CubicSplineCache{Tg},
    y::AbstractVector{Tv},
    x_query::AbstractVector{Tg};
    extrap::Symbol=:none,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    output = Vector{Tv}(undef, length(x_query))
    cubic_interp!(output, cache, y, x_query; extrap=extrap, deriv=deriv, search=search)
    return output
end

"""
    cubic_interp(x, y, x_query; bc=NaturalBC(), extrap=:none, autocache=true, deriv=0, search=Binary()) -> Vector{T}

Cubic spline interpolation with optional automatic caching.

# Arguments
- `deriv::Int=0`: Derivative order (0=value, 1=first derivative, 2=second derivative)
- `search::AbstractSearchPolicy=Binary()`: Search algorithm for interval finding

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

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = cubic_interp(x, y, sorted_queries; search=LinearBinary(linear_window=8))
```
"""
function cubic_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::AbstractVector{Tg};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    _validate_extrap(extrap)

    output = Vector{Tv}(undef, length(x_query))

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        if _is_periodic_bc(bc)
            return _cubic_interp_periodic!(output, x, y, x_query, autocache, op, searcher)
        end

        bc_pair = _normalize_bc(bc, Tv)
        return _cubic_interp_bcpair!(output, x, y, x_query, bc_pair, extrap, autocache, op, searcher)
    end
end

# Scalar query - zero allocation
cubic_interp(cache::CubicSplineCache{Tg}, y::AbstractVector{Tv},
             x_query::Tg; extrap::Symbol=:none, deriv::Int=0, search=Binary(), hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv} =
    cubic_interp_scalar(cache, y, x_query; extrap=extrap, deriv=deriv, search=search, hint=hint)

function cubic_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::Tg;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0,
    search=Binary(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    _validate_extrap(extrap)

    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        if _is_periodic_bc(bc)
            return _cubic_interp_periodic_scalar(x, y, x_query, autocache, op, searcher)
        end

        bc_pair = _normalize_bc(bc, Tv)
        return _cubic_interp_bcpair_scalar(x, y, x_query, bc_pair, extrap, autocache, op, searcher)
    end
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GENERIC WRAPPERS - CONVENIENCE                        ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Note: CubicInterpolant callable methods and 2-arg form are in cubic_interpolant.jl
# POLICY: Tg is computed from x/y ONLY, not from x_query

# Allocating - vector query
function cubic_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::AbstractVector{Tq};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:Real, Tv, Tq<:Real}
    x_typed, y_typed, xq_typed = _promote_itp_inputs(x, y, x_query)
    return cubic_interp(x_typed, y_typed, xq_typed; bc, extrap, autocache, deriv, search)
end

# Allocating - scalar query
function cubic_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::Tq;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0,
    search=Binary(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:Real, Tv, Tq<:Real}
    x_typed, y_typed = _promote_itp_inputs(x, y)
    Tg_float = eltype(x_typed)
    return cubic_interp(x_typed, y_typed, Tg_float(x_query); bc, extrap, autocache, deriv, search, hint)
end

# In-place - vector query
function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::AbstractVector{Tq};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:Real, Tv, Tq<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_query) "output must match x_query length"

    x_typed, y_typed, xq_typed = _promote_itp_inputs(x, y, x_query)
    Tg_float = eltype(x_typed)
    Tv_float = eltype(y_typed)

    # Validate output can hold result type
    Tout = eltype(output)
    if promote_type(Tout, Tv_float) !== Tout
        throw(ArgumentError(
            "output eltype $Tout cannot hold interpolation result type $Tv_float. " *
            "Use Vector{$Tv_float} or a wider type (e.g., Vector{Complex{$Tg_float}} for complex y-values)."
        ))
    end

    cubic_interp!(output, x_typed, y_typed, xq_typed; bc, extrap, autocache, deriv, search)
end

# In-place - scalar query
function cubic_interp!(
    output::AbstractVector,
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_query::Tq;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:Real, Tv, Tq<:Real}
    @assert length(output) >= 1 "output must have at least 1 element"

    x_typed, y_typed = _promote_itp_inputs(x, y)
    Tg_float = eltype(x_typed)
    output[1] = cubic_interp(x_typed, y_typed, Tg_float(x_query); bc, extrap, autocache, deriv, search)
    return output
end
