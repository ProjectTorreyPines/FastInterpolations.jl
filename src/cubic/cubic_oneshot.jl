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
    cubic_interp!(output, cache, y, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

In-place cubic spline interpolation using cached LU factorization.

Solves the tridiagonal system ONCE, then evaluates at all query points.
Thread-safe: workspaces allocated from task-local pool.

# Arguments
- `output::AbstractVector{T}`: Pre-allocated output buffer (modified in-place)
- `cache::CubicSplineCache{T}`: Pre-computed cache with LU factorization
- `y::AbstractVector{T}`: Function values at grid points
- `x_query::AbstractVector{T}`: Query points
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `deriv::DerivOp=EvalValue()`: Derivative order (0=value, 1=first derivative, 2=second derivative)
- `search::AbstractSearchPolicy=AutoSearch()`: Search algorithm for interval finding
"""
@inline @with_pool pool function cubic_interp!(
        output::AbstractVector{Tv},
        cache::CubicSplineCache{Tg, X, F, BC},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv, X, F, BC}
    @assert length(y) == length(cache.x) "y length must match cache grid"
    @assert length(output) == length(x_query) "output length must match x_query"

    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    searcher = _resolve_search(cache.x, x_query, search, nothing)
    _cubic_vector_loop!(output, cache, y, z, x_query, extrap, deriv, searcher)

    return output
end

# ========================================
# Core Fallback Functions (Type-Stable Dispatch)
# ========================================
#
# These are the two core implementations that all public APIs
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
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_query::AbstractVector{<:Real},
        bc::BCPair{L, R},
        extrap::AbstractExtrap,
        autocache::Bool,
        op::O,
        searcher::S
    ) where {Tg, L <: PointBC, R <: PointBC, O <: AbstractEvalOp, S <: Searcher}
    @assert length(y) == length(x) "y length must match x"
    @assert length(output) == length(x_query) "output length must match x_query"

    # Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair internally)
    cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, Tg))
    Tz = _output_eltype(eltype(y), eltype(cache.x))
    z = acquire!(pool, Tz, length(y))
    # Solve uses original BC for proper RHS materialization
    _solve_system!(z, cache, y, bc)

    _cubic_vector_loop!(output, cache, y, z, x_query, extrap, op, searcher)

    return output
end

"""
Core implementation for BCPair boundary conditions (scalar query).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.

Type-Free design: handles both concrete (Deriv1{T}) and lazy (PolyFit{D}) types.
AD-compatible: xq is unconstrained to support ForwardDiff.Dual types.
"""
@inline @with_pool pool function _cubic_interp_bcpair_scalar(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,  # Accepts Tg, Real, or Dual for AD (Dual <: Real)
        bc::BCPair{L, R},
        extrap::AbstractExtrap,
        autocache::Bool,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, L <: PointBC, R <: PointBC, O <: AbstractEvalOp, S <: Searcher}
    # Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair internally)
    cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, Tg))
    Tz = _output_eltype(Tv, eltype(cache.x))
    tmp_z = acquire!(pool, Tz, length(y))
    # Solve uses original BC for proper RHS materialization
    _solve_system!(tmp_z, cache, y, bc)

    _check_domain(cache.x, xq, extrap)
    return _eval_with_bc(cache, y, tmp_z, xq, extrap, op, searcher)
end

"""
    _cubic_periodic_solve!(pool, x, y, bc, autocache) -> (cache, y_p, z)

Shared periodic setup: extend exclusive→inclusive grid, then solve tridiagonal system.
Pool buffers (x_p, y_p, z) are acquired from `pool` — caller's @with_pool scope
manages their lifetime. Follows the `_create_spacing_pooled(pool, ...)` pattern.
"""
@inline function _cubic_periodic_solve!(
        pool::AbstractArrayPool,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc::PeriodicBC,
        autocache::Bool
    ) where {Tg, Tv}
    @assert length(x) == length(y) "x and y must have the same length"

    # ── Extend exclusive → inclusive (pool-based, zero-alloc after warmup) ──
    if bc isa PeriodicBC{:exclusive}
        period = _resolve_exclusive_period(x, bc)
        x_end = first(x) + Tg(period)
        last(x) < x_end || throw(
            ArgumentError(
                "period=$period places virtual endpoint at $x_end, " *
                    "not after last grid point x[end]=$(last(x))"
            )
        )

        n = length(x)
        # Grid: Range → direct construction (type-stable, O(1)), Vector → pool
        # Note: _resolve_exclusive_period guarantees period = step(x) * length(x) for Range,
        # so x_end == last(x) + step(x) and direct Range extension is always valid.
        if x isa AbstractRange
            x_p = _to_float_adding_endpoint(x, Tg)
        else
            x_p = acquire!(pool, Tg, n + 1)
            @inbounds copyto!(x_p, 1, x, 1, n)
            @inbounds x_p[n + 1] = x_end
        end
        y_p = acquire!(pool, Tv, n + 1)
        @inbounds copyto!(y_p, 1, y, 1, n)
        @inbounds y_p[n + 1] = y[1]
    else
        x_p, y_p = x, y
    end

    # ── Solve periodic tridiagonal system ──
    _check_periodic_endpoints(bc, y_p)
    cache = _get_cubic_cache(x_p, PeriodicBC(), _effective_autocache(autocache, Tg))
    Tz = _output_eltype(Tv, eltype(cache.x))
    z = acquire!(pool, Tz, length(y_p))
    _solve_system!(z, cache, y_p, cache.bc_config)

    return cache, y_p, z
end

"""
Core implementation for PeriodicBC boundary conditions (vector output).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
Pool-based exclusive extension: zero-alloc after warmup.
"""
@inline @with_pool pool function _cubic_interp_periodic!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_query::AbstractVector{<:Real},
        bc::PeriodicBC,
        autocache::Bool,
        op::O,
        searcher::S
    ) where {Tg, O <: AbstractEvalOp, S <: Searcher}
    @assert length(output) == length(x_query) "output length must match x_query"

    cache, y_p, z = _cubic_periodic_solve!(pool, x, y, bc, autocache)

    _cubic_vector_loop!(output, cache, y_p, z, x_query, WrapExtrap(), op, searcher)
    return output
end

"""
Core implementation for PeriodicBC boundary conditions (scalar query).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
AD-compatible: xq is unconstrained to support ForwardDiff.Dual types.
Pool-based exclusive extension: zero-alloc after warmup.
"""
@inline @with_pool pool function _cubic_interp_periodic_scalar(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,  # Accepts Tg, Real, or Dual for AD (Dual <: Real)
        bc::PeriodicBC,
        autocache::Bool,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, O <: AbstractEvalOp, S <: Searcher}
    cache, y_p, z = _cubic_periodic_solve!(pool, x, y, bc, autocache)

    _check_domain(cache.x, xq, WrapExtrap())
    return _eval_with_bc(cache, y_p, z, xq, WrapExtrap(), op, searcher)
end

"""
    cubic_interp!(output, x, y, x_query; bc=CubicFit(), extrap=NoExtrap(), autocache=true, deriv=EvalValue(), search=AutoSearch())

In-place cubic spline interpolation with optional automatic caching.
"""
@inline function cubic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{<:Real};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv}
    x = _to_float(x, _promote_grid_float(Tg, Tv))
    Tq_float = eltype(x) <: AbstractFloat ? eltype(x) : float(eltype(x_query))
    x_query = _to_float(x_query, Tq_float)
    searcher = _resolve_search(x, x_query, search, nothing)
    # Periodic BC
    if _is_periodic_bc(bc)
        return _cubic_interp_periodic!(output, x, y, x_query, bc, autocache, deriv, searcher)
    end

    # Normalize to BCPair and dispatch to core
    bc_pair = _normalize_bc(bc, first(y))
    return _cubic_interp_bcpair!(output, x, y, x_query, bc_pair, extrap, autocache, deriv, searcher)
end


# Scalar query - zero allocation
@inline function cubic_interp!(
        output::AbstractVector{Tv},
        cache::CubicSplineCache{Tg, X, F, BC},
        y::AbstractVector{Tv},
        x_query::Tg;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv, X, F, BC}
    @assert length(output) >= 1 "output must have at least 1 element"
    output[1] = cubic_interp_scalar(cache, y, x_query; extrap = extrap, deriv = deriv, search = search)
    return output
end

@inline function cubic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::Real;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv}
    @assert length(output) >= 1 "output must have at least 1 element"
    output[1] = cubic_interp(x, y, x_query; bc, extrap, autocache, deriv, search)
    return output
end

# ========================================
# Allocating Vector API
# ========================================

"""
    cubic_interp(cache, y, x_query; extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch()) -> Vector{T}

Allocating version of cubic spline interpolation using cached LU factorization.

# Arguments
- `deriv::DerivOp=EvalValue()`: Derivative order (0=value, 1=first derivative, 2=second derivative)
- `search::AbstractSearchPolicy=AutoSearch()`: Search algorithm for interval finding

# Example
```julia
cache = CubicSplineCache(collect(range(0.0, 1.0, 51)))
y = sin.(cache.x)
result = cubic_interp(cache, y, [0.25, 0.5, 0.75])
derivs = cubic_interp(cache, y, [0.25, 0.5, 0.75]; deriv=DerivOp(1))  # First derivative

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = cubic_interp(cache, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
function cubic_interp(
        cache::CubicSplineCache{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{Tg};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv}
    output = Vector{Tv}(undef, length(x_query))
    cubic_interp!(output, cache, y, x_query; extrap = extrap, deriv = deriv, search = search)
    return output
end

"""
    cubic_interp(x, y, x_query; bc=CubicFit(), extrap=NoExtrap(), autocache=true, deriv=EvalValue(), search=AutoSearch()) -> Vector{T}

Cubic spline interpolation with optional automatic caching.

# Arguments
- `deriv::DerivOp=EvalValue()`: Derivative order (0=value, 1=first derivative, 2=second derivative)
- `search::AbstractSearchPolicy=AutoSearch()`: Search algorithm for interval finding

# Extrapolation Modes
- `NoExtrap()` (default): Throws DomainError if query point is outside domain
- `ClampExtrap()`: Returns boundary values outside domain (0 for derivatives)
- `ExtendExtrap()`: Extends boundary polynomial outside domain
- `WrapExtrap()`: Wraps coordinates to domain (for sawtooth/triangle patterns)
- For `bc=PeriodicBC()`: extrapolation is ignored (coordinates are always wrapped)

# Example
```julia
result = cubic_interp(x, y, x_query)                     # Auto-cached (default)
derivs = cubic_interp(x, y, x_query; deriv=DerivOp(1))    # First derivative
result = cubic_interp(x, y, x_query; extrap=ExtendExtrap())  # Extend beyond domain

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = cubic_interp(x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
function cubic_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractVector{<:Real};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv}
    Tq = eltype(x_query)
    Tr = _output_eltype(Tv, Tg, Tq)
    output = Vector{Tr}(undef, length(x_query))
    cubic_interp!(output, x, y, x_query; bc, extrap, autocache, deriv, search)
    return output
end

# Scalar query - zero allocation
cubic_interp(
    cache::CubicSplineCache{Tg}, y::AbstractVector{Tv},
    x_query::Tg; extrap::AbstractExtrap = NoExtrap(), deriv::DerivOp = EvalValue(), search::AbstractSearchPolicy = AutoSearch(), hint::Union{Nothing, Base.RefValue{Int}} = nothing
) where {Tg, Tv} =
    cubic_interp_scalar(cache, y, x_query; extrap = extrap, deriv = deriv, search = search, hint = hint)

# Primary scalar method - AD-compatible
# xq accepts Real including ForwardDiff.Dual (Dual <: Real)
function cubic_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq;  # Accepts Tg, Real, or Dual for AD (Dual <: Real)
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    x = _to_float(x, _promote_grid_float(Tg, Tv))
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        return _cubic_interp_periodic_scalar(x, y, xq, bc, autocache, deriv, searcher)
    end

    bc_pair = _normalize_bc(bc, first(y))
    return _cubic_interp_bcpair_scalar(x, y, xq, bc_pair, extrap, autocache, deriv, searcher)
end


# Note: Real wrappers (Tg <: Real) removed — duck-typed Tg dispatch handles
# all grid types including ForwardDiff.Dual. See _to_float + _promote_grid_float
# for type promotion inside @with_pool paths.
