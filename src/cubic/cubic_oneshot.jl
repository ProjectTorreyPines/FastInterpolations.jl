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
        output::AbstractArray,
        cache::CubicSplineCache{Tg, X, F, BC},
        y::AbstractVector{Tv},
        x_query::AbstractArray{Tq};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: Number, Tv, Tq <: Number, X, F, BC}
    @assert length(y) == length(cache.x) "y length must match cache grid"
    _check_query_output_size(output, x_query)

    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), Tv)
    z = acquire!(pool, Tz, length(y))
    _solve_system!(z, cache, y, cache.bc)

    searcher = _resolve_search(cache.x, x_query, search, nothing)
    # BC-aware extrap: PeriodicBC forces WrapExtrap, otherwise passthrough.
    extrap_eff = _resolve_extrap(extrap, cache.bc, cache.x)
    _cubic_vector_loop!(output, cache, y, z, x_query, extrap_eff, deriv, searcher)

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
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_query::AbstractArray,
        bc::BCPair{L, R},
        extrap::AbstractExtrap,
        autocache::Bool,
        op::O,
        searcher::S
    ) where {Tg, L <: PointBC, R <: PointBC, O <: AbstractEvalOp, S <: Searcher}
    @assert length(y) == length(x) "y length must match x"
    _check_query_output_size(output, x_query)

    # Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair internally).
    # Value-matched `Tg_eff` (Int grid + Float32 data → Float32) selects the data-aware
    # cache bank so `cache.x` — and thus the solve — is at the value width.
    Tg_eff = _promote_grid_float(Tg, eltype(y))
    cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, Tg), Tg_eff)
    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), eltype(y))
    z = acquire!(pool, Tz, length(y))
    # Solve uses original BC for proper RHS materialization
    _solve_system!(z, cache, y, bc)

    # BC-aware extrap: PeriodicBC forces WrapExtrap, otherwise passthrough.
    extrap_eff = _resolve_extrap(extrap, cache.bc, cache.x)
    _cubic_vector_loop!(output, cache, y, z, x_query, extrap_eff, op, searcher)

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
    ) where {Tg, Tv, Tq, L <: PointBC, R <: PointBC, O <: AbstractEvalOp, S <: Searcher}
    # Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair internally).
    # Value-matched `Tg_eff` selects the data-aware cache bank (Int grid + Float32
    # data → Float32 cache), so scalar ≡ batch ≡ persistent at the value width.
    Tg_eff = _promote_grid_float(Tg, Tv)
    cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, Tg), Tg_eff)
    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), Tv)
    tmp_z = acquire!(pool, Tz, length(y))
    # Solve uses original BC for proper RHS materialization
    _solve_system!(tmp_z, cache, y, bc)

    # BC-aware extrap: PeriodicBC forces WrapExtrap, otherwise passthrough.
    extrap_eff = _resolve_extrap(extrap, cache.bc, cache.x)
    _check_domain(cache.x, xq, extrap_eff)
    return _eval_cubic_at_point(cache.x, y, tmp_z, xq, extrap_eff, op, searcher)
end

"""
    _cubic_periodic_solve!(pool, x, y, bc, autocache) -> (cache, y_eff, z)

Shared periodic setup: build BC-aware cache and solve the periodic tridiagonal
system on the user's grid as-is.

- `:inclusive` input (`length(x) = n+1`, `y[1] ≈ y[end]`): solver consumes the
  full closed-cycle grid; eval kernel may read `z[n+1]` (mirrored to `z[1]`).
- `:exclusive` input (`length(x) = n`, virtual seam): solver uses the n-cell
  cycle directly; the seam-cell width is held in `cache.bc.h_n`. No
  grid extension or memcpy — `_resolve_search`'s seam dispatch handles
  eval-time wrap (`q ≥ x[n] → idx_R = 1, xR = x[1] + period`).

Pool buffer `z` is acquired from `pool` (caller's `@with_pool` scope manages
lifetime). `y_eff` returned for caller convenience — same object as `y`
(caller may bind a separate variable for clarity in the existing API shape).
"""
@inline function _cubic_periodic_solve!(
        pool::AbstractArrayPool,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc::PeriodicBC,
        autocache::Bool
    ) where {Tg, Tv}
    @assert length(x) == length(y) "x and y must have the same length"

    # Build cache on the user's grid (BC-aware: `:inclusive` user length n+1 OR
    # `:exclusive` user length n → wrapped axis virtual n+1). Zero-copy.
    # Value-matched `Tg_eff` selects the data-aware cache bank (Int grid + Float32
    # data → Float32 cache).
    Tg_eff = _promote_grid_float(Tg, Tv)
    cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, Tg), Tg_eff)
    # `_resolve_data` handles the per-bc endpoint validation (`:inclusive`
    # checks `y[1] ≈ y[end]`; `:exclusive` is a no-op wrap to length n+1).
    y_eff = _resolve_data(y, bc)
    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), Tv)
    z = acquire!(pool, Tz, length(cache.x))
    _solve_system!(z, cache, y_eff, cache.bc)

    return cache, y_eff, z
end

"""
Core implementation for PeriodicBC boundary conditions (vector output).
Thread-safe: uses _get_cubic_cache + @with_pool pattern.
Pool-based exclusive extension: zero-alloc after warmup.
"""
@inline @with_pool pool function _cubic_interp_periodic!(
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_query::AbstractArray,
        bc::PeriodicBC,
        autocache::Bool,
        op::O,
        searcher::S
    ) where {Tg, O <: AbstractEvalOp, S <: Searcher}
    _check_query_output_size(output, x_query)

    cache, y_p, z = _cubic_periodic_solve!(pool, x, y, bc, autocache)

    # Periodic `_cubic_vector_loop!` dispatch ignores the extrap arg — pass the
    # zero-size singleton to keep the path allocation-free.
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
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    cache, y_p, z = _cubic_periodic_solve!(pool, x, y, bc, autocache)

    # Hoist the domain check so the in-domain query takes the `InBounds`
    # eval path directly (skips the per-call `_wrap_to_domain` dispatch
    # inside `_eval_cubic_at_point(::WrapExtrap)`). OOB queries fall to
    # the wrap path. Mirrors the batch loop's function-barrier pattern.
    xq_p = _extract_primal(xq)
    return if _is_inbounds(cache.x, xq_p)
        _eval_cubic_at_point(cache.x, y_p, z, xq, InBounds(), op, searcher)
    else
        _eval_cubic_at_point(cache.x, y_p, z, xq, WrapExtrap(), op, searcher)
    end
end

"""
    cubic_interp!(output, x, y, x_query; bc=CubicFit(), extrap=NoExtrap(), autocache=true, deriv=EvalValue(), search=AutoSearch())

In-place cubic spline interpolation with optional automatic caching.
"""
@inline function cubic_interp!(
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_query::AbstractArray{Tq};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Number, Tv, Tq <: Number}
    # Value-matched Tg: Int/OneTo grid + Float32 data → Float32 axis, so the spline
    # cache builds (and memoises — `_CachedRange` is isbits, objectid-deterministic)
    # at the value width instead of the blind Float64.
    x = _resolve_axis(x, _promote_grid_float(Tg, Tv))
    # No BC on Searcher: seam handled by axis-level dispatch on `cache.x` at eval.
    searcher = _resolve_search(x, x_query, search, hint)
    # Periodic BC
    if _is_periodic_bc(bc)
        return _cubic_interp_periodic!(output, x, y, x_query, bc, autocache, deriv, searcher)
    end

    bc_pair = _normalize_bc_solve(bc, x, y)
    return _cubic_interp_bcpair!(output, x, y, x_query, bc_pair, extrap, autocache, deriv, searcher)
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
        x_query::AbstractArray{Tq};
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: Number, Tv, Tq <: Number}
    Tr = _deriv_eltype(
        _promote_eltype(_interp_op, eltype(cache.x), Tv, Tq), eltype(cache.x), deriv
    )
    output = _alloc_query_output(Tr, x_query)
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
        x_query::AbstractArray{Tq};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Number, Tv, Tq <: Number}
    # Output space is deriv-aware: value space folded by grid⁻ⁿ (`_deriv_eltype`
    # — Real grids: identity). The reroute era hid this behind the persistent
    # build's own allocation.
    Tr = _deriv_eltype(
        _promote_eltype(_interp_op, Tg, Tv, Tq), _promote_grid_float(Tg, Tv), deriv
    )
    output = _alloc_query_output(Tr, x_query)
    cubic_interp!(output, x, y, x_query; bc, extrap, autocache, deriv, search, hint)
    return output
end

# Scalar query - zero allocation
cubic_interp(
    cache::CubicSplineCache{Tg}, y::AbstractVector{Tv},
    x_query::Tq; extrap::AbstractExtrap = NoExtrap(), deriv::DerivOp = EvalValue(), search::AbstractSearchPolicy = AutoSearch(), hint::Union{Nothing, Base.RefValue{Int}} = nothing
) where {Tg <: Number, Tv, Tq <: Number} =
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
    ) where {Tg <: Number, Tv, Tq <: Number}
    # Value-matched Tg (see the in-place form above): Ranges resolve to the value
    # width; raw Vectors pass through (identity-keyed cache — legacy width there).
    x = _resolve_axis(x, _promote_grid_float(Tg, Tv))
    # No BC on Searcher: seam handled by axis-level dispatch on `cache.x` at eval.
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        return _cubic_interp_periodic_scalar(x, y, xq, bc, autocache, deriv, searcher)
    end

    bc_pair = _normalize_bc_solve(bc, x, y)
    return _cubic_interp_bcpair_scalar(x, y, xq, bc_pair, extrap, autocache, deriv, searcher)
end


# Note: Real wrappers (Tg <: Real) removed — duck-typed Tg dispatch handles
# all grid types including ForwardDiff.Dual. See _to_float + _promote_grid_float
# for type promotion inside @with_pool paths.
