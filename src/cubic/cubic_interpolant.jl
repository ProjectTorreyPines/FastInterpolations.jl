# ========================================
# CubicInterpolant: 2-Argument Form API
# ========================================
#
# This file contains:
# - CubicInterpolant callable methods (scalar, vector, in-place)
# - 2-argument form `cubic_interp(x, y; ...)` that returns CubicInterpolant
# - Internal build helpers for type-stable interpolant construction
#
# Dependencies (from cubic_interp.jl):
# - _is_periodic_bc(bc)
# - _get_cubic_cache(x, bc, autocache)
# - _solve_system!(out_z, cache, y, bc_config)
# - _check_periodic_endpoints(y)
# - _cubic_vector_loop!(output, cache, y, z, x_query, extrap, op, searcher)

# ========================================
# Protocol Trait Implementations
# ========================================
# Generic callables inherited from AbstractInterpolant1D (interpolant_protocol.jl).
# Cubic overrides _itp_grid because grid lives in itp.cache.x, not itp.x.

@inline _itp_grid(itp::CubicInterpolant) = itp.cache.x

@inline function _itp_eval_scalar(itp::CubicInterpolant, xq, extrap, op, searcher)
    return _eval_cubic_at_point(itp.cache.x, itp.y, itp.z, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::CubicInterpolant, xq, extrap, op, searcher)
    return _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xq, extrap, op, searcher)
end

# ========================================
# Internal Build Helpers
# ========================================
# These helpers unify the interpolant construction logic,
# using the cache helpers from cubic_interp.jl.

"""
    _build_interpolant_bcpair(x, y, bc_pair, extrap, autocache, search) -> CubicInterpolant

Build a CubicInterpolant for BCPair boundary conditions.
Tg = grid type, Tv = value type (can be Complex)

# Thread-Safety
Pool-allocated `tmp_z` is copied by the CubicInterpolant constructor,
so the pool memory can be safely reused after this function returns.
"""
@inline @with_pool pool function _build_interpolant_bcpair(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc_pair::BCPair{L, R},
        extrap::AbstractExtrap,
        autocache::Bool,
        search::AbstractSearchPolicy = AutoSearch();
        store::StorePolicy = StorePolicy()
    ) where {Tg, Tv, L <: PointBC, R <: PointBC}
    # Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair internally)
    cache = _get_cubic_cache(x, bc_pair, _effective_autocache(autocache, Tg))
    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), Tv)
    tmp_z = acquire!(pool, Tz, length(y))
    # Solve uses original BC for proper RHS materialization
    _solve_system!(tmp_z, cache, y, bc_pair)
    # 3-arg form: promote FillExtrap value type to Tv (no-op for other extraps).
    extrap_p = _resolve_extrap(extrap, cache.x, Tv)
    return CubicInterpolant(cache, y, tmp_z, bc_pair, extrap_p, search; store = store)
end

"""
    _build_interpolant_periodic(x, y, autocache, search) -> CubicInterpolant

Build a CubicInterpolant for PeriodicBC boundary conditions.
Periodic BC always uses WrapExtrap extrapolation.
Tg = grid type, Tv = value type (can be Complex)

# Thread-Safety
Pool-allocated `tmp_z` is copied by the CubicInterpolant constructor,
so the pool memory can be safely reused after this function returns.
"""
@inline @with_pool pool function _build_interpolant_periodic(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc::PeriodicBC,
        autocache::Bool,
        search::AbstractSearchPolicy = AutoSearch();
        store::StorePolicy = StorePolicy()
    ) where {Tg, Tv}
    x, y = _prepare_periodic(x, y, bc)
    _check_periodic_endpoints(bc, y)
    cache = _get_cubic_cache(x, _bc_after_extend(bc), _effective_autocache(autocache, Tg))
    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), eltype(y))
    tmp_z = acquire!(pool, Tz, length(y))
    _solve_system!(tmp_z, cache, y, cache.bc)
    # Normalize stored bc to `:inclusive` (matching cache state) with period
    # materialized for introspection. Prevents re-extension when this
    # interpolant is later passed to `cubic_adjoint(itp.cache.x; bc=itp.bc)`.
    bc_normalized = _with_resolved_period(_bc_after_extend(bc), cache.bc.period)
    return CubicInterpolant(cache, y, tmp_z, bc_normalized, WrapExtrap(), search; store = store)
end

# ========================================
# BC Type Promotion for Complex Support
# ========================================

"""
    _promote_bc(bc::AbstractBC, ::Type{Tv}) -> AbstractBC

Promote BC to target value type Tv for cubic splines.
Handles conversion of Real BC values to Complex when needed.
"""
@inline _promote_bc(bc::BCPair, ::Type{Tv}) where {Tv} = _normalize_bc(bc)
@inline _promote_bc(::ZeroCurvBC, ::Type{Tv}) where {Tv} = ZeroCurvBC()
@inline _promote_bc(::ZeroSlopeBC, ::Type{Tv}) where {Tv} = ZeroSlopeBC()
@inline _promote_bc(bc::PeriodicBC, ::Type{Tv}) where {Tv} = bc
@inline _promote_bc(bc::PointBC, ::Type{Tv}) where {Tv} = _promote_pointbc(bc, Tv)
@inline _promote_bc(bcs::AbstractVector{<:AbstractBC}, ::Type{Tv}) where {Tv} =
    [_promote_bc(bc, Tv) for bc in bcs]

# ========================================
# 2-Argument Form: Return CubicInterpolant
# ========================================

"""
    cubic_interp(x, y; bc=CubicFit(), extrap=NoExtrap(), autocache=true, search=AutoSearch()) -> CubicInterpolant

Create a callable interpolant for broadcast fusion and reuse.

Pre-computes second derivative coefficients z ONCE at construction time,
enabling true zero-allocation scalar evaluations in broadcast operations.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: y-values
- `bc::AbstractBC`: Boundary condition (default: `CubicFit()`)
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `autocache::Bool`: Enable automatic caching (default: `true`)
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

# Example
```julia
itp = cubic_interp(x, y)           # Pre-computes z coefficients
val = itp(0.5)                      # Scalar (zero-allocation)
vals = itp.(query_points)           # Broadcast
result = @. coef * itp(rho) * ne    # Fused broadcast

# Search policy: AutoSearch adapts to query type (scalar→BinarySearch, vector→LinearBinarySearch)
itp = cubic_interp(x, y)
val = itp(0.5)                      # AutoSearch resolves to BinarySearch() for scalar
itp = cubic_interp(x, y; search=LinearBinarySearch())  # explicit override
val = itp(0.5; search=BinarySearch())     # per-call override

# Complex values
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im, 9.0+10.0im]
itp = cubic_interp(x, y)
val = itp(0.5)  # returns ComplexF64
```
"""
# Internal implementation - takes AbstractBC only (type-stable)
# Tg = grid type, Tv = value type (can be Complex)
@inline function _cubic_interp_impl(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc::AbstractBC,
        extrap::AbstractExtrap,
        autocache::Bool,
        search::P = AutoSearch();
        store::StorePolicy = StorePolicy()
    ) where {Tg, Tv, P <: AbstractSearchPolicy}
    if _is_periodic_bc(bc)
        return _build_interpolant_periodic(x, y, bc, autocache, search; store = store)
    else
        bc_pair = _normalize_bc(bc, first(x), first(y))
        return _build_interpolant_bcpair(x, y, bc_pair, extrap, autocache, search; store = store)
    end
end

# Unified entry: handles all grid types including duck-typed (Dual).
function cubic_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        search::P = AutoSearch(),
        store::StorePolicy = StorePolicy()
    ) where {Tg <: Number, Tv, P <: AbstractSearchPolicy}
    _check_grid_orderable(Tg)
    Tg_f = _promote_grid_float(Tg, Tv)
    # Non-Real (unit-carrying) grids: strip→solve→reattach (type-level branch folds).
    Tg_f <: Real ||
        return _cubic_interp_units(x, y, bc, extrap, autocache, search, store)
    xc = _store_grid(x, Tg_f)
    Tv_out = _value_type(Tv, Tg_f)
    bc_promoted = _promote_bc(bc, Tv_out)
    return _cubic_interp_impl(xc, y, bc_promoted, extrap, autocache, search; store = store)
end

# ── Non-Real (unit-carrying) grids: nondimensionalized solve ──
# The Thomas machinery is unit-hostile by STORAGE, not algebra: factorization
# overwrites h-typed arrays with L multipliers (dimensionless) and inv-diagonal
# (1/X), and the ldiv turns the RHS buffer (Y/X) into the solution (Y/X²) in
# place. Rather than triple-aliasing the core solver, solve on a oneunit-
# stripped twin (division by `oneunit` is exact — bit-identical mantissas),
# reattach units to `z`, and keep the ORIGINAL unit axis for eval/search.
function _cubic_interp_units(x, y, bc, extrap, autocache, search, store)
    _is_periodic_bc(bc) && throw(
        ArgumentError(
            "cubic PeriodicBC on a unit-carrying grid is not supported yet — " *
                "strip units (e.g. `ustrip`) or use a Real grid"
        )
    )
    ux = oneunit(eltype(x))
    uy = _carrier_oneunit(eltype(y))
    xs = x ./ ux
    ys = y ./ uy
    tw = _cubic_interp_impl(
        xs, ys, _strip_bc_units(bc, uy, ux), NoExtrap(), autocache, search
    )
    z = tw.z .* (uy / (ux * ux))
    Tgu = eltype(x)
    # Cubic 1D ALWAYS owns its axis (copy-then-wrap) — deliberately NOT the
    # store-aware `_policy_axis` the quadratic units path uses; do not "unify".
    xc = _cache_axis(_convert_copy(x, Tgu), NoBC())
    bc_u = _normalize_bc(bc, first(y))
    # NOTE: `thomas` is the STRIPPED twin's factorization paired with a unit
    # axis — unused by eval/integrate, but do not feed this cache back into
    # `cubic_interp(cache, y2)`-style rebuilds with unit data.
    cache = CubicSplineCache(xc, bc_u, tw.cache.thomas, nothing)
    extrap_p = _resolve_extrap(extrap, xc, eltype(y))
    return CubicInterpolant(cache, y, z, bc_u, extrap_p, search; store = store)
end

# BC payloads carry derivative units (`Y/X`, `Y/X²`, `Y/X³`) — strip to match
# the nondimensionalized twin; structural BCs pass through.
@inline _strip_bc_units(bc::Union{PolyFit, ZeroCurvBC, ZeroSlopeBC}, uy, ux) = bc
@inline _strip_bc_units(bc::Deriv1, uy, ux) = Deriv1(bc.val / (uy / ux))
@inline _strip_bc_units(bc::Deriv2, uy, ux) = Deriv2(bc.val / (uy / (ux * ux)))
@inline _strip_bc_units(bc::Deriv3, uy, ux) = Deriv3(bc.val / (uy / (ux * ux * ux)))
@inline _strip_bc_units(bc::BCPair, uy, ux) =
    BCPair(_strip_bc_units(bc.left, uy, ux), _strip_bc_units(bc.right, uy, ux))
# Catch-all: an unhandled BC type must fail HERE with an actionable message,
# not as a MethodError deep inside the stripped solve.
@noinline _strip_bc_units(bc::AbstractBC, uy, ux) = throw(
    ArgumentError(
        "BC type $(typeof(bc)) is not supported on a unit-carrying grid yet — " *
            "strip units (e.g. `ustrip`) or use a Real grid"
    )
)

"""
    cubic_interp(cache, y; extrap=NoExtrap(), search=AutoSearch()) -> CubicInterpolant

Create a callable interpolant from a pre-built cache.

# Note on BC Values
Uses `cache.bc` for boundary condition values. This is correct when:
- Cache was built via `CubicSplineCache(x; bc=...)` with actual BC values
- BC is ZeroCurvBC/ZeroSlopeBC/PeriodicBC (values are always zero)

**Warning**: Caches from `get_cubic_cache` contain placeholder zeros in `bc_config`.
For non-zero BC values, use the full API: `cubic_interp(x, y; bc=Deriv1(val))`.

Tg = grid type, Tv = value type (can be Complex)

# Thread-Safety
Pool-allocated `tmp_z` is copied by the CubicInterpolant constructor,
so the pool memory can be safely reused after this function returns.
"""
@with_pool pool function cubic_interp(
        cache::CubicSplineCache{Tg},
        y::AbstractVector{Tv};
        extrap::AbstractExtrap = NoExtrap(),
        search::P = AutoSearch(),
        store::StorePolicy = StorePolicy()
    ) where {Tg, Tv, P <: AbstractSearchPolicy}
    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), Tv)
    tmp_z = acquire!(pool, Tz, length(y))
    _solve_system!(tmp_z, cache, y, cache.bc)

    if cache.bc isa PeriodicBC
        _check_periodic_endpoints(y)
        # Store cache.bc verbatim (already :extended/:inclusive normalized).
        return CubicInterpolant(cache, y, tmp_z, cache.bc, WrapExtrap(), search; store = store)
    end

    # cache.bc is BCPair - use it directly.
    # 3-arg form: promote FillExtrap value type to Tv (no-op for other extraps).
    extrap_p = _resolve_extrap(extrap, cache.x, Tv)
    return CubicInterpolant(cache, y, tmp_z, cache.bc, extrap_p, search; store = store)
end


# Note: Real wrapper (TX <: Real) removed — unified entry above handles
# all grid types including ForwardDiff.Dual via _promote_itp_inputs.
