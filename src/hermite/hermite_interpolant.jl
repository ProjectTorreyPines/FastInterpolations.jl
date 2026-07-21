# ========================================
# Hermite Interpolant Callable Methods
# ========================================
# Callable methods for CubicHermiteInterpolant1D and interpolant-form API.
# Type definition is in hermite_types.jl.
# Oneshot API (hermite_interp 4-arg) is in hermite_oneshot.jl.

# ========================================
# Protocol Trait Implementations (shared for all Hermite family)
# ========================================
# Dispatches on AbstractHermiteInterpolant1D: covers Hermite, PCHIP, Cardinal, Akima.
# All subtypes store (x, y, dy) — axis-as-truth: `x` is the wrapped axis and
# `_get_h(x, idx)` covers the cached-h lookup that the legacy `spacing` field
# served. _itp_grid, _itp_extrap, _itp_search use defaults
# (itp.x, itp.extrap, itp.search_policy).

@inline function _itp_eval_scalar(itp::AbstractHermiteInterpolant1D, xq, extrap, op, searcher)
    return _hermite_eval_at_point(itp.x, itp.y, itp.dy, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::AbstractHermiteInterpolant1D, xq, extrap, op, searcher)
    return _hermite_vector_loop!(output, itp.x, itp.y, itp.dy, xq, extrap, op, searcher)
end

# Hermite-family kernel mixes `y` and `dy` (`h00·y0 + h01·y1 + h10·h·dy0 + h11·h·dy1`).
# Two-call pattern over `_interp_op` keeps `Tv` and `eltype(dy)` in
# disjoint promote chains so a duck-typed `dy` (e.g., Float64 y + Vector{Dual} dy
# for AD on slopes) widens the result without poisoning the `y` chain.
@inline function _promote_eltype(itp::AbstractHermiteInterpolant1D{Tg, Tv}, ::Type{Tq}) where {Tg, Tv, Tq}
    Tdy = eltype(itp.dy)
    return promote_type(
        _promote_eltype(_interp_op, Tg, Tv, Tq),
        _promote_eltype(_interp_op, Tg, Tdy, Tq),
    )
end

# ========================================
# Outer Constructor: typed inputs only
# ========================================

# Outer kwarg wrapper. Wraps the axis here so the inner ctor's `_cache_axis`
# insurance is an idempotent passthrough.
@inline function CubicHermiteInterpolant1D(
        x::AbstractVector,
        y::AbstractVector,
        dy::AbstractVector;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
        store::StorePolicy = StorePolicy()
    )
    Tg = _promote_grid_float(eltype(x), eltype(y))
    x_eff = _policy_axis(x, bc, Tg, store)
    return CubicHermiteInterpolant1D(x_eff, y, dy, extrap, search; bc = bc, store = store)
end

# ========================================
# Interpolant Form: Return Callable
# ========================================

"""
    hermite_interp(x, y, dy; extrap=NoExtrap(), search=AutoSearch()) -> CubicHermiteInterpolant1D

Create a callable cubic Hermite interpolant from user-supplied slopes.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: Function values at grid points
- `dy::AbstractVector`: First derivatives at grid points
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`)
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

# Returns
`CubicHermiteInterpolant1D` callable object. Supports:
- Scalar: `itp(0.5)`
- Vector: `itp(xq_vec)`
- In-place: `itp(out, xq_vec)`
- Broadcast: `itp.(xq)` or `@. coef * itp(rho)`
- Derivative: `itp(0.5; deriv=DerivOp(1))`

# Example
```julia
x  = range(0, 2π, 100)
y  = sin.(x)
dy = cos.(x)
itp = hermite_interp(x, y, dy)
itp(1.0)                          # ≈ sin(1.0)
itp(1.0; deriv=DerivOp(1))       # ≈ cos(1.0)
```
"""
@inline function hermite_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        dy::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
        store::StorePolicy = StorePolicy(),
    ) where {TX, TY}
    _check_grid_orderable(TX)
    x_p, y_p, dy_p = _promote_hermite_inputs(x, y, dy)
    extrap_p = _resolve_extrap(extrap, x_p, eltype(y_p))
    # Caching wrap (zero-copy of buffer); ownership copy in inner ctor.
    x_p = _cache_axis(x_p, NoBC())
    return CubicHermiteInterpolant1D(x_p, y_p, dy_p; extrap = extrap_p, search, store = store)
end
