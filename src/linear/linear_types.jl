# ========================================
# Linear Interpolant Types
# ========================================
# Type definition for LinearInterpolant.
# Constructor and callable methods are in linear_interpolant.jl.

"""
    LinearInterpolant{Tg,Tv,X,Y,E,P}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `linear_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg`: Grid type — normally `Float32`/`Float64`, but **unconstrained** to admit
        duck-typed grid scalars (e.g. `ForwardDiff.Dual`) that satisfy the grid
        arithmetic/ordering protocol (`-`, `inv`, `*`, `<`).
- `Tv`: Value type (unconstrained)
- `X<:AbstractVector{Tg}`: Grid vector type — `_CachedRange{Tg}` for Range input
        (uniform, O(1) search + cached `h`/`inv_h` scalars), `_CachedVector{Tg,Tinv}`
        for Vector input (non-uniform, O(log n) search + cached `h`/`inv_h` arrays).
- `Y<:AbstractVector{Tv}`: Values vector type
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type

# Fields
- `x::X`: x-coordinates (sorted, wrapped — grid is the source of truth for spacing)
- `y::Y`: y-values
- `extrap::E`: Extrapolation mode (NoExtrap(), ExtendExtrap(), ClampExtrap(), or WrapExtrap())
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
# Create interpolator (minimal allocation)
itp = linear_interp(x, y)  # default extrap=NoExtrap(), search=AutoSearch()

# Create with custom search policy (overrides default AutoSearch)
itp = linear_interp(x, y; search=BinarySearch())

# Complex-valued interpolation
y_complex = exp.(2im .* x)
itp_c = linear_interp(x, y_complex)  # Works natively with Complex

# Use in broadcast (fused, no intermediate arrays)
result = @. coef * itp(rho) * other_terms

# Reuse interpolator multiple times
vals1 = itp.(query_points1)
vals2 = @. compute(itp(query_points2))

# Extrapolation options
itp_ext = linear_interp(x, y; extrap=ExtendExtrap())  # linear extrap
itp_const = linear_interp(x, y; extrap=ClampExtrap())  # clamp to boundary values
itp_wrap = linear_interp(x, y; extrap=WrapExtrap())  # wrap to domain
val = itp_wrap(2.5)  # wraps to domain

# Override search policy at call time
itp(0.5; search=BinarySearch())  # override stored policy
```
"""
struct LinearInterpolant{
        Tg,
        Tv,
        X <: AbstractVector{Tg},
        Y <: AbstractVector{Tv},
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
    } <: AbstractInterpolant1D{Tg, Tv}
    x::X
    y::Y
    extrap::E  # Extrapolation mode (compile-time specialized)
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: promotes x/y, wraps grid for cached spacing, stores.
    # _store_grid_cached: Vector → _CachedVector (cached h/inv_h),
    #                     Range → _CachedRange (cached scalar h/inv_h).
    # Spacing access via `_get_h(itp.x, i)` / `_get_inv_h(itp.x, i)` —
    # grid is the single source of truth; no separate spacing field.
    function LinearInterpolant(
            x::AbstractVector, y::AbstractVector, ev::E, search::P
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        # `_grid_length(x)` returns physical knot count: identity for plain
        # vectors/ranges, `length(x.inner)` for `_ExclusivePeriodicVector`
        # (whose `length` reports the virtually extended n+1 search domain).
        n_grid = _grid_length(x)
        n_grid == length(y) || _throw_length_mismatch(n_grid, length(y))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid_cached(x, Tg)
        yc = _convert_copy(y, Tv)
        return new{Tg, Tv, typeof(xc), typeof(yc), E, P}(xc, yc, ev, search)
    end
end

# ========================================
# Outer Constructor: convenience kwarg wrapper
# ========================================
@inline function LinearInterpolant(
        x::AbstractVector,
        y::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    # Materialize WrapExtrap{Nothing} against grid so kernels never see the
    # unmaterialized singleton when users construct the struct directly.
    return LinearInterpolant(x, y, _resolve_extrap(extrap, x), search)
end
