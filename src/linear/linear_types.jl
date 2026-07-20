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

    # Inner: `_cache_axis` (insurance — passthrough on wrapped, wraps raw)
    # then `_convert_copy` for ownership + eltype promotion. `bc` kwarg lets
    # direct-ctor callers request periodic without the factory.
    function LinearInterpolant(
            x::AbstractVector, y::AbstractVector, ev::E, search::P;
            bc::AbstractBC = NoBC(),
            store::StorePolicy = StorePolicy()
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        _check_compatible_length(x, y)
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_axis(x, bc, Tg, store)
        yc = _own_or_ref_values(y, Tv, store)
        return new{Tg, Tv, typeof(xc), typeof(yc), E, P}(xc, yc, ev, search)
    end
end

# Outer kwarg wrapper. Wraps the axis here so the inner ctor's `_cache_axis`
# insurance is an idempotent passthrough.
@inline function LinearInterpolant(
        x::AbstractVector,
        y::AbstractVector;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
        store::StorePolicy = StorePolicy()
    )
    Tg = _promote_grid_float(eltype(x), eltype(y))
    x_eff = _policy_axis(x, bc, Tg, store)
    return LinearInterpolant(x_eff, y, _resolve_extrap(extrap, x_eff), search; bc = bc, store = store)
end
