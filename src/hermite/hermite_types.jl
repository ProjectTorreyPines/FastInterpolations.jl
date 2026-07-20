# ========================================
# Hermite Interpolant Types
# ========================================
# CubicHermiteInterpolant1D is the callable interpolant for cubic Hermite interpolation
# with user-supplied slopes. No wrapper type — (x, y, dy) are passed directly.

# ========================================
# Cubic Hermite Interpolant (1D)
# ========================================

"""
    CubicHermiteInterpolant1D{Tg, Tv, X, Y, DY, E, P, CS}

Callable interpolant for cubic Hermite interpolation with user-supplied slopes.
Returned by `hermite_interp(x, y, dy)` (interpolant form).

The grid `x` is wrapped via `_cache_axis` (Range → `_CachedRange`,
Vector → `_CachedVector`) so `_get_h(x, idx)` / `_get_inv_h(x, idx)` are
O(1) cached lookups — no separate `spacing` field is stored. Evaluation uses
`_hermite_kernel_1d` (derivative-based Hermite basis functions), NOT
`_cubic_kernel` (moment-based spline formulation).

# Properties
- **C\$^1\$ continuous** (continuous first derivative, discontinuous second derivative at knots)
- **User-supplied slopes**: `dy` is the first derivative at grid points — no global solve, no BC
- **Local**: each cell depends only on its 2 endpoints — change 1 data point, only 2 cells affected

# Usage
```julia
x  = range(0, 2π, 100)
y  = sin.(x)
dy = cos.(x)
itp = hermite_interp(x, y, dy)
itp(0.5)                          # scalar query
itp(xq_vec)                       # vector query (allocating)
itp(out, xq_vec)                  # vector query (in-place)
itp(0.5; deriv=DerivOp(1))       # first derivative
itp(0.5; search=BinarySearch())  # override search
```
"""
struct CubicHermiteInterpolant1D{
        Tg,
        Tv,
        X <: AbstractVector{Tg},
        Y <: AbstractVector{Tv},
        DY,
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
        CS <: AbstractCoeffStrategy,
    } <: AbstractHermiteInterpolant1D{Tg, Tv}
    x::X
    y::Y
    dy::DY
    extrap::E
    search_policy::P

    # PreCompute inner: `_cache_axis` (insurance) + `_convert_copy` ownership.
    # `dy` is always a fresh `copy` (precomputed externally).
    function CubicHermiteInterpolant1D(
            x::AbstractVector, y::AbstractVector, dy::AbstractVector,
            extrap::E, search::P;
            bc::AbstractBC = NoBC(),
            store::StorePolicy = StorePolicy()
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) == length(dy) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
        length(x) >= 2 || throw(ArgumentError("Hermite interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_axis(x, bc, Tg, store)
        yc = _own_or_ref_values(y, Tv, store)
        dyc = copy(dy)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dyc), E, P, PreCompute}(
            xc, yc, dyc, extrap, search
        )
    end
end
