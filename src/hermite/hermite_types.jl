# ========================================
# Hermite Interpolant Types
# ========================================
# CubicHermiteInterpolant1D is the callable interpolant for cubic Hermite interpolation
# with user-supplied slopes. No wrapper type — (x, y, dy) are passed directly.

# ========================================
# Cubic Hermite Interpolant (1D)
# ========================================

"""
    CubicHermiteInterpolant1D{Tg, Tv, X, Y, DY, S, E, P}

Callable interpolant for cubic Hermite interpolation with user-supplied slopes.
Returned by `hermite_interp(x, y, dy)` (interpolant form).

Stores precomputed grid spacing for O(1) `h`/`inv_h` lookup on uniform grids.
Evaluation uses `_hermite_kernel_1d` (derivative-based Hermite basis functions),
NOT `_cubic_kernel` (moment-based spline formulation).

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
        Tg <: AbstractFloat,
        Tv,
        X <: AbstractVector{Tg},
        Y <: AbstractVector{Tv},
        DY <: AbstractVector{Tv},
        S <: AbstractGridSpacing{Tg},
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
    } <: AbstractHermiteInterpolant1D{Tg, Tv}
    x::X
    y::Y
    dy::DY
    spacing::S
    extrap::E
    search_policy::P

    function CubicHermiteInterpolant1D{Tg, Tv, X, Y, DY, S, E, P}(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, dy::AbstractVector{Tv},
            spacing::S, extrap::E, search::P
        ) where {
            Tg <: AbstractFloat, Tv,
            X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, DY <: AbstractVector{Tv},
            S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy,
        }
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) == length(dy) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
        length(x) >= 2 || throw(ArgumentError("Hermite interpolation requires at least 2 points, got $(length(x))"))
        xc, yc, dyc = copy(x), copy(y), copy(dy)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dyc), S, E, P}(
            xc, yc, dyc, spacing, extrap, search
        )
    end
end
