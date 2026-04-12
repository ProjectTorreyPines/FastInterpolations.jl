# ========================================
# Cardinal Spline Interpolant Type
# ========================================
# Cardinal spline: slopes = (1 - tension) * central finite difference.
# CatmullRom is the special case with tension=0.
# Structurally identical to CubicHermiteInterpolant1D / PchipInterpolant1D
# but separate type for dispatch (show, plot, future integrate/adjoint).

"""
    CardinalInterpolant1D{Tg, Tv, X, Y, DY, S, E, P}

Callable interpolant for cardinal spline interpolation.
Returned by `cardinal_interp(x, y)` (2-argument form).

Slopes are computed once at construction via:
    `d_k = (1 - tension) * (y[k+1] - y[k-1]) / (x[k+1] - x[k-1])`

When `tension=0`, this is the **Catmull-Rom** spline (simple central FD).

# Properties
- **C\$^1\$ continuous** (continuous first derivative)
- **Local**: each slope depends only on immediate neighbors
- **Tension parameter**: controls overshoot (0 = CatmullRom, 1 = zero slopes at knots)

# Usage
```julia
itp = cardinal_interp(x, y)                     # default: CatmullRom (tension=0)
itp = cardinal_interp(x, y; tension=0.5)        # tighter curve
itp(0.5)
itp(0.5; deriv=DerivOp(1))
```
"""
struct CardinalInterpolant1D{
        Tg,
        Tv,
        X <: AbstractVector{Tg},
        Y <: AbstractVector{Tv},
        DY,
        S <: AbstractGridSpacing{Tg},
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
        CS <: AbstractCoeffStrategy,
    } <: AbstractHermiteInterpolant1D{Tg, Tv}
    x::X
    y::Y
    dy::DY
    spacing::S
    extrap::E
    search_policy::P
    tension::Tg

    # PreCompute inner: promotes x/y, creates spacing, stores everything.
    function CardinalInterpolant1D(
            x::AbstractVector, y::AbstractVector, dy::AbstractVector,
            extrap::E, search::P, tension::Real
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) == length(dy) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid(x, Tg)
        yc = _convert_copy(y, Tv)
        spacing = _create_spacing(xc)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dy), typeof(spacing), E, P, PreCompute}(
            xc, yc, dy, spacing, extrap, search, Tg(tension)
        )
    end

    # OnTheFly inner: promotes x/y, creates spacing, dy is a slope method tag.
    function CardinalInterpolant1D(
            x::AbstractVector, y::AbstractVector, dy::AbstractSlopeMethod,
            extrap::E, search::P, tension::Real
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid(x, Tg)
        yc = _convert_copy(y, Tv)
        spacing = _create_spacing(xc)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dy), typeof(spacing), E, P, OnTheFly}(
            xc, yc, dy, spacing, extrap, search, Tg(tension)
        )
    end
end

# ========================================
# Outer Constructor: kwarg wrapper
# ========================================
@inline function CardinalInterpolant1D(
        x::AbstractVector,
        y::AbstractVector,
        dy;  # AbstractVector (PreCompute) or AbstractSlopeMethod (OnTheFly)
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    return CardinalInterpolant1D(x, y, dy, extrap, search, tension)
end
