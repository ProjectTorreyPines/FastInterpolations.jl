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

    # PreCompute inner: dy is a precomputed slope vector
    function CardinalInterpolant1D(
            x::AbstractVector{Tg}, y::AbstractVector, dy::AbstractVector,
            spacing::S, extrap::E, search::P, tension::Tg
        ) where {Tg, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) == length(dy) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
        Tv = _value_type(eltype(y), Tg)
        xc = copy(x)
        yc = _convert_copy(y, Tv)
        dyc = copy(dy)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dyc), S, E, P, PreCompute}(
            xc, yc, dyc, spacing, extrap, search, tension
        )
    end

    # OnTheFly inner: dy is a slope method tag
    function CardinalInterpolant1D(
            x::AbstractVector{Tg}, y::AbstractVector, dy::AbstractSlopeMethod,
            spacing::S, extrap::E, search::P, tension::Tg
        ) where {Tg, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
        Tv = _value_type(eltype(y), Tg)
        xc = copy(x)
        yc = _convert_copy(y, Tv)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dy), S, E, P, OnTheFly}(
            xc, yc, dy, spacing, extrap, search, tension
        )
    end
end

# ========================================
# Outer Constructor
# ========================================

# Outer constructor: computes spacing, dispatches to inner via dy type.
@inline function CardinalInterpolant1D(
        x::AbstractVector{Tg},
        y::AbstractVector,
        dy;  # AbstractVector (PreCompute) or AbstractSlopeMethod (OnTheFly)
        tension::Tg = zero(Tg),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg}
    spacing = _create_spacing(x)
    return CardinalInterpolant1D(x, y, dy, spacing, extrap, search, tension)
end
