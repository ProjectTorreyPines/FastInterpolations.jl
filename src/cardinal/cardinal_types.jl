# ========================================
# Cardinal Spline Interpolant Type
# ========================================
# Cardinal spline: slopes = (1 - tension) * central finite difference.
# CatmullRom is the special case with tension=0.
# Structurally identical to CubicHermiteInterpolant1D / PchipInterpolant1D
# but separate type for dispatch (show, plot, future integrate/adjoint).

"""
    CardinalInterpolant1D{Tg, Tv, X, Y, DY, E, P, CS}

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
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
        CS <: AbstractCoeffStrategy,
    } <: AbstractHermiteInterpolant1D{Tg, Tv}
    x::X
    y::Y
    dy::DY
    extrap::E
    search_policy::P
    tension::Tg

    # PreCompute inner: promotes x/y, computes slopes — axis-as-truth (`xc`
    # wraps cached h/inv_h via `_resolve_axis_copied`).
    function CardinalInterpolant1D(
            x::AbstractVector, y::AbstractVector, ::Type{PreCompute},
            extrap::E, search::P, tension::Real
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _resolve_axis_copied(x, NoBC(), Tg)
        yc = _convert_copy(y, Tv)
        Tdy = _output_eltype(Tv, Tg)
        dy = Vector{Tdy}(undef, length(yc))
        _cardinal_slopes!(dy, xc, yc, Tg(tension))
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dy), E, P, PreCompute}(
            xc, yc, dy, extrap, search, Tg(tension)
        )
    end

    # Pre-computed slopes inner: caller-supplied `dy`. Used by periodic path.
    function CardinalInterpolant1D(
            x::AbstractVector, y::AbstractVector, dy::AbstractVector,
            extrap::E, search::P, tension::Real
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(dy) == length(y) || throw(ArgumentError("dy length ($(length(dy))) must match y length ($(length(y)))"))
        length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _resolve_axis_copied(x, NoBC(), Tg)
        yc = _convert_copy(y, Tv)
        Tdy = _output_eltype(Tv, Tg)
        dyc = _convert_copy(dy, Tdy)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dyc), E, P, PreCompute}(
            xc, yc, dyc, extrap, search, Tg(tension)
        )
    end

    # OnTheFly inner: slope_strategy is a slope method tag.
    function CardinalInterpolant1D(
            x::AbstractVector, y::AbstractVector, slope_strategy::AbstractSlopeMethod,
            extrap::E, search::P, tension::Real
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("Cardinal interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _resolve_axis_copied(x, NoBC(), Tg)
        yc = _convert_copy(y, Tv)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(slope_strategy), E, P, OnTheFly}(
            xc, yc, slope_strategy, extrap, search, Tg(tension)
        )
    end
end

# ========================================
# Outer Constructor: kwarg wrapper
# ========================================
@inline function CardinalInterpolant1D(
        x::AbstractVector,
        y::AbstractVector,
        slope_strategy::AbstractSlopeMethod;
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    return CardinalInterpolant1D(x, y, slope_strategy, extrap, search, tension)
end
