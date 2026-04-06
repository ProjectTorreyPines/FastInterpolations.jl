# ========================================
# Akima Interpolant Type
# ========================================
# Akima (1970): slopes from weighted average of adjacent secants.
# Outlier-robust — gives less weight to deviant secants.
# Minimum 3 grid points (2 secants needed).

"""
    AkimaInterpolant1D{Tg, Tv, X, Y, DY, S, E, P}

Callable interpolant for Akima interpolation.
Returned by `akima_interp(x, y)` (2-argument form).

Slopes are computed once at construction via the Akima (1970) weighted-average
algorithm. Evaluation uses the same cubic Hermite kernel.

# Properties
- **C\$^1\$ continuous** (continuous first derivative)
- **Local**: each slope depends on 4 adjacent secants (5-point stencil)
- **Outlier-robust**: deviant secants receive less weight

# Usage
```julia
itp = akima_interp(x, y)
itp(0.5)
itp(0.5; deriv=DerivOp(1))
```
"""
struct AkimaInterpolant1D{
        Tg <: AbstractFloat,
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

    # PreCompute inner constructor
    function AkimaInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, PreCompute}(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, dy::AbstractVector{Tv},
            spacing::S, extrap::E, search::P
        ) where {
            Tg <: AbstractFloat, Tv,
            X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, DY <: AbstractVector{Tv},
            S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy,
        }
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) == length(dy) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
        xc, yc, dyc = copy(x), copy(y), copy(dy)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dyc), S, E, P, PreCompute}(
            xc, yc, dyc, spacing, extrap, search
        )
    end

    # OnTheFly inner constructor
    function AkimaInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, OnTheFly}(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, dy::AbstractSlopeMethod,
            spacing::S, extrap::E, search::P
        ) where {
            Tg <: AbstractFloat, Tv,
            X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, DY <: AbstractSlopeMethod,
            S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy,
        }
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("Akima interpolation requires at least 2 points, got $(length(x))"))
        xc, yc = copy(x), copy(y)
        return new{Tg, Tv, typeof(xc), typeof(yc), DY, S, E, P, OnTheFly}(
            xc, yc, dy, spacing, extrap, search
        )
    end
end

# ========================================
# Outer Constructor
# ========================================

# Outer constructor: PreCompute (dy is a vector)
@inline function AkimaInterpolant1D(
        x::X,
        y::Y,
        dy::DY;
        extrap::AbstractExtrap = NoExtrap(),
        search::P = AutoSearch()
    ) where {
        Tg <: AbstractFloat, Tv,
        X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, DY <: AbstractVector{Tv},
        P <: AbstractSearchPolicy,
    }
    E = typeof(extrap)
    spacing = _create_spacing(x)
    S = typeof(spacing)
    return AkimaInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, PreCompute}(
        x, y, dy, spacing, extrap, search
    )
end

# Outer constructor: OnTheFly (dy is a slope method tag)
@inline function AkimaInterpolant1D(
        x::X,
        y::Y,
        sm::DY;
        extrap::AbstractExtrap = NoExtrap(),
        search::P = AutoSearch()
    ) where {
        Tg <: AbstractFloat, Tv,
        X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, DY <: AbstractSlopeMethod,
        P <: AbstractSearchPolicy,
    }
    E = typeof(extrap)
    spacing = _create_spacing(x)
    S = typeof(spacing)
    return AkimaInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, OnTheFly}(
        x, y, sm, spacing, extrap, search
    )
end
