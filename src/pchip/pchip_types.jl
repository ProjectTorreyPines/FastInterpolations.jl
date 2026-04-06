# ========================================
# PCHIP Interpolant Type
# ========================================
# PchipInterpolant1D is structurally identical to CubicHermiteInterpolant1D
# but exists as a separate type for dispatch (show, plot recipe, future
# integrate/adjoint). Slopes are computed via Fritsch-Carlson algorithm
# at construction, then stored in `dy` for O(1) evaluation.

"""
    PchipInterpolant1D{Tg, Tv, X, Y, DY, S, E, P}

Callable interpolant for PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
Returned by `pchip_interp(x, y)` (2-argument form).

Slopes are computed once at construction via the Fritsch-Carlson monotone-preserving
algorithm. Evaluation uses the same cubic Hermite kernel as `CubicHermiteInterpolant1D`.

# Properties
- **C\$^1\$ continuous** (continuous first derivative, discontinuous second derivative at knots)
- **Monotonicity preserving**: monotone input data → monotone interpolant
- **Local**: each slope depends only on neighboring data — no global solve

# Usage
```julia
itp = pchip_interp(x, y)
itp(0.5)                          # scalar query
itp(xq_vec)                       # vector query (allocating)
itp(out, xq_vec)                  # vector query (in-place)
itp(0.5; deriv=DerivOp(1))       # first derivative
```
"""
struct PchipInterpolant1D{
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

    # PreCompute inner constructor: dy is a precomputed slope vector
    function PchipInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, PreCompute}(
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

    # OnTheFly inner constructor: dy is a slope method tag (no slope data)
    function PchipInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, OnTheFly}(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, dy::AbstractSlopeMethod,
            spacing::S, extrap::E, search::P
        ) where {
            Tg <: AbstractFloat, Tv,
            X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, DY <: AbstractSlopeMethod,
            S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy,
        }
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("PCHIP interpolation requires at least 2 points, got $(length(x))"))
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
@inline function PchipInterpolant1D(
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
    return PchipInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, PreCompute}(
        x, y, dy, spacing, extrap, search
    )
end

# Outer constructor: OnTheFly (dy is a slope method tag)
@inline function PchipInterpolant1D(
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
    return PchipInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, OnTheFly}(
        x, y, sm, spacing, extrap, search
    )
end
