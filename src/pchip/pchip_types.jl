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

    # PreCompute inner: promotes x/y, computes slopes, creates spacing — all in one place.
    function PchipInterpolant1D(
            x::AbstractVector, y::AbstractVector, ::Type{PreCompute}, extrap::E, search::P
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("PCHIP interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid(x, Tg)
        yc = _convert_copy(y, Tv)
        spacing = _create_spacing(xc)
        Tdy = _output_eltype(Tv, Tg)
        dy = Vector{Tdy}(undef, length(yc))
        _pchip_slopes!(dy, xc, yc)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dy), typeof(spacing), E, P, PreCompute}(
            xc, yc, dy, spacing, extrap, search
        )
    end

    # OnTheFly inner: promotes x/y, creates spacing, slope_strategy is a slope method tag.
    function PchipInterpolant1D(
            x::AbstractVector, y::AbstractVector, slope_strategy::AbstractSlopeMethod, extrap::E, search::P
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("PCHIP interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid(x, Tg)
        yc = _convert_copy(y, Tv)
        spacing = _create_spacing(xc)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(slope_strategy), typeof(spacing), E, P, OnTheFly}(
            xc, yc, slope_strategy, spacing, extrap, search
        )
    end
end

# ========================================
# Outer Constructor: kwarg wrapper
# ========================================
@inline function PchipInterpolant1D(
        x::AbstractVector,
        y::AbstractVector,
        slope_strategy::AbstractSlopeMethod;
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    return PchipInterpolant1D(x, y, slope_strategy, extrap, search)
end
