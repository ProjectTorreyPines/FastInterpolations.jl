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

    # PreCompute inner: promotes x/y, computes slopes — axis-as-truth (`xc`
    # wraps cached h/inv_h via `_store_grid_cached`).
    function AkimaInterpolant1D(
            x::AbstractVector, y::AbstractVector, ::Type{PreCompute}, extrap::E, search::P
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("Akima interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid_cached(x, Tg)
        yc = _convert_copy(y, Tv)
        Tdy = _output_eltype(Tv, Tg)
        dy = Vector{Tdy}(undef, length(yc))
        _akima_slopes!(dy, xc, yc)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dy), E, P, PreCompute}(
            xc, yc, dy, extrap, search
        )
    end

    # Pre-computed slopes inner: caller-supplied `dy`. Used by periodic path.
    function AkimaInterpolant1D(
            x::AbstractVector, y::AbstractVector, dy::AbstractVector, extrap::E, search::P
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(dy) == length(y) || throw(ArgumentError("dy length ($(length(dy))) must match y length ($(length(y)))"))
        length(x) >= 2 || throw(ArgumentError("Akima interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid_cached(x, Tg)
        yc = _convert_copy(y, Tv)
        Tdy = _output_eltype(Tv, Tg)
        dyc = _convert_copy(dy, Tdy)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dyc), E, P, PreCompute}(
            xc, yc, dyc, extrap, search
        )
    end

    # OnTheFly inner: slope_strategy is a slope method tag.
    function AkimaInterpolant1D(
            x::AbstractVector, y::AbstractVector, slope_strategy::AbstractSlopeMethod, extrap::E, search::P
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("Akima interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid_cached(x, Tg)
        yc = _convert_copy(y, Tv)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(slope_strategy), E, P, OnTheFly}(
            xc, yc, slope_strategy, extrap, search
        )
    end
end

# Outer: kwarg wrapper.
@inline function AkimaInterpolant1D(
        x::AbstractVector,
        y::AbstractVector,
        slope_strategy::AbstractSlopeMethod;
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    return AkimaInterpolant1D(x, y, slope_strategy, extrap, search)
end
