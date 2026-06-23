# ========================================
# PCHIP Interpolant Type
# ========================================
# PchipInterpolant1D is structurally identical to CubicHermiteInterpolant1D
# but exists as a separate type for dispatch (show, plot recipe, future
# integrate/adjoint). Slopes are computed via Fritsch-Carlson algorithm
# at construction, then stored in `dy` for O(1) evaluation.

"""
    PchipInterpolant1D{Tg, Tv, X, Y, DY, E, P, CS}

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
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
        CS <: AbstractCoeffStrategy,
    } <: AbstractHermiteInterpolant1D{Tg, Tv}
    x::X
    y::Y
    dy::DY
    extrap::E
    search_policy::P

    # PreCompute inner: builds slopes via Fritsch-Carlson.
    function PchipInterpolant1D(
            x::AbstractVector, y::AbstractVector, ::Type{PreCompute}, extrap::E, search::P;
            bc::AbstractBC = NoBC()
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("PCHIP interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _convert_copy(_cache_axis(x, bc, Tg), Tg)
        yc = _convert_copy(y, Tv)
        Tdy = _promote_eltype(_coeff_op, Tg, Tv)
        dy = Vector{Tdy}(undef, length(yc))
        _pchip_slopes!(dy, xc, yc)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dy), E, P, PreCompute}(
            xc, yc, dy, extrap, search
        )
    end

    # Pre-computed slopes inner: caller-supplied `dy` (used by periodic path).
    function PchipInterpolant1D(
            x::AbstractVector, y::AbstractVector, dy::AbstractVector, extrap::E, search::P;
            bc::AbstractBC = NoBC()
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(dy) == length(y) || throw(ArgumentError("dy length ($(length(dy))) must match y length ($(length(y)))"))
        length(x) >= 2 || throw(ArgumentError("PCHIP interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _convert_copy(_cache_axis(x, bc, Tg), Tg)
        yc = _convert_copy(y, Tv)
        Tdy = _promote_eltype(_coeff_op, Tg, Tv)
        dyc = _convert_copy(dy, Tdy)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dyc), E, P, PreCompute}(
            xc, yc, dyc, extrap, search
        )
    end

    # OnTheFly inner: stores slope_strategy tag.
    function PchipInterpolant1D(
            x::AbstractVector, y::AbstractVector, slope_strategy::AbstractSlopeMethod, extrap::E, search::P;
            bc::AbstractBC = NoBC()
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || throw(ArgumentError("PCHIP interpolation requires at least 2 points, got $(length(x))"))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _convert_copy(_cache_axis(x, bc, Tg), Tg)
        yc = _convert_copy(y, Tv)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(slope_strategy), E, P, OnTheFly}(
            xc, yc, slope_strategy, extrap, search
        )
    end
end

# Outer kwarg wrapper. Wraps the axis here so the inner ctor's `_cache_axis`
# insurance is an idempotent passthrough.
@inline function PchipInterpolant1D(
        x::AbstractVector,
        y::AbstractVector,
        slope_strategy::AbstractSlopeMethod;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    Tg = _promote_grid_float(eltype(x), eltype(y))
    x_eff = _cache_axis(x, bc, Tg)
    return PchipInterpolant1D(x_eff, y, slope_strategy, extrap, search; bc = bc)
end
