# ========================================
# Cardinal Interpolant Callable Methods
# ========================================
# Callable methods for CardinalInterpolant1D and 2-arg API.

# Protocol traits: inherited from AbstractHermiteInterpolant1D (hermite_interpolant.jl).

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    cardinal_interp(x, y; tension=0.0, extrap=NoExtrap(), search=AutoSearch()) -> CardinalInterpolant1D

Create a callable cardinal spline interpolant.

Default `tension=0` gives **Catmull-Rom** (central finite difference slopes).
Increasing tension reduces overshoot; `tension=1` gives zero slopes at knots (smooth S-curves between knots).

# Example
```julia
itp = cardinal_interp(x, y)                # CatmullRom (tension=0)
itp = cardinal_interp(x, y; tension=0.5)   # tighter curve
itp(0.5)
```
"""
@inline function cardinal_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY};
        tension::Real = 0.0,
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX, TY}
    Tg = _promote_grid_float(TX, TY)
    extrap_p = _promote_extrap(extrap, _value_type(TY, Tg))
    resolved = _resolve_coeffs(coeffs)
    if resolved isa OnTheFly
        return CardinalInterpolant1D(x, y, CardinalSlopes(Tg(tension)), extrap_p, search, Tg(tension))
    else
        return CardinalInterpolant1D(x, y, PreCompute, extrap_p, search, Tg(tension))
    end
end
