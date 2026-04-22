# ========================================
# Akima Interpolant Callable Methods
# ========================================

# Protocol traits: inherited from AbstractHermiteInterpolant1D (hermite_interpolant.jl).

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    akima_interp(x, y; extrap=NoExtrap(), search=AutoSearch()) -> AkimaInterpolant1D

Create a callable Akima interpolant with outlier-robust slopes.

Slopes are computed once at construction. Requires ≥ 2 grid points.

# Example
```julia
itp = akima_interp(x, y)
itp(0.5)
itp(0.5; deriv=DerivOp(1))
```
"""
@inline function akima_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX, TY}
    Tg = _promote_grid_float(TX, TY)
    extrap_mat = _resolve_extrap(extrap, x)
    extrap_p = _promote_extrap(extrap_mat, _value_type(TY, Tg))
    resolved = _resolve_coeffs(coeffs)
    if resolved isa OnTheFly
        return AkimaInterpolant1D(x, y, AkimaSlopes(), extrap_p, search)
    else
        return AkimaInterpolant1D(x, y, PreCompute, extrap_p, search)
    end
end
