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
        coeffs::AbstractCoeffStrategy = PreCompute(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX <: Real, TY}
    x_p, y_p = _promote_itp_inputs(x, y)
    extrap_p = _promote_extrap(extrap, eltype(y_p))
    if coeffs isa OnTheFly
        return AkimaInterpolant1D(x_p, y_p, AkimaSlopes(); extrap = extrap_p, search)
    else
        dy_p = similar(y_p)
        _akima_slopes!(dy_p, x_p, y_p)
        return AkimaInterpolant1D(x_p, y_p, dy_p; extrap = extrap_p, search)
    end
end
