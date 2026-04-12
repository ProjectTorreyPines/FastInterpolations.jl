# ========================================
# PCHIP Interpolant Callable Methods
# ========================================
# Callable methods for PchipInterpolant1D and 2-arg API.
# Type definition is in pchip_types.jl.
# Oneshot API (pchip_interp 3-arg) is in pchip_oneshot.jl.

# Protocol traits: inherited from AbstractHermiteInterpolant1D (hermite_interpolant.jl).

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    pchip_interp(x, y; extrap=NoExtrap(), search=AutoSearch()) -> PchipInterpolant1D

Create a callable PCHIP interpolant with monotone-preserving slopes.

Slopes are computed once at construction via the Fritsch-Carlson algorithm.
Subsequent `itp(xq)` calls just evaluate — zero slope recomputation.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted, ≥2 points)
- `y::AbstractVector`: y-values
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`)
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

# Returns
`PchipInterpolant1D` callable object. Supports:
- Scalar: `itp(0.5)`
- Vector: `itp(xq_vec)`
- In-place: `itp(out, xq_vec)`
- Broadcast: `itp.(xq)` or `@. coef * itp(rho)`
- Derivative: `itp(0.5; deriv=DerivOp(1))`

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [0.0, 1.0, 0.5, 0.8, 0.2]  # non-monotone data
itp = pchip_interp(x, y)
itp(1.5)                          # monotone within each monotone segment
itp(1.5; deriv=DerivOp(1))       # first derivative
```
"""
@inline function pchip_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY};
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX, TY}
    Tg = _promote_grid_float(TX, TY)
    extrap_p = _promote_extrap(extrap, _value_type(TY, Tg))
    resolved = _resolve_coeffs(coeffs)
    if resolved isa OnTheFly
        return PchipInterpolant1D(x, y, PchipSlopes(); extrap = extrap_p, search)
    else
        xc = _store_grid(x, Tg)
        Tdy = _output_eltype(eltype(y), Tg)
        dy_p = Vector{Tdy}(undef, length(y))
        _pchip_slopes!(dy_p, xc, y)
        return PchipInterpolant1D(xc, y, dy_p; extrap = extrap_p, search)
    end
end
