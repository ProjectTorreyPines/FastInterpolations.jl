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
        bc::AbstractBC = NoBC(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
        store::StorePolicy = StorePolicy()
    ) where {TX, TY}
    # Periodic extension (no-op for NoBC). bc_eff flips :exclusive → :extended
    # post-extension; :inclusive passes through. Slope-side dispatches on bc_eff.
    x_eff, y_eff, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)
    Tg = _promote_grid_float(eltype(x_eff), eltype(y_eff))
    extrap_p = _promote_extrap(extrap_eff, _value_type(eltype(y_eff), Tg))
    resolved = _resolve_coeffs(coeffs)
    # Caching wrap (zero-copy of buffer); ownership copy in inner ctor.
    x_eff = _cache_axis(x_eff, bc_eff, Tg)

    if resolved isa OnTheFly
        return AkimaInterpolant1D(x_eff, y_eff, AkimaSlopes(bc_eff), extrap_p, search; store = store)
    end
    # PreCompute
    Tdy = _promote_eltype(_coeff_op, Tg, _value_type(eltype(y_eff), Tg))
    dy = Vector{Tdy}(undef, length(x_eff))
    xf = _to_float(x_eff, Tg)
    _akima_slopes!(dy, xf, y_eff; bc = bc_eff)
    return AkimaInterpolant1D(x_eff, y_eff, dy, extrap_p, search; store = store)
end
