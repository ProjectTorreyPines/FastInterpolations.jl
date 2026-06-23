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
        bc::AbstractBC = NoBC(),
        tension::Real = 0.0,
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX, TY}
    # Periodic extension (no-op for NoBC). bc_eff flips :exclusive → :extended
    # post-extension; :inclusive passes through. Slope-side dispatches on bc_eff.
    x_eff, y_eff, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)
    Tg = _promote_grid_float(eltype(x_eff), eltype(y_eff))
    extrap_p = _promote_extrap(extrap_eff, _value_type(eltype(y_eff), Tg))
    resolved = _resolve_coeffs(coeffs)
    tens_t = Tg(tension)
    # Caching wrap (zero-copy of buffer): post-extension grid → `_CachedVector`
    # (Vector) / `_CachedRange` (Range). Ownership copy in inner ctor's
    # `_convert_copy(x, Tg)`. Mirrors Linear/Constant 1D outer flow.
    x_eff = _cache_axis(x_eff, bc_eff, Tg)

    if resolved isa OnTheFly
        return CardinalInterpolant1D(x_eff, y_eff, CardinalSlopes(tens_t, bc_eff), extrap_p, search, tens_t)
    end
    # PreCompute
    Tdy = _promote_eltype(_coeff_op, Tg, _value_type(eltype(y_eff), Tg))
    dy = Vector{Tdy}(undef, length(x_eff))
    xf = _to_float(x_eff, Tg)
    _cardinal_slopes!(dy, xf, y_eff, tens_t; bc = bc_eff)
    return CardinalInterpolant1D(x_eff, y_eff, dy, extrap_p, search, tens_t)
end
