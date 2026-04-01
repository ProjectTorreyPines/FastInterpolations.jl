# ========================================
# Cardinal Interpolant Callable Methods
# ========================================
# Callable methods for CardinalInterpolant1D and 2-arg API.

# ========================================
# Protocol Trait Implementations
# ========================================

@inline function _itp_eval_scalar(itp::CardinalInterpolant1D, xq, extrap, op, searcher)
    return _cubic_hermite_eval_at_point(itp.x, itp.spacing, itp.y, itp.dy, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::CardinalInterpolant1D, xq, extrap, op, searcher)
    return _cubic_hermite_vector_loop!(output, itp.x, itp.spacing, itp.y, itp.dy, xq, extrap, op, searcher)
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    cardinal_interp(x, y; tension=0.0, extrap=NoExtrap(), search=AutoSearch()) -> CardinalInterpolant1D

Create a callable cardinal spline interpolant.

Default `tension=0` gives **Catmull-Rom** (central finite difference slopes).
Increasing tension reduces overshoot; `tension=1` gives flat (zero-slope) segments.

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
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX <: Real, TY}
    x_p, y_p = _promote_itp_inputs(x, y)
    Tg = eltype(x_p)
    dy_p = similar(y_p)
    _cardinal_slopes!(dy_p, x_p, y_p, Tg(tension))
    extrap_p = _promote_extrap(extrap, eltype(y_p))
    return CardinalInterpolant1D(x_p, y_p, dy_p; tension = Tg(tension), extrap = extrap_p, search)
end
