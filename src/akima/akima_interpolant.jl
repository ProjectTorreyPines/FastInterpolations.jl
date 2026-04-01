# ========================================
# Akima Interpolant Callable Methods
# ========================================

# ========================================
# Protocol Trait Implementations
# ========================================

@inline function _itp_eval_scalar(itp::AkimaInterpolant1D, xq, extrap, op, searcher)
    return _cubic_hermite_eval_at_point(itp.x, itp.spacing, itp.y, itp.dy, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::AkimaInterpolant1D, xq, extrap, op, searcher)
    return _cubic_hermite_vector_loop!(output, itp.x, itp.spacing, itp.y, itp.dy, xq, extrap, op, searcher)
end

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
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX <: Real, TY}
    x_p, y_p = _promote_itp_inputs(x, y)
    dy_p = similar(y_p)
    _akima_slopes!(dy_p, x_p, y_p)
    extrap_p = _promote_extrap(extrap, eltype(y_p))
    return AkimaInterpolant1D(x_p, y_p, dy_p; extrap = extrap_p, search)
end
