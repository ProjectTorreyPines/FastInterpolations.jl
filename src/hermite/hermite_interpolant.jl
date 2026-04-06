# ========================================
# Hermite Interpolant Callable Methods
# ========================================
# Callable methods for CubicHermiteInterpolant1D and interpolant-form API.
# Type definition is in hermite_types.jl.
# Oneshot API (hermite_interp 4-arg) is in hermite_oneshot.jl.

# ========================================
# Protocol Trait Implementations (shared for all Hermite family)
# ========================================
# Dispatches on AbstractHermiteInterpolant1D: covers Hermite, PCHIP, Cardinal, Akima.
# All subtypes store (x, y, dy, spacing) and use the same Hermite kernel.
# _itp_grid, _itp_extrap, _itp_search use defaults (itp.x, itp.extrap, itp.search_policy).

@inline function _itp_eval_scalar(itp::AbstractHermiteInterpolant1D, xq, extrap, op, searcher)
    return _hermite_eval_at_point(itp.x, itp.spacing, itp.y, itp.dy, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::AbstractHermiteInterpolant1D, xq, extrap, op, searcher)
    return _hermite_vector_loop!(output, itp.x, itp.spacing, itp.y, itp.dy, xq, extrap, op, searcher)
end

# ========================================
# Outer Constructor: typed inputs only
# ========================================

@inline function CubicHermiteInterpolant1D(
        x::X,
        y::Y,
        dy::DY;
        extrap::AbstractExtrap = NoExtrap(),
        search::P = AutoSearch()
    ) where {
        Tg <: AbstractFloat, Tv,
        X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, DY <: AbstractVector{Tv},
        P <: AbstractSearchPolicy,
    }
    E = typeof(extrap)
    spacing = _create_spacing(x)
    S = typeof(spacing)
    return CubicHermiteInterpolant1D{Tg, Tv, X, Y, DY, S, E, P, PreCompute}(
        x, y, dy, spacing, extrap, search
    )
end

# ========================================
# Interpolant Form: Return Callable
# ========================================

"""
    hermite_interp(x, y, dy; extrap=NoExtrap(), search=AutoSearch()) -> CubicHermiteInterpolant1D

Create a callable cubic Hermite interpolant from user-supplied slopes.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: Function values at grid points
- `dy::AbstractVector`: First derivatives at grid points
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`)
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

# Returns
`CubicHermiteInterpolant1D` callable object. Supports:
- Scalar: `itp(0.5)`
- Vector: `itp(xq_vec)`
- In-place: `itp(out, xq_vec)`
- Broadcast: `itp.(xq)` or `@. coef * itp(rho)`
- Derivative: `itp(0.5; deriv=DerivOp(1))`

# Example
```julia
x  = range(0, 2π, 100)
y  = sin.(x)
dy = cos.(x)
itp = hermite_interp(x, y, dy)
itp(1.0)                          # ≈ sin(1.0)
itp(1.0; deriv=DerivOp(1))       # ≈ cos(1.0)
```
"""
@inline function hermite_interp(
        x::AbstractVector{TX},
        y::AbstractVector{TY},
        dy::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
    ) where {TX <: Real, TY}
    x_p, y_p, dy_p = _promote_hermite_inputs(x, y, dy)
    extrap_p = _promote_extrap(extrap, eltype(y_p))
    return CubicHermiteInterpolant1D(x_p, y_p, dy_p; extrap = extrap_p, search)
end
