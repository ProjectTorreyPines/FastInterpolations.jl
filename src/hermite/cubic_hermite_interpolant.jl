# ========================================
# Cubic Hermite Interpolant Callable Methods
# ========================================
# Callable methods for CubicHermiteInterpolant1D and 2-arg API.
# Type definition is in hermite_types.jl.
# Oneshot API (cubic_interp 3-arg) is in hermite_oneshot.jl.

# ========================================
# Protocol Trait Implementations
# ========================================
# Generic callables inherited from AbstractInterpolant1D (interpolant_protocol.jl).
# _itp_grid, _itp_extrap, _itp_search use defaults (itp.x, itp.extrap, itp.search_policy).

@inline function _itp_eval_scalar(itp::CubicHermiteInterpolant1D, xq, extrap, op, searcher)
    return _cubic_hermite_eval_at_point(itp.x, itp.spacing, itp.y, itp.dy, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::CubicHermiteInterpolant1D, xq, extrap, op, searcher)
    return _cubic_hermite_vector_loop!(output, itp.x, itp.spacing, itp.y, itp.dy, xq, extrap, op, searcher)
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
    return CubicHermiteInterpolant1D{Tg, Tv, X, Y, DY, S, E, P}(
        x, y, dy, spacing, extrap, search
    )
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    cubic_interp(x, Hermite(y, dy); extrap=NoExtrap(), search=AutoSearch()) -> CubicHermiteInterpolant1D

Create a callable cubic Hermite interpolant from user-supplied slopes.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `Hermite(y, dy)`: Function values and first derivatives at grid points
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
itp = cubic_interp(x, Hermite(y, dy))
itp(1.0)                          # ≈ sin(1.0)
itp(1.0; deriv=DerivOp(1))       # ≈ cos(1.0)
```
"""
function cubic_interp(
        x::AbstractVector{Tg},
        h::Hermite{<:AbstractVector{Tv}, <:AbstractVector{Tv}};
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
        bc::Union{Nothing, AbstractBC} = nothing,
        autocache::Union{Nothing, Bool} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    _reject_hermite_kwargs(bc, autocache)
    x_f = _to_float(x, Tg)
    extrap_p = _promote_extrap(extrap, Tv)
    return CubicHermiteInterpolant1D(x_f, h.y, h.dy; extrap = extrap_p, search)
end

# Real wrapper for 2-arg form (handles Integer grids, mixed types)
function cubic_interp(
        x::AbstractVector{TX},
        h::Hermite;
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
        bc::Union{Nothing, AbstractBC} = nothing,
        autocache::Union{Nothing, Bool} = nothing
    ) where {TX <: Real}
    _reject_hermite_kwargs(bc, autocache)
    x_p, y_p = _promote_itp_inputs(x, h.y)
    Tv_float = eltype(y_p)
    dy_p = convert(Vector{Tv_float}, h.dy)
    extrap_p = _promote_extrap(extrap, Tv_float)
    return CubicHermiteInterpolant1D(x_p, y_p, dy_p; extrap = extrap_p, search)
end
