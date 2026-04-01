# ========================================
# Hermite Data Wrapper & Interpolant Types
# ========================================
# AbstractDataWrapper enables dispatch separation: Hermite(y,dy) vs raw AbstractVector y.
# CubicHermiteInterpolant1D is the callable interpolant for Hermite-based cubic interpolation.

# ========================================
# Abstract Data Wrapper
# ========================================

"""
    AbstractDataWrapper

Base type for data wrappers that change how interpolation coefficients are obtained.

`cubic_interp(x, y, xq)` dispatches on `y::AbstractVector` (spline path).
`cubic_interp(x, h::Hermite, xq)` dispatches on `h::AbstractDataWrapper` (Hermite path).
"""
abstract type AbstractDataWrapper end

# ========================================
# Hermite Data Wrapper
# ========================================

"""
    Hermite(y, dy)

Wrap user-supplied function values and first derivatives for Hermite interpolation.
Use with `cubic_interp` to skip the global spline solve and use the provided
slopes directly.

# 1D Usage
```julia
y  = sin.(x)
dy = cos.(x)   # exact first derivative
val = cubic_interp(x, Hermite(y, dy), 0.5)
itp = cubic_interp(x, Hermite(y, dy))
itp(0.5)
```

# Notes
- No boundary conditions (`bc`) — user slopes replace them.
- No auto-caching (`autocache`) — no global solve to cache.
- C\$^1\$ continuity (not C\$^2\$ like cubic spline).
- `deriv`, `extrap`, and `search` kwargs are supported as usual.
"""
struct Hermite{Y, DY} <: AbstractDataWrapper
    y::Y
    dy::DY

    # 1D constructor: both AbstractVector, validated
    function Hermite(y::AbstractVector, dy::AbstractVector)
        length(y) == length(dy) || throw(
            ArgumentError(
                "y (length $(length(y))) and dy (length $(length(dy))) must have same length"
            )
        )
        return new{typeof(y), typeof(dy)}(y, dy)
    end
end

# ========================================
# Type Promotion for Hermite Data
# ========================================

"""
    _promote_itp_inputs(x, h::Hermite) -> (x_typed, h_typed::Hermite)

Promote grid and Hermite data (y, dy) to compatible Float types.
Grid type `Tg` is determined from all three inputs (x, y, dy) jointly
via `_promote_grid_float`, so no precision is lost when dy has a wider
type than y (e.g., `y::Float32, dy::Float64 → Tg = Float64`).
"""
@inline function _promote_itp_inputs(
        x::AbstractVector{TX},
        h::Hermite{<:AbstractVector{TY}, <:AbstractVector{TDY}}
    ) where {TX <: Real, TY, TDY}
    # Grid type considers x, y, AND dy — widest promotable type wins
    Tg_y = _promote_grid_float(TX, TY)
    Tg = TDY <: _PromotableValue ? promote_type(Tg_y, float(_real_eltype(TDY))) : Tg_y
    x_p = _to_float(x, Tg)
    if TY <: _PromotableValue
        _, y_p = _promote_value_type(h.y, Tg)
    else
        y_p = h.y
    end
    if TDY <: _PromotableValue
        _, dy_p = _promote_value_type(h.dy, Tg)
    else
        dy_p = h.dy
    end
    return x_p, Hermite(y_p, dy_p)
end

"""
    _promote_itp_inputs(x, h::Hermite, xq) -> (x_typed, h_typed::Hermite, xq_typed)

Promote grid, Hermite data, and vector query to compatible Float types.
"""
@inline function _promote_itp_inputs(
        x::AbstractVector{TX},
        h::Hermite{<:AbstractVector{TY}, <:AbstractVector},
        xq::AbstractVector{TQ}
    ) where {TX <: Real, TY, TQ <: Real}
    x_p, h_p = _promote_itp_inputs(x, h)
    Tg = eltype(x_p)
    xq_p = _to_float(xq, Tg)
    return x_p, h_p, xq_p
end

# ========================================
# Cubic Hermite Interpolant (1D)
# ========================================

"""
    CubicHermiteInterpolant1D{Tg, Tv, X, Y, DY, S, E, P}

Callable interpolant for cubic Hermite interpolation with user-supplied slopes.
Returned by `cubic_interp(x, Hermite(y, dy))` (2-argument form).

Stores precomputed grid spacing for O(1) `h`/`inv_h` lookup on uniform grids.
Evaluation uses `_hermite_kernel_1d` (derivative-based Hermite basis functions),
NOT `_cubic_kernel` (moment-based spline formulation).

# Type Parameters
- `Tg<:AbstractFloat`: Grid coordinate type
- `Tv`: Value type (unconstrained — supports Complex, etc.)
- `X<:AbstractVector{Tg}`: Grid vector type (preserves Range for O(1) lookup)
- `Y<:AbstractVector{Tv}`: Values vector type
- `DY<:AbstractVector{Tv}`: Slopes vector type
- `S<:AbstractGridSpacing{Tg}`: Grid spacing type
- `E<:AbstractExtrap`: Extrapolation mode type
- `P<:AbstractSearchPolicy`: Search policy type

# Usage
```julia
itp = cubic_interp(x, Hermite(y, dy))
itp(0.5)                          # scalar query
itp(xq_vec)                       # vector query (allocating)
itp(out, xq_vec)                  # vector query (in-place)
itp(0.5; deriv=DerivOp(1))       # first derivative
itp(0.5; search=BinarySearch())  # override search
```
"""
struct CubicHermiteInterpolant1D{
        Tg <: AbstractFloat,
        Tv,
        X <: AbstractVector{Tg},
        Y <: AbstractVector{Tv},
        DY <: AbstractVector{Tv},
        S <: AbstractGridSpacing{Tg},
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
    } <: AbstractInterpolant1D{Tg, Tv}
    x::X
    y::Y
    dy::DY
    spacing::S
    extrap::E
    search_policy::P

    function CubicHermiteInterpolant1D{Tg, Tv, X, Y, DY, S, E, P}(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, dy::AbstractVector{Tv},
            spacing::S, extrap::E, search::P
        ) where {
            Tg <: AbstractFloat, Tv,
            X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, DY <: AbstractVector{Tv},
            S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy,
        }
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) == length(dy) || _throw_length_mismatch(length(x), length(dy), "x", "dy")
        xc, yc, dyc = copy(x), copy(y), copy(dy)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(dyc), S, E, P}(
            xc, yc, dyc, spacing, extrap, search
        )
    end
end
