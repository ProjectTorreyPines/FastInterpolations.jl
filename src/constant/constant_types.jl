# ========================================
# Constant Interpolant Types
# ========================================
# Type definition for ConstantInterpolant.
# Constructor and callable methods are in constant_interpolant.jl.
# Internal evaluation functions are in constant_oneshot.jl.

"""
    ConstantInterpolant{Tg,Tv,X,Y,E,SD,P}

Lightweight callable interpolant for constant (step) interpolation.
Returned by `constant_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg`: Grid type (unconstrained) for x-coordinates
- `Tv`: Value type (unconstrained)
- `X<:AbstractVector{Tg}`: Grid vector type — `_CachedRange{Tg}` for Range
        input, `_CachedVector{Tg,Tinv}` for Vector input,
        `_ExclusivePeriodicAxis` for Vector + `:exclusive` PeriodicBC.
- `Y<:AbstractVector{Tv}`: y values — plain `Vector{Tv}` for non-periodic,
        `_ExclusivePeriodicData{Tv,1}` for Vector + `:exclusive` PeriodicBC.
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `SD<:AbstractSide`: Side selection type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type

# Fields
- `x::X`: x-coordinates (sorted, wrapped — grid is the source of truth for spacing)
- `y::Y`: y-values (cyclic-wrapped for `:exclusive` PeriodicBC Vector path)
- `extrap::E`: Extrapolation mode (NoExtrap(), ExtendExtrap(), ClampExtrap(), or WrapExtrap())
- `side::SD`: Side selection (NearestSide(), LeftSide(), RightSide())
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
itp = constant_interp(x, y)  # default: extrap=NoExtrap(), side=NearestSide()
val = itp(0.5)               # scalar evaluation
vals = itp.(query_points)    # broadcast

# With Complex values
x = [0.0, 1.0, 2.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im]
itp = constant_interp(x, y)
val = itp(0.5)  # returns ComplexF64

# With options
itp_left = constant_interp(x, y; side=LeftSide())
itp_wrap = constant_interp(x, y; extrap=WrapExtrap(), side=RightSide())

# Search policy: AutoSearch adapts to query type (scalar→BinarySearch, vector→LinearBinarySearch)
itp = constant_interp(x, y)
val = itp(0.5)               # AutoSearch resolves to BinarySearch() for scalar
itp = constant_interp(x, y; search=LinearBinarySearch())  # explicit override
val = itp(0.5; search=BinarySearch())  # per-call override
```
"""
struct ConstantInterpolant{Tg, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy} <: AbstractInterpolant1D{Tg, Tv}
    x::X
    y::Y
    extrap::E        # Extrapolation mode (compile-time specialized)
    side::SD         # Side selection (compile-time specialized)
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner: `_cache_axis` (insurance) then `_convert_copy` for ownership.
    # `{Tg, Tv}` parametrized directly — selection kernel → raw eltype
    # contract (Int in → Int out), unlike arithmetic methods that thread
    # `_promote_grid_float(TX, TY)` through here.
    function ConstantInterpolant(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, ev::E, sv::SD, search::P;
            bc::AbstractBC = NoBC()
        ) where {Tg, Tv, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy}
        _check_compatible_length(x, y)
        length(x) >= 2 || _throw_grid_too_small(length(x))
        xc = _convert_copy(_cache_axis(x, bc, Tg), Tg)
        yc = _convert_copy(y, Tv)
        return new{Tg, Tv, typeof(xc), typeof(yc), E, SD, P}(xc, yc, ev, sv, search)
    end
end

# Outer constructor: convenience kwarg wrapper. Wraps the axis here so the
# inner ctor's `_cache_axis` insurance is an idempotent passthrough.
@inline function ConstantInterpolant(
        x::AbstractVector{Tg},
        y::AbstractVector;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        side::AbstractSide = NearestSide(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg}
    x_eff = _cache_axis(x, bc, Tg)
    return ConstantInterpolant(x_eff, y, extrap, side, search; bc = bc)
end
