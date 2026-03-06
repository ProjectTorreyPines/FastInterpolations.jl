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
- `Tg<:AbstractFloat`: Grid type (Float32, Float64) for x-coordinates
- `Tv`: Value type (unconstrained)
- `X<:AbstractVector{Tg}`: Type of x-coordinates
- `Y<:AbstractVector{Tv}`: Type of y-values
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `SD<:AbstractSide`: Side selection type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
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
struct ConstantInterpolant{Tg <: AbstractFloat, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy} <: AbstractInterpolant{Tg, Tv}
    x::X
    y::Y
    extrap::E        # Extrapolation mode (compile-time specialized)
    side::SD         # Side selection (compile-time specialized)
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: parametric, only calls new (handles validation only)
    function ConstantInterpolant{Tg, Tv, X, Y, E, SD, P}(
            x::X, y::Y, ev::E, sv::SD, search::P
        ) where {Tg <: AbstractFloat, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"
        return new{Tg, Tv, X, Y, E, SD, P}(x, y, ev, sv, search)
    end
end

# ========================================
# Outer Constructor: typed inputs only
# ========================================
# - Call inner constructor
#
# PERFORMANCE: Typed signature + @inline enables compile-time specialization.
# Use constant_interp() for automatic type promotion from Real inputs.
@inline function ConstantInterpolant(
        x::X,
        y::Y;
        extrap::AbstractExtrap = NoExtrap(),
        side::AbstractSide = NearestSide(),
        search::P = AutoSearch()
    ) where {Tg <: AbstractFloat, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, P <: AbstractSearchPolicy}
    E = typeof(extrap)
    SD = typeof(side)
    return ConstantInterpolant{Tg, Tv, X, Y, E, SD, P}(x, y, extrap, side, search)
end
