# ========================================
# Constant Interpolant Types
# ========================================
# Type definition for ConstantInterpolant.
# Constructor and callable methods are in constant_interpolant.jl.
# Internal evaluation functions are in constant_oneshot.jl.

"""
    ConstantInterpolant{Tg,Tv,X,Y,S,E,SD,P}

Lightweight callable interpolant for constant (step) interpolation.
Returned by `constant_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg`: Grid type (unconstrained) for x-coordinates
- `Tv`: Value type (unconstrained)
- `X<:AbstractVector{Tg}`: Type of x-coordinates
- `Y<:AbstractVector{Tv}`: Type of y-values
- `S<:AbstractGridSpacing{Tg}`: Grid spacing type (ScalarSpacing for Range, VectorSpacing for Vector)
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `SD<:AbstractSide`: Side selection type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `spacing::S`: Precomputed grid spacing (avoids TwicePrecision overhead on Range grids)
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
struct ConstantInterpolant{Tg, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy} <: AbstractInterpolant1D{Tg, Tv}
    x::X
    y::Y
    spacing::S       # Grid spacing (ScalarSpacing for Range, VectorSpacing for Vector)
    extrap::E        # Extrapolation mode (compile-time specialized)
    side::SD         # Side selection (compile-time specialized)
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: parametric, only calls new (handles validation only)
    function ConstantInterpolant{Tg, Tv, X, Y, S, E, SD, P}(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, spacing::S, ev::E, sv::SD, search::P
        ) where {Tg, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || _throw_grid_too_small(length(x))
        # Copy to ensure immutability: once constructed, the interpolant owns
        # its data and returns identical results regardless of external mutation.
        # copy() on immutable Range types is a no-op (zero allocation).
        # typeof() rebinds X/Y to the post-copy concrete type (e.g. SubArray → Vector).
        xc, yc = copy(x), copy(y)
        return new{Tg, Tv, typeof(xc), typeof(yc), S, E, SD, P}(xc, yc, spacing, ev, sv, search)
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
    ) where {Tg, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, P <: AbstractSearchPolicy}
    E = typeof(extrap)
    SD = typeof(side)
    spacing = _create_spacing(x)
    S = typeof(spacing)
    return ConstantInterpolant{Tg, Tv, X, Y, S, E, SD, P}(x, y, spacing, extrap, side, search)
end
