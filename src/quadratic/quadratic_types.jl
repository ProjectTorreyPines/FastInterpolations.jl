# ========================================
# Quadratic Interpolant Types
# ========================================
# Type definitions for quadratic interpolation.
# QuadraticBC type alias and _compute_quadratic_coeffs are in quadratic_solver.jl.
# Callable methods are in quadratic_interpolant.jl.
# Oneshot API is in quadratic_oneshot.jl.
#
# Note: PolyFit{D} point validation uses generic `validate_polyfit_points(bc, n)`
# from bc_types.jl (shared with cubic and other interpolators).

"""
    QuadraticInterpolant{Tg,Tv,X,Y,E,P}

Lightweight callable interpolant for quadratic spline interpolation.
Returned by `quadratic_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type (Float32, Float64) for x-coordinates
- `Tv`: Value type (unconstrained)
- `X<:AbstractVector{Tg}`: Type of x-coordinates
- `Y<:AbstractVector{Tv}`: Type of y-values
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `h::Vector{Tg}`: Grid spacing (precomputed, geometry)
- `a::Vector{Tv}`: Quadratic coefficients (value-derived)
- `d::Vector{Tv}`: Slope coefficients (value-derived)
- `extrap::E`: Extrapolation mode (NoExtrap(), ExtendExtrap(), ClampedExtrap(), or WrapExtrap())
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
val = itp(0.5)               # scalar evaluation
vals = itp.([0.5, 1.5])      # broadcast
vals = itp([0.5, 1.5])       # vector call

# Derivatives
d1 = itp(0.5; deriv=DerivOp(1))       # first derivative
d2 = itp(0.5; deriv=DerivOp(2))       # second derivative

# Complex values
x = [0.0, 1.0, 2.0, 3.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im]
itp = quadratic_interp(x, y)
val = itp(0.5)  # returns ComplexF64

# Search policy: AutoSearch adapts to query type (scalar→BinarySearch, vector→LinearBinarySearch)
itp = quadratic_interp(x, y)
val = itp(0.5)               # AutoSearch resolves to BinarySearch() for scalar
itp = quadratic_interp(x, y; search=LinearBinarySearch())  # explicit override
val = itp(0.5; search=BinarySearch())  # per-call override
```
"""
struct QuadraticInterpolant{Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, E<:AbstractExtrap, P<:AbstractSearchPolicy} <: AbstractInterpolant{Tg, Tv}
    x::X
    y::Y
    h::Vector{Tg}   # Grid spacing (geometry, always Tg)
    a::Vector{Tv}   # Quadratic coefficients (value-derived)
    d::Vector{Tv}   # Slope coefficients (value-derived)
    extrap::E        # Extrapolation mode (compile-time specialized)
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: parametric, only calls new (handles validation only)
    function QuadraticInterpolant{Tg,Tv,X,Y,E,P}(
        x::X, y::Y, h::Vector{Tg}, a::Vector{Tv}, d::Vector{Tv}, ev::E, search::P
    ) where {Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, E<:AbstractExtrap, P<:AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"
        new{Tg,Tv,X,Y,E,P}(x, y, h, a, d, ev, search)
    end
end

# ========================================
# Outer Constructor: typed inputs only
# ========================================
# - Call inner constructor
#
# PERFORMANCE: Typed signature + @inline enables compile-time specialization.
# Use quadratic_interp() for automatic type promotion and coefficient computation.
@inline function QuadraticInterpolant(
    x::X,
    y::Y,
    h::Vector{Tg},
    a::Vector{Tv},
    d::Vector{Tv};
    extrap::AbstractExtrap=NoExtrap(),
    search::P=AutoSearch()
) where {Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, P<:AbstractSearchPolicy}
    E = typeof(extrap)
    return QuadraticInterpolant{Tg,Tv,X,Y,E,P}(x, y, h, a, d, extrap, search)
end
