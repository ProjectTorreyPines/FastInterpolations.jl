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
    QuadraticInterpolant{Tg,Tv,X,Y,S,E,P,BC}

Lightweight callable interpolant for quadratic spline interpolation.
Returned by `quadratic_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type (Float32, Float64) for x-coordinates
- `Tv`: Value type (unconstrained)
- `X<:AbstractVector{Tg}`: Type of x-coordinates
- `Y<:AbstractVector{Tv}`: Type of y-values
- `S<:AbstractGridSpacing{Tg}`: Grid spacing type (ScalarSpacing or VectorSpacing)
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type
- `BC<:QuadraticBC`: Boundary condition type (retained for adjoint/matrix convenience)

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `spacing::S`: Precomputed grid spacing (avoids TwicePrecision overhead on Range grids)
- `a::Vector{Tv}`: Quadratic coefficients (value-derived)
- `d::Vector{Tv}`: Slope coefficients (value-derived)
- `extrap::E`: Extrapolation mode (NoExtrap(), ExtendExtrap(), ClampExtrap(), or WrapExtrap())
- `search_policy::P`: Default search policy for interval lookup
- `bc::BC`: Boundary condition used during construction (retained for `Matrix(itp, xq)` convenience)

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
struct QuadraticInterpolant{Tg <: AbstractFloat, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy, BC <: QuadraticBC} <: AbstractInterpolant1D{Tg, Tv}
    x::X
    y::Y
    spacing::S       # Precomputed grid spacing (ScalarSpacing for Range, VectorSpacing for Vector)
    a::Vector{Tv}   # Quadratic coefficients (value-derived)
    d::Vector{Tv}   # Slope coefficients (value-derived)
    extrap::E        # Extrapolation mode (compile-time specialized)
    search_policy::P  # Default search policy (immutable, thread-safe)
    bc::BC           # Boundary condition (retained for Matrix(itp, xq) convenience)

    # Inner constructor: parametric, only calls new (handles validation only)
    function QuadraticInterpolant{Tg, Tv, X, Y, S, E, P, BC}(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, spacing::S, a::Vector{Tv}, d::Vector{Tv}, ev::E, search::P, bc::BC
        ) where {Tg <: AbstractFloat, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy, BC <: QuadraticBC}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        length(x) >= 2 || _throw_grid_too_small(length(x))
        # Copy to ensure immutability: once constructed, the interpolant owns
        # its data and returns identical results regardless of external mutation.
        # copy() on immutable Range types is a no-op (zero allocation).
        # typeof() rebinds X/Y to the post-copy concrete type (e.g. SubArray → Vector).
        xc, yc = copy(x), copy(y)
        return new{Tg, Tv, typeof(xc), typeof(yc), S, E, P, BC}(xc, yc, spacing, a, d, ev, search, bc)
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
        spacing::S,
        a::Vector{Tv},
        d::Vector{Tv};
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        search::P = AutoSearch()
    ) where {Tg <: AbstractFloat, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, S <: AbstractGridSpacing{Tg}, P <: AbstractSearchPolicy}
    E = typeof(extrap)
    BC = typeof(bc)
    return QuadraticInterpolant{Tg, Tv, X, Y, S, E, P, BC}(x, y, spacing, a, d, extrap, search, bc)
end
