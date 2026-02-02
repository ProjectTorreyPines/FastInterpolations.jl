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
    QuadraticInterpolant{Tg,Tv,X,Y,P}

Lightweight callable interpolant for quadratic spline interpolation.
Returned by `quadratic_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type (Float32, Float64) for x-coordinates
- `Tv`: Value type for y-values (can be Tg, Complex{Tg}, or other Number)
- `X<:AbstractVector{Tg}`: Type of x-coordinates
- `Y<:AbstractVector{Tv}`: Type of y-values
- `P<:AbstractSearchPolicy`: Search policy type

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `h::Vector{Tg}`: Grid spacing (precomputed, geometry)
- `a::Vector{Tv}`: Quadratic coefficients (value-derived)
- `d::Vector{Tv}`: Slope coefficients (value-derived)
- `extrap::ExtrapVal`: Extrapolation mode
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
val = itp(0.5)               # scalar evaluation
vals = itp.([0.5, 1.5])      # broadcast
vals = itp([0.5, 1.5])       # vector call

# Derivatives
d1 = itp(0.5; deriv=1)       # first derivative
d2 = itp(0.5; deriv=2)       # second derivative

# Complex values
x = [0.0, 1.0, 2.0, 3.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im]
itp = quadratic_interp(x, y)
val = itp(0.5)  # returns ComplexF64

# Custom search policy
itp = quadratic_interp(x, y; search=LinearBinary())
val = itp(0.5)               # uses LinearBinary() by default
val = itp(0.5; search=Binary())  # override with Binary()
```
"""
struct QuadraticInterpolant{Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, P<:AbstractSearchPolicy} <: AbstractInterpolant{Tg, Tv}
    x::X
    y::Y
    h::Vector{Tg}   # Grid spacing (geometry, always Tg)
    a::Vector{Tv}   # Quadratic coefficients (value-derived)
    d::Vector{Tv}   # Slope coefficients (value-derived)
    extrap::ExtrapVal
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: parametric, only calls new (handles validation only)
    function QuadraticInterpolant{Tg,Tv,X,Y,P}(
        x::X, y::Y, h::Vector{Tg}, a::Vector{Tv}, d::Vector{Tv}, ev::ExtrapVal, search::P
    ) where {Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, P<:AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"
        new{Tg,Tv,X,Y,P}(x, y, h, a, d, ev, search)
    end
end

# ========================================
# Outer Constructor: handles all logic
# ========================================
# - Type conversion (_promote_itp_inputs)
# - BC conversion (_promote_bc)
# - Coefficient computation (_compute_quadratic_coeffs)
# - Symbol → Val dispatch
# - Call inner constructor
function QuadraticInterpolant(
    x::AbstractVector,
    y::AbstractVector;
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
)
    x_p, y_p = _promote_itp_inputs(x, y)
    bc_p = _promote_bc(bc, eltype(x_p))

    # Validate PolyFit{D} point requirements (e.g., CubicFit needs 4+ points)
    validate_polyfit_points(bc_p, length(x_p))

    # Compute coefficients (h::Tg, d::Tv, a::Tv)
    h, d, a = _compute_quadratic_coeffs(x_p, y_p, bc_p)

    X = typeof(x_p)
    Y = typeof(y_p)
    Tg = eltype(x_p)
    Tv = eltype(y_p)
    P = typeof(search)

    @_dispatch_extrap extrap => ev begin
        return QuadraticInterpolant{Tg,Tv,X,Y,P}(x_p, y_p, h, a, d, ev, search)
    end
end
