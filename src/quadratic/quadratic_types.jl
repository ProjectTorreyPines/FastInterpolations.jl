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
    QuadraticInterpolant{T,X,Y,P}

Lightweight callable interpolant for quadratic spline interpolation.
Returned by `quadratic_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `h::Vector{T}`: Grid spacing (precomputed)
- `a::Vector{T}`: Quadratic coefficients
- `d::Vector{T}`: Slope coefficients
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

# Custom search policy
itp = quadratic_interp(x, y; search=LinearBinary())
val = itp(0.5)               # uses LinearBinary() by default
val = itp(0.5; search=Binary())  # override with Binary()
```
"""
struct QuadraticInterpolant{T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}, P<:AbstractSearchPolicy} <: AbstractInterpolant{T}
    x::X
    y::Y
    h::Vector{T}
    a::Vector{T}
    d::Vector{T}
    extrap::ExtrapVal
    search_policy::P  # Default search policy (immutable, thread-safe)

    function QuadraticInterpolant(
        x::X, y::Y;
        bc::QuadraticBC{T}=Left(ParabolaFit{T}()),
        extrap::Symbol=:none,
        search::P=Binary()
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}, P<:AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"

        # Validate PolyFit{D} point requirements (e.g., CubicFit needs 4+ points)
        validate_polyfit_points(bc, length(x))

        # Compute coefficients (no caching)
        h, d, a = _compute_quadratic_coeffs(x, y, bc)

        @_dispatch_extrap extrap => ev begin
            return new{T,X,Y,P}(x, y, h, a, d, ev, search)
        end
    end
end
