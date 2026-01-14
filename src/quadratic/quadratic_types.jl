# ========================================
# Quadratic Interpolant Types
# ========================================
# Type definitions for quadratic interpolation.
# QuadraticBC type alias and _compute_quadratic_coeffs are in quadratic_solver.jl.
# Callable methods are in quadratic_interpolant.jl.
# Oneshot API is in quadratic_oneshot.jl.

"""
    QuadraticInterpolant{T,X,Y}

Lightweight callable interpolant for quadratic spline interpolation.
Returned by `quadratic_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `h::Vector{T}`: Grid spacing (precomputed)
- `a::Vector{T}`: Quadratic coefficients
- `d::Vector{T}`: Slope coefficients
- `mode::ExtrapVal`: Extrapolation mode

# Usage
```julia
itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
val = itp(0.5)               # scalar evaluation
vals = itp.([0.5, 1.5])      # broadcast
vals = itp([0.5, 1.5])       # vector call

# Derivatives
d1 = itp(0.5; deriv=1)       # first derivative
d2 = itp(0.5; deriv=2)       # second derivative
```
"""
struct QuadraticInterpolant{T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}} <: AbstractInterpolant{T}
    x::X
    y::Y
    h::Vector{T}
    a::Vector{T}
    d::Vector{T}
    mode::ExtrapVal

    function QuadraticInterpolant(
        x::X, y::Y;
        bc::QuadraticBC{T}=Left(ParabolaFit{T}()),
        extrap::Symbol=:none
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"

        # Compute coefficients (no caching)
        h, d, a = _compute_quadratic_coeffs(x, y, bc)

        @_dispatch_extrap extrap => ev begin
            return new{T,X,Y}(x, y, h, a, d, ev)
        end
    end
end
