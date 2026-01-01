# ========================================
# Quadratic Spline Type Definitions
# ========================================
#
# Types for quadratic spline interpolation cache.
# Simpler than cubic: no LU factorization, just h and inv_h.

"""
    QuadraticSplineCache{T, X}

Cache for quadratic spline x-grid preprocessing.
Stores grid spacing and inverse spacing for efficient coefficient computation.

Unlike CubicSplineCache, this does not include LU factorization since
quadratic splines use a simple recurrence relation for slope computation.

# Fields
- `x::X`: Grid points (immutable after construction)
- `h::Vector{T}`: Grid spacing h[i] = x[i+1] - x[i]
- `inv_h::Vector{T}`: Inverse spacing for efficient secant computation

# Example
```julia
x = collect(range(0.0, 1.0, 11))
cache = QuadraticSplineCache(x)
```
"""
struct QuadraticSplineCache{T<:AbstractFloat, X<:AbstractVector{T}}
    x::X
    h::Vector{T}
    inv_h::Vector{T}
end

"""
    QuadraticSplineCache(x::AbstractVector{T}) -> QuadraticSplineCache{T,X}

Construct a QuadraticSplineCache from grid points.

# Arguments
- `x`: Strictly increasing grid points (at least 2 elements)

# Throws
- `ArgumentError` if x has fewer than 2 elements
- `ArgumentError` if x is not strictly increasing
"""
function QuadraticSplineCache(x::X) where {T<:AbstractFloat, X<:AbstractVector{T}}
    n = length(x)
    n >= 2 || throw(ArgumentError("x must have at least 2 elements, got $n"))

    h = Vector{T}(undef, n-1)
    inv_h = Vector{T}(undef, n-1)

    @inbounds for i in 1:(n-1)
        h[i] = x[i+1] - x[i]
        h[i] > 0 || throw(ArgumentError("x must be strictly increasing at index $i"))
        inv_h[i] = inv(h[i])
    end

    return QuadraticSplineCache{T,X}(x, h, inv_h)
end
