# ========================================
# Boundary Condition Types
# ========================================
# Types and normalization for specifying endpoint boundary conditions.
# Used by cubic splines and can be extended to other interpolation methods.
#
# Type Hierarchy:
#   AbstractBC{T}
#   ├── PointBC{T}           # Single-point BC (abstract)
#   │   ├── Deriv1{T}            # First derivative
#   │   └── Deriv2{T}            # Second derivative
#   ├── BCPair{T,L,R}        # Both endpoints
#   ├── PeriodicBC{T}        # Periodic BC
#   ├── NaturalBC{T}         # Natural BC (zero curvature at ends)
#   ├── ClampedBC{T}         # Clamped BC (zero slope at ends)
#   ├── Left{T,B}            # Endpoint wrapper: BC at left (x[1])
#   └── Right{T,B}           # Endpoint wrapper: BC at right (x[end])

"""
    AbstractBC{T<:AbstractFloat}

Abstract base type for all boundary condition specifications.

# Subtypes
- `NaturalBC{T}`: Natural BC (zero curvature at both ends) - default
- `ClampedBC{T}`: Clamped BC (zero slope at both ends)
- `PeriodicBC{T}`: Periodic boundary condition
- `PointBC{T}`: Single-point derivative conditions (Deriv1, Deriv2)
- `BCPair{T,L,R}`: Pair of left/right boundary conditions
- `Left{T,B}`: Endpoint wrapper for BC at left (x[1]) - used by quadratic splines
- `Right{T,B}`: Endpoint wrapper for BC at right (x[end]) - used by quadratic splines
"""
abstract type AbstractBC{T<:AbstractFloat} end

"""
    PointBC{T<:AbstractFloat} <: AbstractBC{T}

Abstract type for single-point boundary conditions.
Represents a derivative condition at one endpoint.

# Subtypes
- `Deriv1{T}`: First derivative (slope) BC
- `Deriv2{T}`: Second derivative (curvature) BC
"""
abstract type PointBC{T<:AbstractFloat} <: AbstractBC{T} end

"""
    Deriv1{T<:AbstractFloat} <: PointBC{T}

First derivative (slope) boundary condition: S'(endpoint) = val

# Example
```julia
Deriv1(0.5)  # Slope of 0.5 at endpoint
Deriv1(0)    # Zero slope (horizontal tangent)
```
"""
struct Deriv1{T<:AbstractFloat} <: PointBC{T}
    val::T
end
Deriv1(v::Real) = Deriv1{typeof(float(v))}(float(v))
Deriv1{T}(bc::Deriv1) where {T<:AbstractFloat} = Deriv1{T}(T(bc.val))

"""
    Deriv2{T<:AbstractFloat} <: PointBC{T}

Second derivative (curvature) boundary condition: S''(endpoint) = val

# Example
```julia
Deriv2(0)    # Natural BC (zero curvature)
Deriv2(1.5)  # Specified curvature at endpoint
```
"""
struct Deriv2{T<:AbstractFloat} <: PointBC{T}
    val::T
end
Deriv2(v::Real) = Deriv2{typeof(float(v))}(float(v))
Deriv2{T}(bc::Deriv2) where {T<:AbstractFloat} = Deriv2{T}(T(bc.val))

"""
    BCPair{T, L<:PointBC{T}, R<:PointBC{T}} <: AbstractBC{T}

Container for left and right boundary conditions with type parameters for zero-overhead dispatch.
The BC types are encoded in the type parameters, enabling compile-time specialization.

# Type Parameters
- `T`: Float type
- `L`: Left boundary condition type (Deriv1{T} or Deriv2{T})
- `R`: Right boundary condition type (Deriv1{T} or Deriv2{T})

# Example
```julia
bc = BCPair(Deriv1(0.5), Deriv2(0))  # Left: slope=0.5, Right: natural
```
"""
struct BCPair{T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}} <: AbstractBC{T}
    left::L
    right::R
end

# Convenience constructor from tuple
BCPair(t::Tuple{L, R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}} =
    BCPair{T,L,R}(t[1], t[2])


"""
    PeriodicBC{T<:AbstractFloat} <: AbstractBC{T}

Periodic boundary condition: S(x_0) = S(x_n), S'(x_0) = S'(x_n), S''(x_0) = S''(x_n)

This is a user-facing type. Internally, periodic BC uses Sherman-Morrison
solver with `PeriodicData{T}` for the cache.

# Example
```julia
cache = CubicSplineCache(x; bc=PeriodicBC())
```
"""
struct PeriodicBC{T<:AbstractFloat} <: AbstractBC{T} end
PeriodicBC() = PeriodicBC{Float64}()
PeriodicBC{T}(::PeriodicBC) where {T<:AbstractFloat} = PeriodicBC{T}()

"""
    NaturalBC{T<:AbstractFloat} <: AbstractBC{T}

Natural boundary condition: S''(endpoints) = 0 (zero curvature at both ends).
Equivalent to `BCPair(Deriv2(0), Deriv2(0))`.

This is the default BC for cubic spline interpolation.

# Example
```julia
itp = cubic_interp(x, y; bc=NaturalBC())  # Default
itp = cubic_interp(x, y)                   # Same as above
```
"""
struct NaturalBC{T<:AbstractFloat} <: AbstractBC{T} end
NaturalBC() = NaturalBC{Float64}()
NaturalBC{T}(::NaturalBC) where {T<:AbstractFloat} = NaturalBC{T}()

"""
    ClampedBC{T<:AbstractFloat} <: AbstractBC{T}

Clamped boundary condition: S'(endpoints) = 0 (zero slope at both ends).
Equivalent to `BCPair(Deriv1(0), Deriv1(0))`.

Also known as "complete" spline with zero derivative.

# Example
```julia
itp = cubic_interp(x, y; bc=ClampedBC())
```
"""
struct ClampedBC{T<:AbstractFloat} <: AbstractBC{T} end
ClampedBC() = ClampedBC{Float64}()
ClampedBC{T}(::ClampedBC) where {T<:AbstractFloat} = ClampedBC{T}()


# ========================================
# Type Promotion Helpers
# ========================================
# Generic promotion for extensibility (D0, D3, etc. in future)

"""
    _promote_pointbc(bc::PointBC, ::Type{T}) -> PointBC{T}

Promote a PointBC to a specific float type T.
Extensible: add methods for new PointBC subtypes.
"""
@inline _promote_pointbc(bc::Deriv1, ::Type{T}) where {T<:AbstractFloat} = Deriv1{T}(T(bc.val))
@inline _promote_pointbc(bc::Deriv2, ::Type{T}) where {T<:AbstractFloat} = Deriv2{T}(T(bc.val))


# ========================================
# BC Normalization
# ========================================

"""
    _normalize_bc(bc::AbstractBC, ::Type{T}) -> BCPair{T}

Convert BC specification to normalized BCPair form for cache construction.

Note: PeriodicBC is handled separately via `_is_periodic_bc()` check before
`_normalize_bc` is called. This function only handles derivative BCs.

# Accepted Input Types
- `NaturalBC`: Natural BC (zero curvature) → BCPair(Deriv2(0), Deriv2(0))
- `ClampedBC`: Clamped BC (zero slope) → BCPair(Deriv1(0), Deriv1(0))
- `BCPair`: Left/right BC pair (passed through with type promotion)
- `PointBC` (Deriv1/Deriv2): Single BC applied symmetrically to both ends

# Returns
- `BCPair{T}`: Normalized boundary condition pair
"""
# NaturalBC → BCPair(Deriv2(0), Deriv2(0))
@inline _normalize_bc(::NaturalBC, ::Type{T}) where {T<:AbstractFloat} = BCPair(Deriv2(zero(T)), Deriv2(zero(T)))

# ClampedBC → BCPair(Deriv1(0), Deriv1(0))
@inline _normalize_bc(::ClampedBC, ::Type{T}) where {T<:AbstractFloat} = BCPair(Deriv1(zero(T)), Deriv1(zero(T)))

# BCPair passthrough (already normalized)
@inline _normalize_bc(bc::BCPair{T}, ::Type{T}) where {T<:AbstractFloat} = bc

# BCPair with type promotion
@inline function _normalize_bc(bc::BCPair, ::Type{T}) where {T<:AbstractFloat}
    left_t = _promote_pointbc(bc.left, T)
    right_t = _promote_pointbc(bc.right, T)
    return BCPair(left_t, right_t)
end

# Single PointBC → symmetric BCPair (same BC at both ends)
@inline function _normalize_bc(bc::PointBC{T}, ::Type{T}) where {T<:AbstractFloat}
    return BCPair(bc, bc)
end

# Single PointBC with type promotion
@inline function _normalize_bc(bc::PointBC, ::Type{T}) where {T<:AbstractFloat}
    bc_t = _promote_pointbc(bc, T)
    return BCPair(bc_t, bc_t)
end


# ========================================
# BC Type Predicates
# ========================================

"""
    _is_periodic_bc(bc::AbstractBC) -> Bool

Check if a boundary condition is periodic.
"""
@inline _is_periodic_bc(::AbstractBC) = false  # default for all BC types
@inline _is_periodic_bc(::PeriodicBC) = true   # only PeriodicBC is periodic


# ========================================
# Endpoint-Specific BC Wrappers (Quadratic)
# ========================================

"""
    Left{T, B<:PointBC{T}} <: AbstractBC{T}

Wrapper indicating BC is applied at left endpoint (x[1]).
Used for quadratic splines where only one endpoint BC is specified.

# Example
```julia
bc = Left(Deriv1(0.5))   # slope = 0.5 at left endpoint
bc = Left(Deriv2(0.0))   # curvature = 0 at left endpoint
```
"""
struct Left{T<:AbstractFloat, B<:PointBC{T}} <: AbstractBC{T}
    bc::B
end
# Note: Julia generates outer constructor automatically: Left(bc::B) where {T,B<:PointBC{T}}

"""
    Right{T, B<:PointBC{T}} <: AbstractBC{T}

Wrapper indicating BC is applied at right endpoint (x[end]).
Used for quadratic splines where only one endpoint BC is specified.

# Example
```julia
bc = Right(Deriv1(2.0))  # slope = 2.0 at right endpoint
bc = Right(Deriv2(0.0))  # curvature = 0 at right endpoint
```
"""
struct Right{T<:AbstractFloat, B<:PointBC{T}} <: AbstractBC{T}
    bc::B
end
# Note: Julia generates outer constructor automatically: Right(bc::B) where {T,B<:PointBC{T}}
