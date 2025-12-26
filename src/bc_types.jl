# ========================================
# Boundary Condition Types
# ========================================
# Types and normalization for specifying endpoint boundary conditions.
# Used by cubic splines and can be extended to other interpolation methods.
#
# Type Hierarchy:
#   AbstractBC{T}
#   ├── PointBC{T}           # Single-point BC (abstract)
#   │   ├── D1{T}            # First derivative
#   │   └── D2{T}            # Second derivative
#   ├── BCPair{T,L,R}        # Both endpoints
#   ├── PeriodicBC{T}        # Periodic BC
#   ├── NaturalBC{T}         # Natural BC (zero curvature at ends)
#   └── ClampedBC{T}         # Clamped BC (zero slope at ends)

"""
    AbstractBC{T<:AbstractFloat}

Abstract base type for all boundary condition specifications.

# Subtypes
- `NaturalBC{T}`: Natural BC (zero curvature at both ends) - default
- `ClampedBC{T}`: Clamped BC (zero slope at both ends)
- `PeriodicBC{T}`: Periodic boundary condition
- `PointBC{T}`: Single-point derivative conditions (D1, D2)
- `BCPair{T,L,R}`: Pair of left/right boundary conditions
"""
abstract type AbstractBC{T<:AbstractFloat} end

"""
    PointBC{T<:AbstractFloat} <: AbstractBC{T}

Abstract type for single-point boundary conditions.
Represents a derivative condition at one endpoint.

# Subtypes
- `D1{T}`: First derivative (slope) BC
- `D2{T}`: Second derivative (curvature) BC
"""
abstract type PointBC{T<:AbstractFloat} <: AbstractBC{T} end

"""
    D1{T<:AbstractFloat} <: PointBC{T}

First derivative (slope) boundary condition: S'(endpoint) = val

# Example
```julia
D1(0.5)  # Slope of 0.5 at endpoint
D1(0)    # Zero slope (horizontal tangent)
```
"""
struct D1{T<:AbstractFloat} <: PointBC{T}
    val::T
end
D1(v::Real) = D1{typeof(float(v))}(float(v))
D1{T}(bc::D1) where {T<:AbstractFloat} = D1{T}(T(bc.val))

"""
    D2{T<:AbstractFloat} <: PointBC{T}

Second derivative (curvature) boundary condition: S''(endpoint) = val

# Example
```julia
D2(0)    # Natural BC (zero curvature)
D2(1.5)  # Specified curvature at endpoint
```
"""
struct D2{T<:AbstractFloat} <: PointBC{T}
    val::T
end
D2(v::Real) = D2{typeof(float(v))}(float(v))
D2{T}(bc::D2) where {T<:AbstractFloat} = D2{T}(T(bc.val))

"""
    BCPair{T, L<:PointBC{T}, R<:PointBC{T}} <: AbstractBC{T}

Container for left and right boundary conditions with type parameters for zero-overhead dispatch.
The BC types are encoded in the type parameters, enabling compile-time specialization.

# Type Parameters
- `T`: Float type
- `L`: Left boundary condition type (D1{T} or D2{T})
- `R`: Right boundary condition type (D1{T} or D2{T})

# Example
```julia
bc = BCPair(D1(0.5), D2(0))  # Left: slope=0.5, Right: natural
```
"""
struct BCPair{T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}} <: AbstractBC{T}
    left::L
    right::R
end

# Convenience constructor from tuple
BCPair(t::Tuple{L, R}) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}} =
    BCPair{T,L,R}(t[1], t[2])

# Type aliases for common BC combinations
const NaturalBCPair{T} = BCPair{T, D2{T}, D2{T}}
const ClampedBCPair{T} = BCPair{T, D1{T}, D1{T}}

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
Equivalent to `BCPair(D2(0), D2(0))`.

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
Equivalent to `BCPair(D1(0), D1(0))`.

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
@inline _promote_pointbc(bc::D1, ::Type{T}) where {T<:AbstractFloat} = D1{T}(T(bc.val))
@inline _promote_pointbc(bc::D2, ::Type{T}) where {T<:AbstractFloat} = D2{T}(T(bc.val))


# ========================================
# BC Normalization
# ========================================

"""
    _normalize_bc(bc::AbstractBC, ::Type{T}) -> BCPair{T} | PeriodicBC{T}

Convert BC specification to normalized form for cache construction.

# Accepted Input Types
- `NaturalBC`: Natural BC (zero curvature) - default
- `ClampedBC`: Clamped BC (zero slope)
- `PeriodicBC`: Periodic boundary condition
- `BCPair`: Left/right BC pair (passed through)
- `PointBC` (D1/D2): Single BC applied symmetrically to both ends

# Returns
- `BCPair{T}`: For derivative BCs
- `PeriodicBC{T}`: For periodic BC
"""
# NaturalBC → BCPair(D2(0), D2(0))
@inline _normalize_bc(::NaturalBC, ::Type{T}) where {T<:AbstractFloat} =
    BCPair(D2(zero(T)), D2(zero(T)))

# ClampedBC → BCPair(D1(0), D1(0))
@inline _normalize_bc(::ClampedBC, ::Type{T}) where {T<:AbstractFloat} =
    BCPair(D1(zero(T)), D1(zero(T)))

# PeriodicBC passthrough with type promotion
@inline _normalize_bc(::PeriodicBC, ::Type{T}) where {T<:AbstractFloat} = PeriodicBC{T}()

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
@inline _is_periodic_bc(::PeriodicBC) = true
@inline _is_periodic_bc(::NaturalBC) = false
@inline _is_periodic_bc(::ClampedBC) = false
@inline _is_periodic_bc(::BCPair) = false
@inline _is_periodic_bc(::PointBC) = false
