# ========================================
# Boundary Condition Types
# ========================================
# Types and normalization for specifying endpoint boundary conditions.
# Used by cubic splines and can be extended to other interpolation methods.

"""
    AbstractBC{T<:AbstractFloat}

Abstract base type for endpoint boundary conditions in spline interpolation.
Subtypes define the derivative order and value at each endpoint.

# Subtypes
- `D1{T}`: First derivative (slope) BC
- `D2{T}`: Second derivative (curvature) BC
"""
abstract type AbstractBC{T<:AbstractFloat} end

"""
    D1{T<:AbstractFloat} <: AbstractBC{T}

First derivative (slope) boundary condition: S'(endpoint) = val

# Example
```julia
D1(0.5)  # Slope of 0.5 at endpoint
D1(0)    # Zero slope (horizontal tangent)
```
"""
struct D1{T<:AbstractFloat} <: AbstractBC{T}
    val::T
end
D1(v::Real) = D1{typeof(float(v))}(float(v))
D1{T}(bc::D1) where {T<:AbstractFloat} = D1{T}(T(bc.val))

"""
    D2{T<:AbstractFloat} <: AbstractBC{T}

Second derivative (curvature) boundary condition: S''(endpoint) = val

# Example
```julia
D2(0)    # Natural BC (zero curvature)
D2(1.5)  # Specified curvature at endpoint
```
"""
struct D2{T<:AbstractFloat} <: AbstractBC{T}
    val::T
end
D2(v::Real) = D2{typeof(float(v))}(float(v))
D2{T}(bc::D2) where {T<:AbstractFloat} = D2{T}(T(bc.val))

"""
    DerivativeBCData{T, L<:AbstractBC{T}, R<:AbstractBC{T}}

Container for left and right boundary conditions with type parameters for zero-overhead dispatch.
The BC types are encoded in the type parameters, enabling compile-time specialization.

# Type Parameters
- `T`: Float type
- `L`: Left boundary condition type (D1{T} or D2{T})
- `R`: Right boundary condition type (D1{T} or D2{T})

# Example
```julia
bc = DerivativeBCData(D1(0.5), D2(0))  # Left: slope=0.5, Right: natural
```
"""
struct DerivativeBCData{T<:AbstractFloat, L<:AbstractBC{T}, R<:AbstractBC{T}}
    left::L
    right::R
end

# Type aliases for common BC combinations
const NaturalBCData{T} = DerivativeBCData{T, D2{T}, D2{T}}
const ClampedBCData{T} = DerivativeBCData{T, D1{T}, D1{T}}

# ========================================
# BC Validation and Normalization
# ========================================

# Accept AbstractBC and Tuple{AbstractBC, AbstractBC} as valid BC specifications
@inline _validate_bc(::AbstractBC) = nothing
@inline _validate_bc(::Tuple{<:AbstractBC, <:AbstractBC}) = nothing

"""
    _normalize_bc(bc::Symbol, ::Type{T}) -> Tuple{AbstractBC, AbstractBC} | :periodic

Convert symbol BC specification to (left_bc, right_bc) tuple.

# Symbol Mapping
- `:natural` → `(D2(0), D2(0))` - zero second derivative at both ends
- `:clamped` → `(D1(0), D1(0))` - zero first derivative (flat) at both ends
- `:periodic` → returns `:periodic` (handled separately via Sherman-Morrison)
"""
@inline function _normalize_bc(bc::Symbol, ::Type{T}) where {T<:AbstractFloat}
    bc === :natural  && return (D2(zero(T)), D2(zero(T)))
    bc === :clamped  && return (D1(zero(T)), D1(zero(T)))
    bc === :periodic && return :periodic  # handled separately
    throw(ArgumentError("Unknown BC symbol: $bc"))
end

# Single D1/D2 → symmetric tuple (same BC at both ends)
@inline function _normalize_bc(bc::AbstractBC{T}, ::Type{T}) where {T<:AbstractFloat}
    return (bc, bc)
end

# Type promotion for mismatched BC type
@inline function _normalize_bc(bc::AbstractBC, ::Type{T}) where {T<:AbstractFloat}
    bc_t = bc isa D1 ? D1{T}(T(bc.val)) : D2{T}(T(bc.val))
    return (bc_t, bc_t)
end

# Tuple passthrough (already in normalized form) with type promotion
@inline function _normalize_bc(bc::Tuple{<:AbstractBC, <:AbstractBC}, ::Type{T}) where {T<:AbstractFloat}
    left, right = bc
    left_t = left isa D1 ? D1{T}(T(left.val)) : D2{T}(T(left.val))
    right_t = right isa D1 ? D1{T}(T(right.val)) : D2{T}(T(right.val))
    return (left_t, right_t)
end
