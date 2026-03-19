# ========================================
# Grid Spacing Types
# ========================================
# Abstract type hierarchy and concrete implementations for grid spacing storage.
# ScalarSpacing: O(1) memory for uniform grids (Range inputs)
# VectorSpacing: O(N) memory for non-uniform grids (Vector inputs)
#
# Include order: ops.jl → bc_types.jl → grid_spacing.jl → cubic_types.jl → ...

"""
    AbstractGridSpacing{T}

Abstract supertype for grid spacing storage strategies.

Concrete subtypes:
- `ScalarSpacing{T}`: Stores uniform spacing as scalars (O(1) memory)
- `VectorSpacing{T}`: Stores non-uniform spacing as vectors (O(N) memory)

The type parameter `T` is the floating-point type (Float32 or Float64).
"""
abstract type AbstractGridSpacing{T <: AbstractFloat} end

"""
    ScalarSpacing{T} <: AbstractGridSpacing{T}

Stores uniform grid spacing as scalar values. Used for `AbstractRange` inputs
where all intervals have identical spacing.

# Fields
- `h::T`: Uniform grid spacing (x[i+1] - x[i] for all i)
- `inv_h::T`: Precomputed reciprocal (1/h) for fast division elimination

# Memory
O(1) - just two scalar values regardless of grid size.

# Performance
Enables compiler constant propagation in kernel hot paths since `h` and `inv_h`
are scalars that can be inlined.

# Example
```julia
x = range(0.0, 1.0, 1001)  # 1000 intervals, uniform spacing
spacing = _create_spacing(x)  # ScalarSpacing{Float64}(0.001, 1000.0)
```
"""
struct ScalarSpacing{T <: AbstractFloat} <: AbstractGridSpacing{T}
    h::T
    inv_h::T
end

# Outer constructor for type promotion with mixed Real types
function ScalarSpacing(h::Real, inv_h::Real)
    T = promote_type(typeof(h), typeof(inv_h))
    T = T <: AbstractFloat ? T : Float64
    return ScalarSpacing{T}(T(h), T(inv_h))
end

"""
    VectorSpacing{T} <: AbstractGridSpacing{T}

Stores non-uniform grid spacing as vectors. Used for `AbstractVector` inputs
where intervals may have different spacings.

# Fields
- `h::Vector{T}`: Grid spacings h[i] = x[i+1] - x[i] (length n-1 for n grid points)
- `inv_h::Vector{T}`: Precomputed reciprocals inv_h[i] = 1/h[i]

# Memory
O(N) - stores 2*(n-1) floating-point values.

# Example
```julia
x = [0.0, 0.3, 0.7, 1.0]  # Non-uniform spacing
spacing = _create_spacing(x)  # VectorSpacing with h=[0.3, 0.4, 0.3]
```
"""
struct VectorSpacing{T <: AbstractFloat} <: AbstractGridSpacing{T}
    h::Vector{T}
    inv_h::Vector{T}
end

# _CachedRange is defined in cached_range.jl (included after this file).

# ========================================
# Accessor Functions
# ========================================

"""
    _get_h(spacing, i) -> T

Get the grid spacing for interval `i`.

For `ScalarSpacing`, returns the constant spacing (index ignored).
For `VectorSpacing`, returns `spacing.h[i]`.

Uses `@propagate_inbounds` to enable bounds-check elision in hot loops
when called within `@inbounds` blocks.
"""
@inline @Base.propagate_inbounds _get_h(s::ScalarSpacing, ::Int) = s.h
@inline @Base.propagate_inbounds _get_h(s::VectorSpacing, i::Int) = @inbounds s.h[i]

"""
    _get_inv_h(spacing, i) -> T

Get the precomputed reciprocal of grid spacing for interval `i`.

For `ScalarSpacing`, returns the constant reciprocal (index ignored).
For `VectorSpacing`, returns `spacing.inv_h[i]`.

Uses `@propagate_inbounds` to enable bounds-check elision in hot loops.
"""
@inline @Base.propagate_inbounds _get_inv_h(s::ScalarSpacing, ::Int) = s.inv_h
@inline @Base.propagate_inbounds _get_inv_h(s::VectorSpacing, i::Int) = @inbounds s.inv_h[i]

# ----------------------------------------
# 3-arg: grid-based accessors (no spacing object needed)
# ----------------------------------------
# For non-uniform grids, compute from the search result endpoints.
# _CachedRange overloads are in cached_range.jl (loaded after this file).
@inline _get_h(::AbstractVector, xR::Real, xL::Real) = xR - xL
@inline _get_inv_h(::AbstractVector, xR::Real, xL::Real) = inv(xR - xL)

# ========================================
# Factory Functions
# ========================================

"""
    _create_spacing(x::AbstractRange{T}) -> ScalarSpacing{T}

Create scalar spacing for uniform grids (Range inputs).

Extracts the constant step size and precomputes its reciprocal.
Defensive fallback for non-normalized Range inputs; the primary path
uses `_create_spacing(::_CachedRange)` in `cached_range.jl`.
"""
function _create_spacing(x::AbstractRange{T}) where {T <: AbstractFloat}
    # step(x) already returns T for AbstractRange{T}, avoid redundant conversion
    h = step(x)
    inv_h = inv(h)
    return ScalarSpacing{T}(h, inv_h)
end

# _create_spacing(::_CachedRange) is defined in cached_range.jl

# LinRange specialization removed — LinRange is always normalized to _CachedRange
# via _to_float() before reaching _create_spacing.  The AbstractRange fallback
# above handles any non-normalized Range if needed.

"""
    _create_spacing(x::AbstractVector{T}) -> VectorSpacing{T}

Create vector spacing for non-uniform grids (Vector inputs).

Computes h[i] = x[i+1] - x[i] and inv_h[i] = 1/h[i] for each interval.
"""
function _create_spacing(x::AbstractVector{T}) where {T <: AbstractFloat}
    n = length(x)
    h = Vector{T}(undef, n - 1)
    inv_h = Vector{T}(undef, n - 1)

    @inbounds for i in 1:(n - 1)
        h[i] = x[i + 1] - x[i]
        inv_h[i] = inv(h[i])
    end

    return VectorSpacing{T}(h, inv_h)
end
