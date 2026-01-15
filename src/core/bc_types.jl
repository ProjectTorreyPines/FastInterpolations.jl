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
#   │   ├── Deriv2{T}            # Second derivative
#   │   ├── Deriv3{T}            # Third derivative
#   │   └── ParabolaFit{T}       # 3-point parabola fit (quadratic splines)
#   ├── BCPair{T,L,R}        # Both endpoints
#   ├── PeriodicBC{T}        # Periodic BC
#   ├── NaturalBC{T}         # Natural BC (zero curvature at ends)
#   ├── ClampedBC{T}         # Clamped BC (zero slope at ends)
#   ├── MinCurvFit{T}         # Minimum curvature BC (quadratic splines)
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
    Deriv3{T<:AbstractFloat} <: PointBC{T}

Third derivative boundary condition: S'''(endpoint) = val

For cubic splines, the third derivative is constant within each interval:
S'''(x) = (z[i+1] - z[i]) / h[i]. This BC specifies the third derivative
value at the first (or last) interval.

# Example
```julia
Deriv3(0)    # Zero third derivative at endpoint
Deriv3(1.0)  # Specified third derivative
```
"""
struct Deriv3{T<:AbstractFloat} <: PointBC{T}
    val::T
end
Deriv3(v::Real) = Deriv3{typeof(float(v))}(float(v))
Deriv3{T}(bc::Deriv3) where {T<:AbstractFloat} = Deriv3{T}(T(bc.val))

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

"""
    MinCurvFit{T<:AbstractFloat} <: AbstractBC{T}

Minimum curvature boundary condition for quadratic splines.
Minimizes total curvature (∫S''² dx) by optimizing the initial slope d[1].

This produces globally smooth interpolation by finding the optimal d[1] that
minimizes the integrated squared second derivative. The optimal d[1] is computed
using a closed-form solution in O(n) time with no additional allocation.

# Mathematical Background
For quadratic splines, the slope d[i] depends on d[1] via recurrence:
- d[i] = α[i] * d[1] + β[i]  where α[i] = (-1)^(i+1)

The optimal d[1] minimizes:
- ∫(S'')² dx = Σ 4*a[i]²*h[i] = Σ (s[i] - d[i])²/h[i]

Closed-form solution:
- d[1] = [Σ α[i]*(s[i] - β[i])/h[i]] / [Σ 1/h[i]]

# Example
```julia
x = [0.0, 0.3, 0.8, 1.5, 2.5, 3.0, 4.0]
y = [0.0, 0.8, 1.2, 0.9, 0.3, 0.6, 1.0]

# Default BC uses ParabolaFit
itp_default = quadratic_interp(x, y)

# MinCurvFit gives globally smooth result via curvature minimization
itp_smooth = quadratic_interp(x, y; bc=MinCurvFit())
```
"""
struct MinCurvFit{T<:AbstractFloat} <: AbstractBC{T} end
MinCurvFit() = MinCurvFit{Float64}()
MinCurvFit{T}(::MinCurvFit) where {T<:AbstractFloat} = MinCurvFit{T}()

"""
    ParabolaFit{T<:AbstractFloat} <: PointBC{T}

Parabola-fit boundary condition for quadratic splines.
Computes the initial slope d[1] (or d[n]) using a 3-point derivative formula
that exactly reproduces any polynomial up to degree 2.

This is the recommended BC for quadratic splines when the underlying function
is polynomial-like or smooth. It uses the first (or last) 3 points to fit a
parabola and computes the derivative at the endpoint.

# Mathematical Background
For the first 3 points (x₀, y₀), (x₁, y₁), (x₂, y₂), the 3-point derivative formula is:
- d[1] = y₀·(−(2h₁+h₂))/(h₁(h₁+h₂)) + y₁·(h₁+h₂)/(h₁·h₂) − y₂·h₁/((h₁+h₂)·h₂)

where h₁ = x₁ - x₀ and h₂ = x₂ - x₁.

For uniform grids (h₁ = h₂ = h), this simplifies to:
- d[1] = (−3y₀ + 4y₁ − y₂) / (2h)

# Key Property
**Polynomial Reproduction**: For any quadratic polynomial f(x) = ax² + bx + c,
ParabolaFit BC produces exact interpolation at all query points.

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = x.^2  # f(x) = x²

# ParabolaFit gives exact reproduction
itp = quadratic_interp(x, y; bc=Left(ParabolaFit()))
itp(1.5)  # ≈ 2.25 (exact)

# Works at both endpoints
itp_right = quadratic_interp(x, y; bc=Right(ParabolaFit()))
```
"""
struct ParabolaFit{T<:AbstractFloat} <: PointBC{T} end
ParabolaFit() = ParabolaFit{Float64}()
ParabolaFit{T}(::ParabolaFit) where {T<:AbstractFloat} = ParabolaFit{T}()


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
@inline _promote_pointbc(bc::Deriv3, ::Type{T}) where {T<:AbstractFloat} = Deriv3{T}(T(bc.val))
@inline _promote_pointbc(::ParabolaFit, ::Type{T}) where {T<:AbstractFloat} = ParabolaFit{T}()


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
# BC Array Normalization (for SeriesInterpolant)
# ========================================

"""
    _normalize_bc_array(bcs, T, n_series) -> AbstractVector{<:BCPair{T}}

Normalize an array of BCs to BCPair for per-series boundary conditions.

# Arguments
- `bcs`: Array of AbstractBC (length must equal n_series)
- `T`: Target float type
- `n_series`: Expected number of series

# Returns
- If input is already `AbstractVector{<:BCPair{T}}`: returns input unchanged (zero allocation)
- Otherwise: `Vector{BCPair{T}}` of normalized boundary conditions

# Throws
- `DimensionMismatch`: if length(bcs) != n_series
- `ArgumentError`: if any BC is PeriodicBC (not supported in arrays)
"""
function _normalize_bc_array end

# Fast path: already BCPair{T} - zero allocation, inline away
@inline function _normalize_bc_array(
    bcs::AbstractVector{<:BCPair{T}},
    ::Type{T},
    n_series::Int
) where {T<:AbstractFloat}
    length(bcs) == n_series || throw(DimensionMismatch(
        "BC array length $(length(bcs)) does not match n_series $n_series"
    ))
    return bcs  # No conversion needed
end

# General path: needs normalization to BCPair{T}
function _normalize_bc_array(
    bcs::AbstractVector{<:AbstractBC},
    ::Type{T},
    n_series::Int
) where {T<:AbstractFloat}
    length(bcs) == n_series || throw(DimensionMismatch(
        "BC array length $(length(bcs)) does not match n_series $n_series"
    ))

    # Check for PeriodicBC (not supported in arrays)
    for (i, bc) in enumerate(bcs)
        if _is_periodic_bc(bc)
            throw(ArgumentError(
                "PeriodicBC at index $i is not supported in BC arrays. " *
                "Use uniform PeriodicBC for all series instead."
            ))
        end
    end

    # Create new Vector{BCPair{T}} with normalized BCs
    return BCPair{T}[_normalize_bc(bc, T) for bc in bcs]
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
