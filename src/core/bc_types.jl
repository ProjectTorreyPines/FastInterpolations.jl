# ========================================
# Boundary Condition Types
# ========================================
# Types and normalization for specifying endpoint boundary conditions.
# Used by cubic splines and can be extended to other interpolation methods.
#
# Type Hierarchy:
#   AbstractBC{T}
#   ├── PointBC{T}           # Single-point BC (abstract)
#   │   ├── Deriv1{T}            # User-specified first derivative
#   │   ├── Deriv2{T}            # User-specified second derivative
#   │   ├── Deriv3{T}            # User-specified third derivative
#   │   └── PolyFit{D, T}        # D-degree polynomial fit (auto-estimated)
#   │       ├── LinearFit   = PolyFit{1}  (2 points, O(h))
#   │       ├── QuadraticFit = PolyFit{2}  (3 points, O(h²))
#   │       └── CubicFit    = PolyFit{3}  (4 points, O(h³))
#   ├── BCPair{T,L,R}        # Both endpoints
#   ├── PeriodicBC{T}        # Periodic BC
#   ├── NaturalBC{T}         # Natural BC (zero curvature at ends)
#   ├── ClampedBC{T}         # Clamped BC (zero slope at ends)
#   ├── MinCurvFit{T}        # Minimum curvature BC (quadratic splines)
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

# Default BC uses QuadraticFit
itp_default = quadratic_interp(x, y)

# MinCurvFit gives globally smooth result via curvature minimization
itp_smooth = quadratic_interp(x, y; bc=MinCurvFit())
```
"""
struct MinCurvFit{T<:AbstractFloat} <: AbstractBC{T} end
MinCurvFit() = MinCurvFit{Float64}()
MinCurvFit{T}(::MinCurvFit) where {T<:AbstractFloat} = MinCurvFit{T}()

# ========================================
# Polynomial Fit Boundary Conditions
# ========================================

"""
    PolyFit{D, T<:AbstractFloat} <: PointBC{T}

Generic polynomial fitting boundary condition.

Fits a degree-D polynomial through (D+1) points at the endpoint and evaluates
its derivative. This automatically estimates the first derivative from data
without requiring user-specified values.

# Type Parameters
- `D::Int`: Polynomial degree (1=linear, 2=quadratic, 3=cubic, ...)
- `T`: Floating point type

# Relationships
    Points needed = D + 1
    Accuracy = O(h^D) for smooth functions

# Aliases (Recommended for common cases)
- `LinearFit`   = `PolyFit{1}` → 2 points, O(h)
- `QuadraticFit` = `PolyFit{2}` → 3 points, O(h²)
- `CubicFit`    = `PolyFit{3}` → 4 points, O(h³)

# Mathematical Background
All polynomial fitting methods are mathematically equivalent to finite difference
formulas of the same order. For example, `CubicFit` (4-point) gives identical
coefficients to 4-point one-sided finite difference:
- Left:  `f'(x₁) ≈ (-11f₁ + 18f₂ - 9f₃ + 2f₄) / (6h)`
- Right: `f'(xₙ) ≈ (-2fₙ₋₃ + 9fₙ₋₂ - 18fₙ₋₁ + 11fₙ) / (6h)`

# Key Property
**Polynomial Reproduction**: `PolyFit{D}` exactly reproduces polynomials up to degree D.

# Example
```julia
# Recommended: use named aliases
itp = cubic_interp(x, y; bc=CubicFit())
itp = quadratic_interp(x, y; bc=Left(QuadraticFit()))

# Generic form (equivalent)
itp = cubic_interp(x, y; bc=PolyFit{3}())
```

See also: [`LinearFit`](@ref), [`QuadraticFit`](@ref), [`CubicFit`](@ref), [`Deriv1`](@ref)
"""
struct PolyFit{D, T<:AbstractFloat} <: PointBC{T}
    function PolyFit{D, T}() where {D, T<:AbstractFloat}
        D isa Int || throw(ArgumentError("Polynomial degree D must be an integer"))
        D >= 1 || throw(ArgumentError("Polynomial degree must be ≥ 1, got $D"))
        new{D, T}()
    end
end

# Convenience constructors
PolyFit{D}() where {D} = PolyFit{D, Float64}()
PolyFit{D, T}(::PolyFit{D}) where {D, T<:AbstractFloat} = PolyFit{D, T}()

# ----------------------------------------
# Type Aliases: Named convenience types
# ----------------------------------------

"""
    LinearFit = PolyFit{1}

2-point linear fit boundary condition. Estimates derivative using forward/backward
difference: `f'(x) ≈ (f₂ - f₁) / h`.

Accuracy: O(h) - first order.
Points needed: 2

# Example
```julia
itp = cubic_interp(x, y; bc=LinearFit())
itp32 = cubic_interp(Float32.(x), Float32.(y); bc=LinearFit{Float32}())
```
"""
const LinearFit{T<:AbstractFloat} = PolyFit{1, T}
LinearFit() = LinearFit{Float64}()

"""
    QuadraticFit = PolyFit{2}

3-point quadratic fit boundary condition. Fits a parabola through 3 points
and evaluates its derivative at the endpoint.

Accuracy: O(h²) - second order.
Points needed: 3

For uniform grids: `f'(x₁) ≈ (-3f₁ + 4f₂ - f₃) / (2h)`

# Key Property
**Polynomial Reproduction**: Exactly reproduces quadratic polynomials.

# Example
```julia
# Default BC for quadratic splines
itp = quadratic_interp(x, y; bc=Left(QuadraticFit()))
itp32 = quadratic_interp(Float32.(x), Float32.(y); bc=Left(QuadraticFit{Float32}()))
```
"""
const QuadraticFit{T<:AbstractFloat} = PolyFit{2, T}
QuadraticFit() = QuadraticFit{Float64}()

"""
    ParabolaFit

Deprecated alias for [`QuadraticFit`](@ref).
"""
function ParabolaFit(args...)
    Base.depwarn("ParabolaFit is deprecated and has been renamed to QuadraticFit; please use QuadraticFit instead.", :ParabolaFit)
    QuadraticFit(args...)
end


"""
    CubicFit = PolyFit{3}

4-point cubic fit boundary condition. Fits a cubic polynomial through 4 points
and evaluates its derivative at the endpoint.

Accuracy: O(h³) - third order.
Points needed: 4

For uniform grids:
- Left:  `f'(x₁) ≈ (-11f₁ + 18f₂ - 9f₃ + 2f₄) / (6h)`
- Right: `f'(xₙ) ≈ (-2fₙ₋₃ + 9fₙ₋₂ - 18fₙ₋₁ + 11fₙ) / (6h)`

# Key Property
**Polynomial Reproduction**: Exactly reproduces cubic polynomials.

# Example
```julia
# Recommended BC for cubic splines when derivative is unknown
itp = cubic_interp(x, y; bc=CubicFit())
itp32 = cubic_interp(Float32.(x), Float32.(y); bc=CubicFit{Float32}())

# Mixed with other BCs
itp = cubic_interp(x, y; bc=BCPair(CubicFit(), Deriv2(0)))
```
"""
const CubicFit{T<:AbstractFloat} = PolyFit{3, T}
CubicFit() = CubicFit{Float64}()


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
@inline _promote_pointbc(::PolyFit{D}, ::Type{T}) where {D, T<:AbstractFloat} = PolyFit{D, T}()


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
# Cubic BC Type Alias
# ========================================

"""
    CubicBC{T} = Union{PointBC{T}, BCPair{T,...}, PeriodicBC{T}}

Type alias for boundary conditions accepted by cubic spline interpolants.

Encompasses:
- `PointBC{T}`: Single-point BC (Deriv1, Deriv2, Deriv3) - promoted to BCPair internally
- `BCPair{T,L,R}`: Explicit left/right BC pair
- `PeriodicBC{T}`: Periodic boundary condition

This type is used as a constraint for the `bc` field in `CubicInterpolant`,
ensuring type safety while allowing all valid cubic spline BC types.

# Example
```julia
itp = cubic_interp(x, y; bc=NaturalBC())   # NaturalBC → BCPair stored
itp.bc  # BCPair{Float64, Deriv2{Float64}, Deriv2{Float64}}
```
"""
const CubicBC{T} = Union{PointBC{T}, BCPair{T,<:PointBC{T},<:PointBC{T}}, PeriodicBC{T}} where {T<:AbstractFloat}


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


# ========================================
# BC Materialization (PolyFit → Deriv1)
# ========================================

"""
    materialize_bc(bc::PolyFit{D}, xs, ys, endpoint::Val{:left/:right}) -> Deriv1{T}

Convert a polynomial-fit BC to a concrete `Deriv1` by estimating the derivative from data.

This "materializes" the lazy `PolyFit{D}` specification into an actual derivative value,
allowing all existing `Deriv1` code paths to work unchanged.

# Arguments
- `bc::PolyFit{D,T}`: Polynomial fit BC to materialize
- `xs::AbstractVector{T}`: Grid coordinates
- `ys::AbstractVector{T}`: Function values at grid points
- `endpoint::Val{:left}` or `Val{:right}`: Which endpoint to estimate

# Returns
- `Deriv1{T}(estimated_value)`: Concrete first derivative BC

# Example
```julia
bc = QuadraticFit()  # = PolyFit{2}
concrete_bc = materialize_bc(bc, xs, ys, Val(:left))  # → Deriv1{Float64}(computed_value)
```

See also: [`PolyFit`](@ref), [`_estimate_endpoint_derivative`](@ref)
"""
@inline function materialize_bc(
    ::PolyFit{D, T}, xs::AbstractVector{T}, ys::AbstractVector{T}, endpoint::Val
) where {D, T<:AbstractFloat}
    val = _estimate_endpoint_derivative(xs, ys, endpoint, PolyFit{D}())
    return Deriv1{T}(val)
end

# Passthrough for already-concrete BCs (no materialization needed)
@inline materialize_bc(bc::Deriv1, ::AbstractVector, ::AbstractVector, ::Val) = bc
@inline materialize_bc(bc::Deriv2, ::AbstractVector, ::AbstractVector, ::Val) = bc
@inline materialize_bc(bc::Deriv3, ::AbstractVector, ::AbstractVector, ::Val) = bc


# ========================================
# PolyFit{D} Point Validation (Generic)
# ========================================

"""
    get_polyfit_degree(bc) -> Int

Extract the maximum polynomial degree D from a boundary condition if it contains a `PolyFit{D}`.
Returns 0 for non-PolyFit BCs (no extra point requirement).

This function recursively unwraps container types (`Left`, `Right`, `BCPair`) to find
any inner `PolyFit{D}` type and returns the highest degree found.

# Examples
```julia
get_polyfit_degree(QuadraticFit())                    # → 2
get_polyfit_degree(Left(CubicFit()))                 # → 3
get_polyfit_degree(BCPair(QuadraticFit(), CubicFit())) # → 3 (max of 2 and 3)
get_polyfit_degree(Deriv1(0.0))                       # → 0 (no PolyFit)
```
"""
get_polyfit_degree(::PolyFit{D}) where {D} = D
get_polyfit_degree(::Left{<:AbstractFloat, <:PolyFit{D}}) where {D} = D
get_polyfit_degree(::Right{<:AbstractFloat, <:PolyFit{D}}) where {D} = D

# BCPair: return maximum degree from left and right
@inline function get_polyfit_degree(bc::BCPair)
    d_left = get_polyfit_degree(bc.left)
    d_right = get_polyfit_degree(bc.right)
    return max(d_left, d_right)
end

# Default: no PolyFit, no extra point requirement
get_polyfit_degree(::AbstractBC) = 0
get_polyfit_degree(::PointBC) = 0  # Deriv1, Deriv2, Deriv3


"""
    validate_polyfit_points(bc, n_points::Int)

Validate that the grid has enough points for `PolyFit{D}` boundary conditions.

`PolyFit{D}` requires at least `D+1` data points to estimate the endpoint derivative
using a degree-D polynomial fit. This function checks that requirement and throws
`ArgumentError` if insufficient points are provided.

# Arguments
- `bc`: Boundary condition (any type)
- `n_points`: Number of data points in the grid

# Throws
- `ArgumentError`: If `PolyFit{D}` is used but `n_points < D + 1`

# Examples
```julia
validate_polyfit_points(QuadraticFit(), 5)  # OK: 5 >= 3
validate_polyfit_points(CubicFit(), 3)     # ERROR: 3 < 4
validate_polyfit_points(Deriv1(0.0), 2)    # OK: no PolyFit requirement
```
"""
function validate_polyfit_points(bc, n_points::Int)
    D = get_polyfit_degree(bc)
    if D > 0
        min_points = D + 1
        n_points >= min_points || throw(ArgumentError(
            "PolyFit{$D} requires at least $min_points data points (got $n_points). " *
            "A degree-$D polynomial needs $(min_points) points to estimate endpoint derivatives."
        ))
    end
    return nothing
end
