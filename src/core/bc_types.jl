# ========================================
# Boundary Condition Types
# ========================================
# Types and normalization for specifying endpoint boundary conditions.
# Used by cubic splines and can be extended to other interpolation methods.
#
# DESIGN: Type-Free Abstract Types with Tv on Concrete Types Only
# ===============================================================
# Abstract types (AbstractBC, PointBC) have NO type parameters.
# Only concrete types holding values (Deriv1, Deriv2, Deriv3) have Tv parameter.
# This enables:
# - Clean lazy types: PolyFit{D} <: PointBC (no Nothing type parameter)
# - Complex value support: Deriv1{ComplexF64} still works
# - Cache efficiency via bc_structure() trait (unchanged)
#
# - Deriv1{ComplexF64}(1.0+2.0im) - Complex BC value
# - Deriv1{Float64}(0.5) - Real BC value
# - bc_structure(::Deriv1) = Val(:deriv1) - Same structure, same cache
#
# Type Hierarchy:
#   AbstractBC                    # No type parameter
#   ├── NoBC                      # Sentinel: "use the method's built-in endpoint rule"
#   ├── PointBC                   # No type parameter (abstract)
#   │   ├── Deriv1{Tv}            # User-specified first derivative (has Tv)
#   │   ├── Deriv2{Tv}            # User-specified second derivative (has Tv)
#   │   ├── Deriv3{Tv}            # User-specified third derivative (has Tv)
#   │   └── PolyFit{D}            # D-degree polynomial fit (lazy, just D)
#   │       ├── LinearFit   = PolyFit{1}  (2 points, O(h))
#   │       ├── QuadraticFit = PolyFit{2}  (3 points, O(h²))
#   │       └── CubicFit    = PolyFit{3}  (4 points, O(h³))
#   ├── BCPair{L,R}           # Both endpoints (L, R are PointBC subtypes)
#   ├── PeriodicBC{E,P}       # Periodic BC (inclusive/exclusive endpoint)
#   ├── ZeroCurvBC             # Zero-Curvature BC (singleton, normalized to Deriv2{Tv})
#   ├── ZeroSlopeBC             # Zero-Slope BC (singleton, normalized to Deriv1{Tv})
#   ├── MinCurvFit            # Minimum curvature BC (singleton, quadratic splines)
#   ├── Left{B}               # Endpoint wrapper: BC at left (x[1])
#   └── Right{B}              # Endpoint wrapper: BC at right (x[end])

"""
    AbstractBC

Abstract base type for all boundary condition specifications.
Type-free design: no type parameter on the abstract type.
Concrete subtypes with values (Deriv1, Deriv2, Deriv3) carry their own Tv parameter.

# Subtypes
- `NoBC`: Sentinel meaning "use the method's built-in endpoint rule" - singleton
- `ZeroCurvBC`: Zero-Curvature BC (zero curvature at both ends) - singleton
- `ZeroSlopeBC`: Zero-Slope BC (zero slope at both ends) - singleton
- `PeriodicBC{E,P}`: Periodic boundary condition (inclusive/exclusive endpoint)
- `PointBC`: Single-point derivative conditions (abstract)
- `BCPair{L,R}`: Pair of left/right boundary conditions
- `Left{B}`: Endpoint wrapper for BC at left (x[1]) - used by quadratic splines
- `Right{B}`: Endpoint wrapper for BC at right (x[end]) - used by quadratic splines
"""
abstract type AbstractBC end

"""
    NoBC <: AbstractBC

Sentinel boundary-condition value meaning "no BC requested; use the method's
built-in endpoint rule". Currently the default `bc` kwarg for **Constant and
Linear** interpolation (the only non-cubic methods wired for `bc` in this
release). PCHIP / Cardinal / Akima will adopt the same default when their
`bc` kwarg is added (planned next phase).

For supported methods, `bc=NoBC()` preserves existing behavior, and
`bc=PeriodicBC(...)` engages the periodic build path.

Cubic and Quadratic do not use `NoBC` because their coefficient systems are
under-determined without a concrete closure condition (they default to
`CubicFit()` and `Left(QuadraticFit())` respectively).

`_is_periodic_bc(::NoBC)` falls through the `AbstractBC` default and returns
`false`, so `NoBC` never triggers periodic dispatch.
"""
struct NoBC <: AbstractBC end

"""
    PointBC <: AbstractBC

Abstract type for single-point boundary conditions.
Represents a derivative condition at one endpoint.
Type-free design: concrete subtypes carry their own Tv parameter.

# Subtypes
- `Deriv1{Tv}`: First derivative (slope) BC
- `Deriv2{Tv}`: Second derivative (curvature) BC
- `Deriv3{Tv}`: Third derivative BC
- `PolyFit{D}`: Polynomial fit (lazy, no Tv until materialization)
"""
abstract type PointBC <: AbstractBC end

"""
    Deriv1{Tv} <: PointBC

First derivative (slope) boundary condition: S'(endpoint) = val

The type parameter `Tv` is the value type (unconstrained — supports any type with +, -, scalar *).

# Example
```julia
Deriv1(0.5)           # Slope of 0.5 at endpoint (Float64)
Deriv1(0)             # Zero slope (horizontal tangent)
Deriv1(1.0+2.0im)     # Complex slope (ComplexF64)
Deriv1(MyStruct(0.0))  # Custom type (duck typing)
```
"""
struct Deriv1{Tv} <: PointBC
    val::Tv
end
# Outer constructors: standard numerics auto-promote to float
# Custom types fall through to Julia's auto-generated Deriv1(v) = Deriv1{typeof(v)}(v)
Deriv1(v::Real) = Deriv1{typeof(float(v))}(float(v))
Deriv1(v::Complex{T}) where {T <: AbstractFloat} = Deriv1{Complex{T}}(v)
Deriv1{Tv}(bc::Deriv1) where {Tv} = Deriv1{Tv}(convert(Tv, bc.val))

"""
    Deriv2{Tv} <: PointBC

Second derivative (curvature) boundary condition: S''(endpoint) = val

The type parameter `Tv` is the value type (unconstrained — supports any type with +, -, scalar *).

# Example
```julia
Deriv2(0)             # Zero curvature (same as ZeroCurvBC)
Deriv2(1.5)           # Specified curvature at endpoint
Deriv2(0.0+0.0im)     # Complex curvature
```
"""
struct Deriv2{Tv} <: PointBC
    val::Tv
end
# Outer constructors: standard numerics auto-promote to float
# Custom types fall through to Julia's auto-generated Deriv2(v) = Deriv2{typeof(v)}(v)
Deriv2(v::Real) = Deriv2{typeof(float(v))}(float(v))
Deriv2(v::Complex{T}) where {T <: AbstractFloat} = Deriv2{Complex{T}}(v)
Deriv2{Tv}(bc::Deriv2) where {Tv} = Deriv2{Tv}(convert(Tv, bc.val))

"""
    Deriv3{Tv} <: PointBC

Third derivative boundary condition: S'''(endpoint) = val

For cubic splines, the third derivative is constant within each interval:
S'''(x) = (z[i+1] - z[i]) / h[i]. This BC specifies the third derivative
value at the first (or last) interval.

The type parameter `Tv` is the value type (unconstrained — supports any type with +, -, scalar *).

# Example
```julia
Deriv3(0)             # Zero third derivative at endpoint
Deriv3(1.0)           # Specified third derivative
Deriv3(0.0+0.0im)     # Complex third derivative
```
"""
struct Deriv3{Tv} <: PointBC
    val::Tv
end
# Outer constructors: standard numerics auto-promote to float
# Custom types fall through to Julia's auto-generated Deriv3(v) = Deriv3{typeof(v)}(v)
Deriv3(v::Real) = Deriv3{typeof(float(v))}(float(v))
Deriv3(v::Complex{T}) where {T <: AbstractFloat} = Deriv3{Complex{T}}(v)
Deriv3{Tv}(bc::Deriv3) where {Tv} = Deriv3{Tv}(convert(Tv, bc.val))

"""
    BCPair{L<:PointBC, R<:PointBC} <: AbstractBC

Container for left and right boundary conditions with type parameters for zero-overhead dispatch.
The BC types are encoded in the type parameters, enabling compile-time specialization.

Type-free design: no Tv parameter on BCPair itself. The value type is determined by
the concrete L and R types (e.g., `BCPair{Deriv1{Float64}, Deriv2{Float64}}`).

# Type Parameters
- `L`: Left boundary condition type (Deriv1{Tv}, Deriv2{Tv}, PolyFit{D}, etc.)
- `R`: Right boundary condition type (Deriv1{Tv}, Deriv2{Tv}, PolyFit{D}, etc.)

# Example
```julia
bc = BCPair(Deriv1(0.5), Deriv2(0))           # Left: slope=0.5, Right: zero curvature
bc = BCPair(Deriv1(1.0+0.0im), Deriv2(0.0im)) # Complex BC
bc = BCPair(CubicFit(), Deriv2(0.0))          # Mixed: lazy left, concrete right
```
"""
struct BCPair{L <: PointBC, R <: PointBC} <: AbstractBC
    left::L
    right::R
end

# Convenience constructor from tuple
BCPair(t::Tuple{L, R}) where {L <: PointBC, R <: PointBC} =
    BCPair{L, R}(t[1], t[2])


"""
    PeriodicBC{E, P, C} <: AbstractBC

Periodic boundary condition: S(x_0) = S(x_n), S'(x_0) = S'(x_n), S''(x_0) = S''(x_n)

Internally, periodic BC uses Sherman-Morrison solver with `PeriodicData{T}` for the cache.

# Type Parameters
- `E::Symbol`: `:inclusive` or `:exclusive` (compile-time endpoint convention)
- `P`: `Nothing` (inclusive or auto-infer) or `<:AbstractFloat` (explicit period)
- `C::Bool`: whether to validate `y[1] ≈ y[end]` at construction time (default `true`)

# Endpoint Conventions
- **Inclusive** (`endpoint=:inclusive`, default): `y[1] ≈ y[end]` required (standard convention)
- **Exclusive** (`endpoint=:exclusive`): `y[end]` is the last unique data point; the period boundary
  is handled internally. For `AbstractRange` grids, the period is inferred from `step(x) * length(x)`.
  For non-uniform grids, `period` must be provided explicitly.

# Examples
```julia
# Inclusive (default, backward compatible)
cache = CubicSplineCache(x; bc=PeriodicBC())

# Exclusive with explicit period
itp = cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive, period=2π))

# Exclusive with Range grid (period auto-inferred)
x = range(0, step=2π/64, length=64)
itp = cubic_interp(x, sin.(x); bc=PeriodicBC(endpoint=:exclusive))

# Skip endpoint check (e.g., scaled data where noise > 8eps)
itp = cubic_interp(x, 1e6 .* sin.(x); bc=PeriodicBC(check=false))
```
"""
struct PeriodicBC{E, P, C} <: AbstractBC
    period::P         # Nothing or AbstractFloat
    function PeriodicBC{E, P, C}(period::P) where {E, P, C}
        E isa Symbol || error("PeriodicBC type parameter E must be a Symbol")
        E in (:inclusive, :exclusive) || error("PeriodicBC type parameter E must be :inclusive or :exclusive")
        C isa Bool || error("PeriodicBC type parameter C must be a Bool")
        return new{E, P, C}(period)
    end
end

# Accessors (from type parameters, zero-cost)
@inline endpoint(::PeriodicBC{E}) where {E} = E
@inline periodic_check(::PeriodicBC{E, P, C}) where {E, P, C} = C

# Keyword constructor with validation (also serves as zero-arg constructor via defaults)
function PeriodicBC(; endpoint::Symbol = :inclusive, period::Union{Real, Nothing} = nothing, check::Bool = true)
    endpoint in (:inclusive, :exclusive) || throw(
        ArgumentError(
            "endpoint must be :inclusive or :exclusive, got :$endpoint"
        )
    )
    if endpoint == :inclusive
        period === nothing || throw(
            ArgumentError(
                "period is not applicable for endpoint=:inclusive (y[1]≈y[end] convention)"
            )
        )
        return PeriodicBC{:inclusive, Nothing, check}(nothing)
    else # :exclusive
        if period !== nothing
            p = float(period)
            p > 0 || throw(ArgumentError("period must be positive, got $period"))
            return PeriodicBC{:exclusive, typeof(p), check}(p)
        else
            return PeriodicBC{:exclusive, Nothing, check}(nothing)  # infer from Range at build time
        end
    end
end

"""
    ZeroCurvBC <: AbstractBC

Zero-curvature boundary condition: S''(endpoints) = 0 (zero curvature at both ends).
Equivalent to `BCPair(Deriv2(0), Deriv2(0))`.

This is a singleton type (structure-only). Normalized to `BCPair` with
`Deriv2(zero(Tv))` at both ends based on the value type during interpolant construction.

# Example
```julia
itp = cubic_interp(x, y; bc=ZeroCurvBC())  # Explicit zero-curvature BC
```
"""
struct ZeroCurvBC <: AbstractBC end

"""
    ZeroSlopeBC <: AbstractBC

Zero-slope boundary condition: S'(endpoints) = 0 (zero slope at both ends).
Equivalent to `BCPair(Deriv1(0), Deriv1(0))`.

This is a singleton type (structure-only). Normalized to `BCPair` with
`Deriv1(zero(Tv))` at both ends based on the value type during interpolant construction.

# Example
```julia
itp = cubic_interp(x, y; bc=ZeroSlopeBC())
```
"""
struct ZeroSlopeBC <: AbstractBC end

"""
    MinCurvFit <: AbstractBC

Minimum curvature boundary condition for quadratic splines.
Minimizes total curvature (∫S''² dx) by optimizing the initial slope d[1].

This is a singleton type (structure-only). The optimization is performed
during interpolant construction based on the actual data values.

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
struct MinCurvFit <: AbstractBC end

# ========================================
# Polynomial Fit Boundary Conditions
# ========================================

"""
    PolyFit{D} <: PointBC

Generic polynomial fitting boundary condition (lazy).

Fits a degree-D polynomial through (D+1) points at the endpoint and evaluates
its derivative. This automatically estimates the first derivative from data
without requiring user-specified values.

This is a **lazy** type with no value parameter. The actual derivative value
is computed during `materialize_bc()` based on the data's value type (Tv).

# Type Parameters
- `D::Int`: Polynomial degree (1=linear, 2=quadratic, 3=cubic, ...)

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
struct PolyFit{D} <: PointBC
    function PolyFit{D}() where {D}
        D isa Int || throw(ArgumentError("Polynomial degree D must be an integer"))
        D >= 1 || throw(ArgumentError("Polynomial degree must be ≥ 1, got $D"))
        return new{D}()
    end
end


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
```
"""
const LinearFit = PolyFit{1}

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
```
"""
const QuadraticFit = PolyFit{2}


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

# Mixed with other BCs
itp = cubic_interp(x, y; bc=BCPair(CubicFit(), Deriv2(0)))
```
"""
const CubicFit = PolyFit{3}


# ========================================
# BC Structure Trait (for Cache Keys)
# ========================================
# Thomas factorization depends only on BC **structure**, not on BC **values**.
# This trait enables cache sharing between Float64 and ComplexF64 interpolants
# with the same BC structure.

"""
    bc_structure(bc) -> Val{Symbol}

Return the structural identity of a BC for cache key purposes.
Different Tv types with the same structure share the same cache.

# Example
```julia
bc_structure(Deriv1{Float64}(0.5))       # → Val(:deriv1)
bc_structure(Deriv1{ComplexF64}(1.0im))  # → Val(:deriv1)  # Same!
bc_structure(Deriv2{Float64}(0.0))       # → Val(:deriv2)
bc_structure(PolyFit{2}())               # → Val(:polyfit_2)
```
"""
@inline bc_structure(::Deriv1) = Val(:deriv1)
@inline bc_structure(::Deriv2) = Val(:deriv2)
@inline bc_structure(::Deriv3) = Val(:deriv3)
@inline bc_structure(::PolyFit{D}) where {D} = Val(Symbol(:polyfit_, D))
@inline bc_structure(::ZeroCurvBC) = Val(:zero_curv)
@inline bc_structure(::ZeroSlopeBC) = Val(:zero_slope)
@inline bc_structure(::PeriodicBC) = Val(:periodic)
@inline bc_structure(::MinCurvFit) = Val(:mincurvfit)
@inline bc_structure(bc::BCPair) = (bc_structure(bc.left), bc_structure(bc.right))
# Note: bc_structure for Left/Right is defined after those types (see below)


# ========================================
# Type Promotion Helpers
# ========================================
# Promote BC values to target value type Tv.

"""
    _promote_pointbc(bc::PointBC, ::Type{Tv}) -> PointBC

Promote a PointBC to a specific value type Tv.
For concrete types (Deriv1, Deriv2, Deriv3), returns Deriv*{Tv} with converted value.
For lazy types (PolyFit), returns the same type unchanged.
Extensible: add methods for new PointBC subtypes.
"""
@inline _promote_pointbc(bc::Deriv1, ::Type{Tv}) where {Tv} = Deriv1{Tv}(convert(Tv, bc.val))
@inline _promote_pointbc(bc::Deriv2, ::Type{Tv}) where {Tv} = Deriv2{Tv}(convert(Tv, bc.val))
@inline _promote_pointbc(bc::Deriv3, ::Type{Tv}) where {Tv} = Deriv3{Tv}(convert(Tv, bc.val))
@inline _promote_pointbc(::PolyFit{D}, ::Type{Tv}) where {D, Tv} = PolyFit{D}()  # Lazy, no value


# ========================================
# BC Normalization
# ========================================

"""
    _normalize_bc(bc::AbstractBC) -> BCPair or Left/Right/MinCurvFit

Pure shape normalization: convert BC specification to solver-compatible form.
No type promotion — Julia's arithmetic auto-promotes BC values in the solver RHS.

Note: PeriodicBC is handled separately via `_is_periodic_bc()` check before
`_normalize_bc` is called. This function only handles derivative BCs.

# Accepted Input Types
- `ZeroCurvBC()` → `BCPair(Deriv2(0.0), Deriv2(0.0))`
- `ZeroSlopeBC()` → `BCPair(Deriv1(0.0), Deriv1(0.0))`
- `BCPair` → returned as-is
- `PointBC` (Deriv1/Deriv2) → `BCPair(bc, bc)` (symmetric)
- `Left`/`Right`/`MinCurvFit` → returned as-is (quadratic BC wrappers)
"""
@inline _normalize_bc(::ZeroCurvBC) = BCPair(Deriv2(0.0), Deriv2(0.0))
@inline _normalize_bc(::ZeroSlopeBC) = BCPair(Deriv1(0.0), Deriv1(0.0))
# Type overloads: when a Type is passed, use zero(Tv) for standard numerics.
@inline _normalize_bc(::ZeroCurvBC, ::Type{Tv}) where {Tv} = BCPair(Deriv2(zero(Tv)), Deriv2(zero(Tv)))
@inline _normalize_bc(::ZeroSlopeBC, ::Type{Tv}) where {Tv} = BCPair(Deriv1(zero(Tv)), Deriv1(zero(Tv)))
# Duck-safe overloads: use 0 * sample to produce the correct zero type.
# Needed when data type (e.g. MyDuck) doesn't support convert(MyDuck, Float64).
@inline _normalize_bc(::ZeroCurvBC, sample) = (z = 0 * sample; BCPair(Deriv2(z), Deriv2(z)))
@inline _normalize_bc(::ZeroSlopeBC, sample) = (z = 0 * sample; BCPair(Deriv1(z), Deriv1(z)))
@inline _normalize_bc(bc::BCPair) = bc
@inline _normalize_bc(bc::PointBC) = BCPair(bc, bc)
# Fallback: ignore second arg for all other BC types (only ZeroCurv/ZeroSlope need it)
@inline _normalize_bc(bc::AbstractBC, _sample) = _normalize_bc(bc)

# NOTE: _normalize_bc methods for Left/Right/MinCurvFit are defined after Left/Right structs below.


# ========================================
# Cache-Compatible BC Conversion
# ========================================

"""
    _cache_pointbc(bc::PointBC, ::Type{T}) -> PointBC

Convert a PointBC to cache-compatible structural form with grid type T.

The cubic cache (LU factorization) depends only on BC *structure* (Deriv1 vs Deriv2
vs Deriv3), not BC *values*. The matrix rows are set by `_set_first_row!`/`_set_last_row!`
which dispatch on type, never reading `.val`. BC values are used only during RHS
computation (`_solve_system!`), which receives the original BC from the caller.

All BC types → Deriv{N}{T}(zero(T)): preserves the structural type for bank
selection while discarding values that are irrelevant to the cache key.
PolyFit → Deriv1{T}(zero(T)): same matrix structure as Deriv1.
"""
@inline _cache_pointbc(::Deriv1, ::Type{T}) where {T} = Deriv1{T}(zero(T))
@inline _cache_pointbc(::Deriv2, ::Type{T}) where {T} = Deriv2{T}(zero(T))
@inline _cache_pointbc(::Deriv3, ::Type{T}) where {T} = Deriv3{T}(zero(T))
# PolyFit → Deriv1 (same matrix structure, zero value for cache key)
@inline _cache_pointbc(::PolyFit, ::Type{T}) where {T} = Deriv1{T}(zero(T))

"""
    _cache_bc_pair(bc_pair::BCPair, ::Type{T}) -> BCPair{L, R}

Convert a BCPair to cache-compatible form with value type T.

For BCPair containing lazy types (PolyFit), converts to concrete types
suitable for cache lookup. PolyFit → Deriv1 (same matrix structure).

# Arguments
- `bc_pair`: BCPair (may contain lazy types like PolyFit)
- `T`: Target value type (Float64, Float32, etc.)

# Returns
BCPair{L, R} suitable for cache key matching.
"""
@inline function _cache_bc_pair(bc::BCPair, ::Type{T}) where {T}
    left = _cache_pointbc(bc.left, T)
    right = _cache_pointbc(bc.right, T)
    return BCPair(left, right)
end


# ========================================
# BC Array Normalization (for SeriesInterpolant)
# ========================================

"""
    _normalize_bc_array(bcs, Tv, n_series) -> Vector{<:BCPair}

Normalize an array of BCs to BCPair for per-series boundary conditions.

# Arguments
- `bcs`: Array of AbstractBC (length must equal n_series)
- `Tv`: Target value type (unconstrained)
- `n_series`: Expected number of series

# Returns
- If input is already `AbstractVector{<:BCPair}`: returns input unchanged (zero allocation)
- Otherwise: `Vector` of normalized BCPair boundary conditions

# Throws
- `DimensionMismatch`: if length(bcs) != n_series
- `ArgumentError`: if any BC is PeriodicBC (not supported in arrays)
"""
function _normalize_bc_array end

# Fast path: already BCPair - zero allocation, inline away
@inline function _normalize_bc_array(
        bcs::AbstractVector{<:BCPair},
        ::Type{Tv},
        n_series::Int
    ) where {Tv}
    length(bcs) == n_series || throw(
        DimensionMismatch(
            "BC array length $(length(bcs)) does not match n_series $n_series"
        )
    )
    return bcs  # No conversion needed
end

# General path: needs normalization to BCPair
function _normalize_bc_array(
        bcs::AbstractVector{<:AbstractBC},
        ::Type{Tv},
        n_series::Int
    ) where {Tv}
    length(bcs) == n_series || throw(
        DimensionMismatch(
            "BC array length $(length(bcs)) does not match n_series $n_series"
        )
    )

    # Check for PeriodicBC (not supported in arrays)
    for (i, bc) in enumerate(bcs)
        _is_periodic_bc(bc) && _throw_periodic_in_bc_array(i)
    end

    return [_normalize_bc(bc) for bc in bcs]
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

# ── @noinline throw helpers (keep cold error paths out of hot code) ──

@noinline function _throw_periodic_in_bc_array(i::Int)
    throw(
        ArgumentError(
            "PeriodicBC at index $i is not supported in BC arrays. " *
                "Use uniform PeriodicBC for all series instead."
        )
    )
end

@noinline function _throw_periodic_unsupported(feature::String)
    throw(
        ArgumentError(
            "PeriodicBC is not yet supported by $feature. " *
                "Use BCPair-based boundary conditions (CubicFit, ZeroCurvBC, Deriv1, etc.)."
        )
    )
end

@noinline function _throw_periodic_extrap_mismatch(d::Int, extrap)
    throw(
        ArgumentError(
            "Periodic BC on dim $d only supports NoExtrap() or WrapExtrap(), got $(typeof(extrap))()"
        )
    )
end

# ========================================
# Cubic BC Type Alias
# ========================================

"""
    CubicBC = Union{PointBC, BCPair, PeriodicBC}

Type alias for boundary conditions accepted by cubic spline interpolants.

With the Type-Free design, this is a simple union without type parameters:
- `PointBC`: Single-point BC (Deriv1{Tv}, Deriv2{Tv}, Deriv3{Tv}, PolyFit{D}) - promoted to BCPair internally
- `BCPair{L,R}`: Left/right BC pair (L, R are PointBC subtypes)
- `PeriodicBC{E,P}`: Periodic boundary condition (inclusive or exclusive endpoint)

This type is used as a constraint for the `bc` field in `CubicInterpolant`,
ensuring type safety while allowing all valid cubic spline BC types.

# Example
```julia
itp = cubic_interp(x, y; bc=ZeroCurvBC())   # ZeroCurvBC → BCPair stored
itp.bc  # BCPair{Deriv2{Float64}, Deriv2{Float64}}

itp = cubic_interp(x, y; bc=CubicFit())    # CubicFit → BCPair stored
itp.bc  # BCPair{PolyFit{3}, PolyFit{3}}
```
"""
const CubicBC = Union{PointBC, BCPair, PeriodicBC}


# ========================================
# Endpoint-Specific BC Wrappers (Quadratic)
# ========================================

"""
    Left{B<:PointBC} <: AbstractBC

Wrapper indicating BC is applied at left endpoint (x[1]).
Used for quadratic splines where only one endpoint BC is specified.

The type parameter `B` is the inner PointBC type, which carries the value type Tv
for concrete types (Deriv1{Tv}, etc.) or just the degree D for lazy types (PolyFit{D}).

# Example
```julia
bc = Left(Deriv1(0.5))        # slope = 0.5 at left endpoint
bc = Left(Deriv2(0.0))        # curvature = 0 at left endpoint
bc = Left(Deriv1(1.0+2.0im))  # Complex slope
bc = Left(QuadraticFit())     # Lazy polynomial fit
```
"""
struct Left{B <: PointBC} <: AbstractBC
    bc::B
end


"""
    Right{B<:PointBC} <: AbstractBC

Wrapper indicating BC is applied at right endpoint (x[end]).
Used for quadratic splines where only one endpoint BC is specified.

The type parameter `B` is the inner PointBC type, which carries the value type Tv
for concrete types (Deriv1{Tv}, etc.) or just the degree D for lazy types (PolyFit{D}).

# Example
```julia
bc = Right(Deriv1(2.0))        # slope = 2.0 at right endpoint
bc = Right(Deriv2(0.0))        # curvature = 0 at right endpoint
bc = Right(Deriv1(1.0+2.0im))  # Complex slope
bc = Right(QuadraticFit())     # Lazy polynomial fit
```
"""
struct Right{B <: PointBC} <: AbstractBC
    bc::B
end

# bc_structure for Left/Right (must be after type definitions)
@inline bc_structure(bc::Left) = bc_structure(bc.bc)
@inline bc_structure(bc::Right) = bc_structure(bc.bc)

# Quadratic BC normalization (shape only — no type promotion)
@inline _normalize_bc(bc::Left) = bc
@inline _normalize_bc(bc::Right) = bc
@inline _normalize_bc(::MinCurvFit) = MinCurvFit()


# ========================================
# BC Materialization (PolyFit → Deriv1)
# ========================================

"""
    materialize_bc(bc::PolyFit{D}, xs, ys, endpoint::LeftSide/RightSide) -> Deriv1{Tv}

Convert a polynomial-fit BC to a concrete `Deriv1` by estimating the derivative from data.

This "materializes" the lazy `PolyFit{D}` specification into an actual derivative value,
allowing all existing `Deriv1` code paths to work unchanged.

# Arguments
- `bc::PolyFit{D}`: Polynomial fit BC to materialize (lazy, no value type)
- `xs::AbstractVector{Tg}`: Grid coordinates (real-valued)
- `ys::AbstractVector{Tv}`: Function values at grid points (can be Complex)
- `endpoint::LeftSide` or `::RightSide`: Which endpoint to estimate

# Returns
- `Deriv1{Tv}(estimated_value)`: Concrete first derivative BC with value type from ys

# Example
```julia
bc = QuadraticFit()  # = PolyFit{2}
concrete_bc = materialize_bc(bc, xs, ys, LeftSide())  # → Deriv1{Float64}(computed_value)

# Complex values
y_complex = [1.0+2.0im, 3.0+4.0im, ...]
concrete_bc = materialize_bc(bc, xs, y_complex, LeftSide())  # → Deriv1{ComplexF64}(...)
```

See also: [`PolyFit`](@ref), [`_estimate_endpoint_derivative`](@ref)
"""
@inline function materialize_bc(
        ::PolyFit{D}, xs::AbstractVector{Tg}, ys::AbstractVector{Tv}, endpoint::AbstractSide
    ) where {D, Tg, Tv}
    val = _estimate_endpoint_derivative(xs, ys, endpoint, PolyFit{D}())
    return Deriv1(val)
end

# Passthrough for already-concrete BCs (no materialization needed)
@inline materialize_bc(bc::Deriv1, ::AbstractVector, ::AbstractVector, ::AbstractSide) = bc
@inline materialize_bc(bc::Deriv2, ::AbstractVector, ::AbstractVector, ::AbstractSide) = bc
@inline materialize_bc(bc::Deriv3, ::AbstractVector, ::AbstractVector, ::AbstractSide) = bc


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
get_polyfit_degree(::Left{<:PolyFit{D}}) where {D} = D
get_polyfit_degree(::Right{<:PolyFit{D}}) where {D} = D

# BCPair: return maximum degree from left and right
@inline function get_polyfit_degree(bc::BCPair)
    d_left = get_polyfit_degree(bc.left)
    d_right = get_polyfit_degree(bc.right)
    return max(d_left, d_right)
end

# Default: no PolyFit, no extra point requirement
get_polyfit_degree(::AbstractBC) = 0


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
        n_points >= min_points || throw(
            ArgumentError(
                "PolyFit{$D} requires at least $min_points data points (got $n_points). " *
                    "A degree-$D polynomial needs $(min_points) points to estimate endpoint derivatives."
            )
        )
    end
    return nothing
end
