# ========================================
# Per-Axis Interpolation Method Types
# ========================================
# Lightweight types specifying which 1D interpolation method to use per axis.
# Used by HeteroInterpolantND for heterogeneous per-axis interpolation.
#
# Design: Method-specific options (BC, side) are stored as type parameters
# for zero-cost dispatch — no allocation, no type instability.
#
# Include order: after bc_types.jl (needs AbstractBC, AbstractSide)

"""
    AbstractInterpMethod

Abstract supertype for per-axis interpolation method specification.
Used by [`HeteroInterpolantND`](@ref) to specify interpolation method per dimension.

# Subtypes
- [`CubicInterp`](@ref): Cubic spline interpolation (C² continuous)
- [`LinearInterp`](@ref): Linear interpolation (C⁰ continuous)
- [`QuadraticInterp`](@ref): Quadratic spline interpolation (C¹ continuous)
- [`ConstantInterp`](@ref): Constant (nearest-neighbor) interpolation

# Example
```julia
methods = (CubicInterp(), LinearInterp())  # cubic on axis 1, linear on axis 2
itp = interp((x, y), data; method=methods)
```
"""
abstract type AbstractInterpMethod end

"""
    CubicInterp(; bc::AbstractBC = CubicFit())

Cubic spline interpolation method for one axis.
Requires ≥4 grid points (≥3 for PeriodicBC).

# Arguments
- `bc`: Boundary condition — `CubicFit()`, `ZeroCurvBC()`, `PeriodicBC()`, etc.
"""
struct CubicInterp{BC <: AbstractBC} <: AbstractInterpMethod
    bc::BC
end
CubicInterp(; bc::AbstractBC = CubicFit()) = CubicInterp(bc)

"""
    LinearInterp()

Linear interpolation method for one axis.
Requires ≥2 grid points.
"""
struct LinearInterp <: AbstractInterpMethod end

"""
    QuadraticInterp(; bc::AbstractBC = Left(QuadraticFit()))

Quadratic spline interpolation method for one axis.
Requires ≥3 grid points.

# Arguments
- `bc`: Boundary condition — `Left(QuadraticFit())`, `Right(QuadraticFit())`, `MinCurvFit`, etc.
  Validated by the 1D `quadratic_interp` constructor.
"""
struct QuadraticInterp{BC <: AbstractBC} <: AbstractInterpMethod
    bc::BC
end
QuadraticInterp(; bc::AbstractBC = Left(QuadraticFit())) = QuadraticInterp(bc)

"""
    ConstantInterp(; side::AbstractSide = NearestSide())

Constant (nearest-neighbor) interpolation method for one axis.
Requires ≥2 grid points.

# Arguments
- `side`: Side selection — `NearestSide()`, `LeftSide()`, `RightSide()`.
"""
struct ConstantInterp{SD <: AbstractSide} <: AbstractInterpMethod
    side::SD
end
ConstantInterp(; side::AbstractSide = NearestSide()) = ConstantInterp(side)

"""
    NoInterp()

No-interpolation method for one axis. Declares that this axis is discrete and will
be queried via [`GridIdx`](@ref) at evaluation time.

At build time, no differentiation or precomputation is performed for `NoInterp` axes.
At eval time, the axis is sliced at the given `GridIdx` index — no search, no kernel.

!!! note
    Both tuple and vararg forms work: `itp((0.5, GridIdx(10)))` or `itp(0.5, GridIdx(10))`.
    `gradient`, `hessian`, and `laplacian` are supported — NoInterp axes return zero derivatives.
    Batch queries use `interp!` with SoA format: `(xvec, GridIdx(5), yvec)`.

# Examples
```julia
# 3D data: interpolate x,y; select z by index
itp = interp((x, y, z), data; method=(CubicInterp(), LinearInterp(), NoInterp()))
itp((0.5, 0.3, GridIdx(10)))    # cubic×linear on the z=10 slice
itp((0.5, 0.3, GridIdx(1)))     # same interpolant, different slice
```
"""
struct NoInterp <: AbstractInterpMethod end

# ========================================
# Hermite Family Methods (local slopes)
# ========================================

"""
    AbstractLocalHermiteInterp{BC} <: AbstractInterpMethod

Common parent of PCHIP, Cardinal, and Akima interpolation methods — the local
auto-slope Hermite family. Carries the `BC` type parameter so cross-method
dispatch can specialize on the boundary condition (e.g. `<:AbstractLocalHermiteInterp{<:PeriodicBC}`)
without enumerating each concrete type.
"""
abstract type AbstractLocalHermiteInterp{BC <: AbstractBC} <: AbstractInterpMethod end

"""
    PchipInterp{BC} <: AbstractLocalHermiteInterp{BC}

PCHIP (monotone-preserving) interpolation method for one axis.
Slopes computed via Fritsch-Carlson algorithm. Requires ≥2 grid points.

The `bc` field selects boundary handling at slope time. `NoBC()` (default)
uses one-sided 3-point FD with monotonicity clamping. `PeriodicBC(...)`
uses closed-cycle wrapped secants. BC is a type parameter so dispatch
specializes at compile time.
"""
struct PchipInterp{BC <: AbstractBC} <: AbstractLocalHermiteInterp{BC}
    bc::BC
end
PchipInterp() = PchipInterp(NoBC())   # backward-compat (NoBC default)

"""
    CardinalInterp{T, BC} <: AbstractLocalHermiteInterp{BC}

Cardinal spline interpolation method for one axis.
`tension=0` gives Catmull-Rom. Requires ≥2 grid points.

# Arguments
- `tension`: Tension parameter (0 = CatmullRom, 1 = zero slopes)
- `bc`: Boundary condition (`NoBC()` default, or `PeriodicBC(...)`)
"""
struct CardinalInterp{T, BC <: AbstractBC} <: AbstractLocalHermiteInterp{BC}
    tension::T
    bc::BC
end
CardinalInterp(tension::T) where {T} = CardinalInterp(tension, NoBC())   # backward-compat
CardinalInterp(; tension = 0.0, bc::AbstractBC = NoBC()) = CardinalInterp(tension, bc)

"""
    AkimaInterp{BC} <: AbstractLocalHermiteInterp{BC}

Akima (1970) outlier-robust interpolation method for one axis.
Requires ≥2 grid points.
"""
struct AkimaInterp{BC <: AbstractBC} <: AbstractLocalHermiteInterp{BC}
    bc::BC
end
AkimaInterp() = AkimaInterp(NoBC())   # backward-compat

# Per-method factory used by `_strip_periodic_bc` (and any future BC-swap
# helper) so the abstract dispatch above doesn't have to enumerate each
# concrete type's constructor.
@inline _replace_bc(::PchipInterp, bc::AbstractBC) = PchipInterp(bc)
@inline _replace_bc(m::CardinalInterp, bc::AbstractBC) = CardinalInterp(m.tension, bc)
@inline _replace_bc(::AkimaInterp, bc::AbstractBC) = AkimaInterp(bc)

"""
    CubicHermiteInterp <: AbstractInterpMethod

Cubic Hermite interpolation with user-supplied slopes.
Type-only in Phase 1 — ND support requires separate slope design.
"""
struct CubicHermiteInterp <: AbstractInterpMethod end
