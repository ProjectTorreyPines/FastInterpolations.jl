# ========================================
# Per-Axis Interpolation Method Types
# ========================================
# Lightweight types specifying which 1D interpolation method to use per axis.
# Used by TensorProductInterpolantND for heterogeneous per-axis interpolation.
#
# Design: Method-specific options (BC, side) are stored as type parameters
# for zero-cost dispatch — no allocation, no type instability.
#
# Include order: after bc_types.jl (needs AbstractBC, AbstractSide)

"""
    AbstractInterpMethod

Abstract supertype for per-axis interpolation method specification.
Used by [`TensorProductInterpolantND`](@ref) to specify interpolation method per dimension.

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
    Batch queries use `interp_batch_grididx!` with SoA format: `(xvec, GridIdx(5), yvec)`.

# Examples
```julia
# 3D data: interpolate x,y; select z by index
itp = interp((x, y, z), data; method=(CubicInterp(), LinearInterp(), NoInterp()))
itp((0.5, 0.3, GridIdx(10)))    # cubic×linear on the z=10 slice
itp((0.5, 0.3, GridIdx(1)))     # same interpolant, different slice
```
"""
struct NoInterp <: AbstractInterpMethod end
