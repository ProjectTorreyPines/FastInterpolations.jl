# ========================================
# Abstract Type Hierarchy
# ========================================
# Abstract types for interpolant type hierarchy.
# Enables generic programming over different interpolation methods.
#
# DESIGN: Pure type hierarchy - NO methods defined here.
# All concrete functionality is implemented in specific interpolant files.
#
# TYPE PARAMETERS:
# - Tg: Grid/coordinate type (AbstractFloat) - used for x-coordinates, spacing, search
# - Tv: Value type - used for y-values, coefficients, return values
#       Can be AbstractFloat (same as Tg), Complex{Tg}, or other Number types
#
# Include order: FIRST - before all other interp files
#
# ========================================

"""
    AbstractInterpolant{Tg<:AbstractFloat, Tv}

Abstract supertype for all interpolant objects.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64) - used for x-coordinates, spacing, search
- `Tv`: Value type - used for y-values, coefficients, return values.
        Can be `Tg` (real-valued), `Complex{Tg}` (complex-valued), or other Number types.

# Design Invariant
- Grid operations (search, spacing) always use Tg
- Value operations (kernel, coefficients) use Tv
- Evaluation returns type based on promote_type(Tv, query_type)

# Subtypes
- `AbstractSeriesInterpolant{Tg, Tv}`: Multi-series interpolants
- `LinearInterpolant{Tg, Tv}`: Piecewise linear interpolation
- `ConstantInterpolant{Tg, Tv}`: Piecewise constant (step) interpolation
- `QuadraticInterpolant{Tg, Tv}`: C1 piecewise quadratic spline
- `CubicInterpolant{Tg, Tv}`: C2 natural/clamped/periodic cubic spline

# Note
This is a pure type hierarchy - no methods are defined on `AbstractInterpolant` itself.
All functionality is implemented in concrete subtypes.
"""
abstract type AbstractInterpolant{Tg<:AbstractFloat, Tv} end

"""
    AbstractSeriesInterpolant{Tg<:AbstractFloat, Tv}

Abstract supertype for multi-series interpolant objects.
Series interpolants handle multiple y-series sharing the same x-grid.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type - can be real or complex

# Subtypes
- `LinearSeriesInterpolant{Tg, Tv}`: Multiple linear interpolants sharing x-grid
- `ConstantSeriesInterpolant{Tg, Tv}`: Multiple constant interpolants sharing x-grid
- `QuadraticSeriesInterpolant{Tg, Tv}`: Multiple quadratic interpolants sharing x-grid
- `CubicSeriesInterpolant{Tg, Tv}`: Multiple cubic splines sharing x-grid

# Key Features
- Anchor optimization: compute interval once, evaluate all series
- Matrix storage: unified storage for optimal SIMD vectorization
- Zero-allocation batch evaluation with pre-built anchors

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
y1, y2, y3 = sin.(2π .* x), cos.(2π .* x), exp.(-x)

sitp = cubic_interp(x, [y1, y2, y3])  # Creates CubicSeriesInterpolant

vals = sitp(0.5)            # Returns Vector of 3 values
sitp(output, 0.5)           # In-place evaluation
```

# Note
This is a pure type hierarchy - no methods are defined on `AbstractSeriesInterpolant` itself.
All functionality is implemented in concrete subtypes.
"""
abstract type AbstractSeriesInterpolant{Tg<:AbstractFloat, Tv} <: AbstractInterpolant{Tg, Tv} end

"""
    AbstractInterpolantND{Tg<:AbstractFloat, Tv, N}

Abstract supertype for N-dimensional interpolant objects.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type - can be real, complex, or other Number types
- `N`: Number of dimensions

# API Differences from 1D Interpolants
- **Evaluation**: `itp(x::NTuple{N})` or `itp(x::AbstractVector)` instead of `itp(x::Real)`
- **Derivatives**: Use `deriv` keyword (e.g., `itp(x; deriv=(1,0))`) or `deriv_view(itp, (1,0))`
- **Vector Calculus**: Supports `gradient`, `hessian`, `laplacian`

# Subtypes
- `CubicInterpolantND{Tg, Tv, N}`: N-dimensional cubic Hermite interpolation

# Example
```julia
x, y = range(0, 1, 50), range(0, 1, 50)
data = [sin(xi) * cos(yj) for xi in x, yj in y]
itp = cubic_interp((x, y), data)  # Returns CubicInterpolantND{..., 2}

itp((0.5, 0.5))                    # Evaluate
itp((0.5, 0.5); deriv=(1, 0))      # ∂f/∂x
gradient(itp, (0.5, 0.5))          # (∂f/∂x, ∂f/∂y)
```
"""
abstract type AbstractInterpolantND{Tg<:AbstractFloat, Tv, N} <: AbstractInterpolant{Tg, Tv} end

# ========================================
# Type Helper Functions
# ========================================

"""
    grid_type(::AbstractInterpolant{Tg, Tv}) -> Type{Tg}

Get the grid/coordinate type of an interpolant.
"""
@inline grid_type(::AbstractInterpolant{Tg, Tv}) where {Tg, Tv} = Tg

"""
    value_type(::AbstractInterpolant{Tg, Tv}) -> Type{Tv}

Get the value type of an interpolant.
"""
@inline value_type(::AbstractInterpolant{Tg, Tv}) where {Tg, Tv} = Tv

"""
    eval_type(::AbstractInterpolant{Tg, Tv}, ::Type{Tq}) -> Type

Compute the output type when evaluating an interpolant with query type `Tq`.
This is `promote_type(Tv, Tq)`, accounting for value type (real or complex)
and query type (standard float or ForwardDiff.Dual).

# Examples
```julia
itp = linear_interp([0.0, 1.0], [1.0, 2.0])
eval_type(itp, Float64)  # Float64

itp_c = linear_interp([0.0, 1.0], [1.0+0im, 2.0+0im])
eval_type(itp_c, Float64)  # ComplexF64
```
"""
@inline eval_type(::AbstractInterpolant{Tg, Tv}, ::Type{Tq}) where {Tg, Tv, Tq} = promote_type(Tv, Tq)
