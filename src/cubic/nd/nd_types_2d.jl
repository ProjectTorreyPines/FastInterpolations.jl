# ========================================
# 2D Cubic Interpolation Types (Temporary)
# ========================================
#
# 2D-specific type definitions for cubic interpolation.
# These are TEMPORARY and will be deprecated once CubicInterpolantND is validated.
#
# Type Parameters Convention:
# - Tg: Grid/coordinate type (AbstractFloat) - used for x/y coordinates, spacing
# - Tv: Value type - used for data values, coefficients (can be Complex{Tg}, Dual, etc.)
#
# NOTE: This file exists for performance comparison with generic ND implementation.
#       Once ND is validated, 2D will become a type alias: CubicInterpolant2D = CubicInterpolantND{...,2,...}

# ========================================
# 2D Coefficient Storage
# ========================================

"""
    NodalDerivatives2D{Tv}

Storage for precomputed 2D partial derivatives at grid nodes.

Stores f, ∂f/∂x, ∂f/∂y, and ∂²f/∂x∂y at each (xᵢ, yⱼ) grid node,
enabling O(1) bicubic Hermite evaluation.

# Fields
- `partials::Array{Tv, 3}`: Partial derivatives array of shape (4, nx, ny)
  - `partials[1, i, j]` = f(xᵢ, yⱼ)
  - `partials[2, i, j]` = ∂f/∂x at (xᵢ, yⱼ)
  - `partials[3, i, j]` = ∂f/∂y at (xᵢ, yⱼ)
  - `partials[4, i, j]` = ∂²f/∂x∂y at (xᵢ, yⱼ)

# Type Parameters
- `Tv`: Value type (Float64, ComplexF64, etc.)

# Note
This is equivalent to `NodalDerivativesND{Tv, 2, 3}` but kept separate
for performance comparison during the migration to generic ND.
"""
struct NodalDerivatives2D{Tv}
    partials::Array{Tv, 3}
end

# ========================================
# 2D Interpolant Type
# ========================================

"""
    CubicInterpolant2D{Tg, Tv, GX, GY, SX, SY, BX, BY, EX, EY, PX, PY}

2D bicubic interpolant with precomputed partial derivatives.

Stores function values AND partial derivatives at grid nodes, enabling
ultra-fast O(1) evaluation via tensor-product Hermite polynomials.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type (can be Tg, Complex{Tg}, or other Number)
- `GX, GY`: Grid vector types for x and y axes
- `SX, SY`: Grid spacing types (ScalarSpacing or VectorSpacing)
- `BX, BY`: Boundary condition types for each axis
- `EX, EY`: Extrapolation modes (Val types)
- `PX, PY`: Search policy types for each axis

# Fields
- `x`, `y`: Grid points for each axis
- `spacing_x`, `spacing_y`: Grid spacing info with precomputed reciprocals
- `nodal_derivs`: NodalDerivatives2D containing partial derivatives at grid nodes
- `bc_x`, `bc_y`: Boundary conditions used for construction
- `extrap_x`, `extrap_y`: Per-axis extrapolation modes
- `search_x`, `search_y`: Per-axis search policies

# Stored Partials (in nodal_derivs.partials)
- `partials[1, i, j]` = f(xᵢ, yⱼ)
- `partials[2, i, j]` = ∂f/∂x at (xᵢ, yⱼ)
- `partials[3, i, j]` = ∂f/∂y at (xᵢ, yⱼ)
- `partials[4, i, j]` = ∂²f/∂x∂y at (xᵢ, yⱼ)

# Performance
- **Construction**: O(nx × ny) - computes all partial derivatives
- **Query**: O(1) - tensor-product Hermite polynomial evaluation
- **Memory**: 4 × nx × ny values (4× the original data size)

# Thread-Safety
Immutable after construction; safe for concurrent read access.

# Note
This is equivalent to `CubicInterpolantND{Tg, Tv, 2, 3, ...}` but kept separate
for performance comparison. Will become a type alias after ND validation.
Once validated, `CubicInterpolant2D` will be deprecated in favor of the generic ND type.

# Example
```julia
x = range(0.0, 2π, 50)
y = range(0.0, π, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

itp = cubic_interp((x, y), data)  # Returns CubicInterpolant2D
itp((1.0, 0.5))                    # Evaluate at (1.0, 0.5)
```
"""
struct CubicInterpolant2D{
    Tg<:AbstractFloat,
    Tv,
    GX<:AbstractVector{Tg},
    GY<:AbstractVector{Tg},
    SX<:AbstractGridSpacing{Tg},
    SY<:AbstractGridSpacing{Tg},
    BX<:AbstractBC,
    BY<:AbstractBC,
    EX<:ExtrapVal,
    EY<:ExtrapVal,
    PX<:AbstractSearchPolicy,
    PY<:AbstractSearchPolicy,
} <: AbstractInterpolant{Tg, Tv}
    x::GX
    y::GY
    spacing_x::SX
    spacing_y::SY
    nodal_derivs::NodalDerivatives2D{Tv}
    bc_x::BX
    bc_y::BY
    extrap_x::EX
    extrap_y::EY
    search_x::PX
    search_y::PY
end

# Backwards compatibility alias
const BicubicInterpolant = CubicInterpolant2D

# ========================================
# Type Introspection (2D)
# ========================================

"""
    ndims(::CubicInterpolant2D) -> Int

Return the number of dimensions (always 2 for CubicInterpolant2D).
"""
Base.ndims(::CubicInterpolant2D) = 2

"""
    size(itp::CubicInterpolant2D) -> Tuple{Int, Int}

Return the grid size (nx, ny).
"""
Base.size(itp::CubicInterpolant2D) = (length(itp.x), length(itp.y))

"""
    axes(itp::CubicInterpolant2D) -> Tuple

Return the grid vectors (x, y).
"""
Base.axes(itp::CubicInterpolant2D) = (itp.x, itp.y)
