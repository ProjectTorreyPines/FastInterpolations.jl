# ========================================
# Abstract Type Hierarchy
# ========================================
# Abstract types for interpolant type hierarchy.
# Enables generic programming over different interpolation methods.
#
# DESIGN: Pure type hierarchy - NO methods defined here.
# All concrete functionality is implemented in specific interpolant files.
#
# Include order: FIRST - before all other interp files
#
# ========================================

"""
    AbstractInterpolant{T<:AbstractFloat}

Abstract supertype for all single-series interpolant objects.

# Type Parameter
- `T`: Float type (Float32 or Float64)

# Subtypes
- `LinearInterpolant{T}`: Piecewise linear interpolation
- `ConstantInterpolant{T}`: Piecewise constant (step) interpolation
- `QuadraticInterpolant{T}`: C1 piecewise quadratic spline
- `CubicInterpolant{T}`: C2 natural/clamped/periodic cubic spline

# Usage
This type enables generic programming over different interpolation methods:
```julia
function evaluate_all(itps::Vector{<:AbstractInterpolant}, xq)
    return [itp(xq) for itp in itps]
end
```

# Note
This is a pure type hierarchy - no methods are defined on `AbstractInterpolant` itself.
All functionality is implemented in concrete subtypes.
"""
abstract type AbstractInterpolant{T<:AbstractFloat} end

"""
    AbstractSeriesInterpolant{T<:AbstractFloat}

Abstract supertype for multi-series interpolant objects.
Series interpolants handle multiple y-series sharing the same x-grid.

# Type Parameter
- `T`: Float type (Float32 or Float64)

# Subtypes
- `CubicSeriesInterpolant{T}`: Multiple cubic splines sharing x-grid
- `LinearMultiInterpolant{T}`: Multiple linear interpolants sharing x-grid
- `ConstantMultiInterpolant{T}`: Multiple constant interpolants sharing x-grid
- `QuadraticMultiInterpolant{T}`: Multiple quadratic interpolants sharing x-grid

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
abstract type AbstractSeriesInterpolant{T<:AbstractFloat} end

# Backward compatibility alias
const AbstractMultiInterpolant = AbstractSeriesInterpolant
