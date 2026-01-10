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
    AbstractMultiInterpolant{T<:AbstractFloat}

Abstract supertype for multi-series interpolant objects.
Multi-interpolants wrap multiple single interpolants sharing the same x-grid.

# Type Parameter
- `T`: Float type (Float32 or Float64)

# Subtypes
- `CubicMultiInterpolant{T}`: Multiple cubic splines sharing x-grid

# Key Features
- Anchor optimization: compute interval once, evaluate all series
- Composition-based: wrap `Vector{<:AbstractInterpolant}`, not raw data
- Zero-allocation batch evaluation with pre-built anchors

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
y1, y2, y3 = sin.(2π .* x), cos.(2π .* x), exp.(-x)

mitp = cubic_interp(x, [y1, y2, y3])  # Creates CubicMultiInterpolant

vals = mitp(0.5)            # Returns Vector of 3 values
mitp(output, 0.5)           # In-place evaluation
```

# Note
This is a pure type hierarchy - no methods are defined on `AbstractMultiInterpolant` itself.
All functionality is implemented in concrete subtypes.
"""
abstract type AbstractMultiInterpolant{T<:AbstractFloat} end
