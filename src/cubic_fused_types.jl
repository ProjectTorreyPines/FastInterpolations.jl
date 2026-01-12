# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CUBIC FUSED INTERPOLANT TYPES                           ║
# ║         High-performance fused multi-series cubic interpolant              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Key difference from CubicMultiInterpolant:
# - CubicMultiInterpolant: Composition of Vector{CubicInterpolant}, pointer chasing
# - CubicMultiInterpolantFused: Interleaved matrix layout, SIMD-friendly
#
# Memory layout: y[series, point] and z[series, point] for contiguous column access
# This enables 15-20× speedup for m ≥ 40,000 series by eliminating cache thrashing.
#
# Include order: ... → cubic_interpolant.jl → cubic_fused_types.jl → ...
#

"""
    CubicMultiInterpolantFused{T,X,S,B} <: AbstractMultiInterpolant{T}

High-performance fused multi-series cubic interpolant with interleaved memory layout.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `X`: Grid type (Vector{T} or AbstractRange{T})
- `S`: Grid spacing type (ScalarSpacing{T} for Range, VectorSpacing{T} for Vector)
- `B`: Boundary condition config type (BCPair or PeriodicData)

# Fields
- `x::X`: Grid points (immutable after construction)
- `spacing::S`: Grid spacing data (ScalarSpacing for uniform, VectorSpacing for non-uniform)
- `y::Matrix{T}`: Function values in [n_series × n_points] layout
- `z::Matrix{T}`: Second derivative coefficients in [n_series × n_points] layout
- `bc_config::B`: Boundary condition configuration
- `extrap::ExtrapVal`: Extrapolation mode (:none, :constant, :extension, :wrap)
- `n_series::Int`: Number of y-data series
- `n_points::Int`: Number of grid points

# Memory Layout

The key optimization is the **interleaved matrix layout**:
```
y[series_k, point_i] = y_k(x_i)
z[series_k, point_i] = z_k(x_i)  # second derivative
```

When evaluating at a single query point, we access columns `y[:, i]` and `z[:, i]`
which are **contiguous in memory**. This enables:
- Cache-friendly access pattern (no pointer chasing)
- SIMD vectorization via `@simd` loops
- 15-20× speedup for m ≥ 40,000 series

# Comparison with CubicMultiInterpolant

| Aspect | CubicMultiInterpolant | CubicMultiInterpolantFused |
|--------|----------------------|---------------------------|
| Storage | Vector{CubicInterpolant} | Matrix{T} (interleaved) |
| Memory pattern | Pointer chasing | Contiguous columns |
| Best for | Few series (m < 100) | Many series (m ≥ 40k) |
| Flexibility | Each series independent | All share BC/extrap |

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
y1, y2, y3 = sin.(2π .* x), cos.(2π .* x), exp.(-x)

# Create fused interpolant
mitp = cubic_interp_fused(x, [y1, y2, y3])

# Scalar evaluation (returns Vector of 3 values)
vals = mitp(0.5)

# In-place evaluation (zero allocation)
output = zeros(3)
mitp(output, 0.5)
```

See also: [`CubicMultiInterpolant`](@ref), [`cubic_interp_fused`](@ref)
"""
struct CubicMultiInterpolantFused{
    T<:AbstractFloat,
    X<:AbstractVector{T},
    S<:AbstractGridSpacing{T},
    B
} <: AbstractMultiInterpolant{T}
    x::X                  # Grid points
    spacing::S            # Grid spacing (ScalarSpacing or VectorSpacing)
    y::Matrix{T}          # Function values [n_series × n_points]
    z::Matrix{T}          # Second derivatives [n_series × n_points]
    bc_config::B          # Boundary condition config (BCPair or PeriodicData)
    extrap::ExtrapVal     # Extrapolation mode
    n_series::Int         # Number of series
    n_points::Int         # Number of grid points
end
