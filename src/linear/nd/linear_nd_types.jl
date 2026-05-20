# ========================================
# LinearInterpolantND Type Definition
# ========================================
#
# N-dimensional multilinear interpolation type.
# Performs tensor-product linear interpolation (bilinear for 2D, trilinear for 3D, etc.)
#
# Type Parameters:
# - Tg: Grid/coordinate type (AbstractFloat) - for x/y/z coordinates, spacing
# - Tv: Value type - for data values (can be Complex{Tg}, Dual, etc.)
# - N:  Number of dimensions

"""
    LinearInterpolantND{Tg, Tv, N, G, E, P}

N-dimensional multilinear interpolant for tensor-product linear interpolation.

Performs bilinear interpolation in 2D, trilinear in 3D, and n-linear in N dimensions.
The interpolation is exact at grid points and linearly blended between them.

# Type Parameters
- `Tg`: Grid/coordinate type — normally Float32/Float64, but unconstrained for duck-typed grids (e.g. ForwardDiff.Dual)
- `Tv`: Value type (unconstrained)
- `N`: Number of dimensions
- `G<:Tuple{Vararg{AbstractVector,N}}`: Grid tuple type (supports heterogeneous grids).
  Wrapped grids (`_CachedRange`/`_CachedVector`/`_ExclusivePeriodicAxis`) carry
  cached `h`/`inv_h` directly — no separate spacings field needed.
- `E<:Tuple{Vararg{AbstractExtrap,N}}`: Extrapolation mode tuple type
- `P<:Tuple{Vararg{AbstractSearchPolicy,N}}`: Search policy tuple type

# Fields
- `grids`: N-tuple of (wrapped) grid vectors for each dimension
- `data`: N-dimensional array of values at grid points
- `extraps`: N-tuple of extrapolation modes
- `searches`: N-tuple of search policies

# Performance
- **Construction**: O(1) - just stores references, no precomputation
- **Query**: O(2^N) - evaluates 2^N corner weights and sums
- **Memory**: Only stores reference to data array

# Thread-Safety
Immutable after construction; safe for concurrent read access.

# Mathematical Formulation
For a query point (x₁, x₂, ..., xₙ), compute:
1. Find cell indices (i₁, i₂, ..., iₙ) containing the point
2. Compute normalized coordinates αₖ = (xₖ - grid[iₖ]) / hₖ
3. Sum over all 2^N corners:

   f(x) = Σ_{b∈{0,1}^N} data[i₁+b₁, i₂+b₂, ..., iₙ+bₙ] × ∏ₖ wₖ(bₖ)

   where wₖ(0) = 1-αₖ and wₖ(1) = αₖ

# Derivatives
- ∂f/∂xₖ: Replace wₖ with derivative weights: w'ₖ(0) = -1/hₖ, w'ₖ(1) = 1/hₖ
- ∂²f/∂xₖ² = 0 (linear interpolation has zero second derivative within cells)

# Example
```julia
x = range(0.0, 1.0, 11)
y = range(0.0, 2.0, 21)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

itp = linear_interp((x, y), data)  # Returns LinearInterpolantND{..., 2, ...}
itp((0.5, 1.0))                     # Bilinear interpolation at (0.5, 1.0)
itp((0.5, 1.0); deriv=Val((1,0)))   # ∂f/∂x at (0.5, 1.0)
```
"""
struct LinearInterpolantND{
        Tg,
        Tv,
        N,
        G <: Tuple{Vararg{AbstractVector, N}},
        E <: Tuple{Vararg{AbstractExtrap, N}},
        P <: Tuple{Vararg{AbstractSearchPolicy, N}},
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    data::Array{Tv, N}
    extraps::E
    searches::P

    # Inner ctor: type params inferred from arg signature; outer factories
    # call this without spelling out `{Tg, Tv, N, G, E, P}`.
    function LinearInterpolantND(
            grids::Tuple{Vararg{AbstractVector{Tg}, N}},
            data::AbstractArray{Tv, N},
            extraps::Tuple{Vararg{AbstractExtrap, N}},
            searches::Tuple{Vararg{AbstractSearchPolicy, N}};
            bcs::NTuple{N, AbstractBC} = ntuple(_ -> NoBC(), Val(N))
        ) where {Tg, Tv, N}
        grids_c = map((g, bc) -> _convert_copy(_cache_axis(g, bc, Tg), Tg), grids, bcs)
        return new{Tg, Tv, N, typeof(grids_c), typeof(extraps), typeof(searches)}(
            grids_c, Array(data), extraps, searches
        )
    end
end

# ========================================
# Type Introspection
# ========================================

"""
    ndims(::LinearInterpolantND{..., N, ...}) -> Int

Return the number of dimensions.
"""
Base.ndims(::LinearInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = N

"""
    size(itp::LinearInterpolantND) -> NTuple{N, Int}

Return the grid sizes for all dimensions.
"""
Base.size(itp::LinearInterpolantND) = size(itp.data)

"""
    axes(itp::LinearInterpolantND) -> NTuple{N, AbstractVector}

Return the grid vectors for all dimensions.
"""
Base.axes(itp::LinearInterpolantND) = itp.grids

# Mirrors the 1D override: shared `_arithmetic_kernel_shape` drives inference.
@inline _output_eltype(::LinearInterpolantND{Tg, Tv, N}, ::Type{Tq}) where {Tg, Tv, N, Tq} =
    _output_eltype(_arithmetic_kernel_shape, Tg, Tv, Tq)
