# ========================================
# ND Quadratic Interpolation Types (Generic)
# ========================================
#
# Generic N-dimensional type definitions for quadratic interpolation.
#
# Type Parameters Convention:
# - Tg: Grid/coordinate type (AbstractFloat) - used for x/y/z coordinates, spacing
# - Tv: Value type - used for data values, coefficients (can be Complex{Tg}, Dual, etc.)
# - N:  Number of dimensions
#
# Reuses NodalDerivativesND from core/nd_utils.jl for partial derivative storage.
# Uses Tuple{Vararg{T, N}} instead of NTuple{N, <:T} to support heterogeneous types.

# ========================================
# Generic ND Interpolant Type
# ========================================

"""
    QuadraticInterpolantND{Tg, Tv, N, NP1, G, S, B, P}

Generic N-dimensional quadratic interpolant with precomputed partial derivatives.

Stores function values AND all partial derivatives at grid nodes, enabling
ultra-fast O(1) evaluation via tensor-product quadratic polynomials.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type (can be Tg, Complex{Tg}, or other Number)
- `N`: Number of dimensions
- `NP1`: N + 1 (partials array dimensionality)
- `G`: Tuple type for grids
- `S`: Tuple type for spacings
- `B`: Tuple type for boundary conditions
- `P`: Tuple type for search policies

# Fields
- `grids`: N-tuple of grid vectors for each dimension
- `spacings`: N-tuple of grid spacing info (for O(1) h lookup)
- `nodal_derivs`: NodalDerivativesND containing partial derivatives at grid nodes
- `bcs`: N-tuple of boundary conditions used for construction
- `extraps`: N-tuple of extrapolation modes
- `searches`: N-tuple of search policies

# Performance
- **Construction**: O(2^N x n^N) - computes all partial derivatives via recurrence
- **Query**: O(1) - tensor-product quadratic polynomial evaluation
- **Memory**: 2^N x n^N values

# Thread-Safety
Immutable after construction; safe for concurrent read access.

# Example
```julia
x = range(0.0, 2.0, 20)
y = range(0.0, 1.0, 15)
data = [xi^2 + yi^2 for xi in x, yi in y]

itp = quadratic_interp((x, y), data)
itp((1.0, 0.5))                  # Evaluate at (1.0, 0.5)
itp((1.0, 0.5); deriv=(1, 0))   # dfdx
```
"""
struct QuadraticInterpolantND{
    Tg<:AbstractFloat,
    Tv,
    N,
    NP1,
    G<:Tuple{Vararg{AbstractVector, N}},
    S<:Tuple{Vararg{AbstractGridSpacing, N}},
    B<:Tuple{Vararg{AbstractBC, N}},
    P<:Tuple{Vararg{AbstractSearchPolicy, N}},
} <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    spacings::S
    nodal_derivs::NodalDerivativesND{Tv, N, NP1}
    bcs::B
    extraps::NTuple{N, ExtrapVal}
    searches::P

    function QuadraticInterpolantND{Tg, Tv, N, NP1, G, S, B, P}(
        grids::G, spacings::S, nodal_derivs::NodalDerivativesND{Tv, N, NP1},
        bcs::B, extraps::NTuple{N, ExtrapVal}, searches::P
    ) where {Tg, Tv, N, NP1, G, S, B, P}
        NP1 == N + 1 || throw(ArgumentError("NP1 must equal N+1"))
        new{Tg, Tv, N, NP1, G, S, B, P}(grids, spacings, nodal_derivs, bcs, extraps, searches)
    end
end

# ========================================
# Type Introspection
# ========================================

Base.ndims(::QuadraticInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = N

Base.size(itp::QuadraticInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    ntuple(d -> length(itp.grids[d]), Val(N))

Base.axes(itp::QuadraticInterpolantND) = itp.grids

num_partials(::QuadraticInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = 1 << N
num_partials(::Type{<:QuadraticInterpolantND{Tg, Tv, N}}) where {Tg, Tv, N} = 1 << N

# ========================================
# Generic Accessors (Val{D} dispatch)
# ========================================

@inline _grid(itp::QuadraticInterpolantND, ::Val{D}) where {D} = itp.grids[D]
@inline _spacing(itp::QuadraticInterpolantND, ::Val{D}) where {D} = itp.spacings[D]
@inline _bc(itp::QuadraticInterpolantND, ::Val{D}) where {D} = itp.bcs[D]
@inline _extrap(itp::QuadraticInterpolantND, ::Val{D}) where {D} = itp.extraps[D]
@inline _search(itp::QuadraticInterpolantND, ::Val{D}) where {D} = itp.searches[D]

