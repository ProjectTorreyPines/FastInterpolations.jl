# ========================================
# ND Quadratic Interpolation Types (Generic)
# ========================================
#
# Generic N-dimensional type definitions for quadratic interpolation.
#
# Type Parameters Convention:
# - Tg: Grid/coordinate type (unconstrained) - used for x/y/z coordinates, spacing
# - Tv: Value type - used for data values, coefficients (can be Complex{Tg}, Dual, etc.)
# - N:  Number of dimensions
#
# Reuses _NodalDerivativesND from core/nd_utils.jl for partial derivative storage.
# Uses Tuple{Vararg{T, N}} instead of NTuple{N, <:T} to support heterogeneous types.

# ========================================
# Generic ND Interpolant Type
# ========================================

"""
    QuadraticInterpolantND{Tg, Tv, N, NP1, G, B, E, P}

Generic N-dimensional quadratic interpolant with precomputed partial derivatives.

Stores function values AND all partial derivatives at grid nodes, enabling
ultra-fast O(1) evaluation via tensor-product quadratic polynomials.

# Type Parameters
- `Tg`: Grid/coordinate type (unconstrained)
- `Tv`: Value type (unconstrained)
- `N`: Number of dimensions
- `NP1`: N + 1 (partials array dimensionality)
- `G`: Tuple type for (wrapped) grids — each axis is `_CachedRange{Tg}` /
       `_CachedVector{Tg, Tinv}` / `_ExclusivePeriodicAxis`, carrying cached
       `h`/`inv_h` directly (no separate spacings field needed)
- `B`: Tuple type for boundary conditions
- `E`: Tuple type for extrapolation modes
- `P`: Tuple type for search policies

# Fields
- `grids`: N-tuple of (wrapped) grid vectors for each dimension
- `nodal_derivs`: _NodalDerivativesND containing partial derivatives at grid nodes
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
        Tg,
        Tv,
        N,
        NP1,
        G <: Tuple{Vararg{AbstractVector, N}},
        B <: Tuple{Vararg{AbstractBC, N}},
        E <: Tuple{Vararg{AbstractExtrap, N}},
        P <: Tuple{Vararg{AbstractSearchPolicy, N}},
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    nodal_derivs::_NodalDerivativesND{Tv, N, NP1}
    bcs::B
    extraps::E
    searches::P

    function QuadraticInterpolantND{Tg, Tv, N, NP1, G, B, E, P}(
            grids::Tuple{Vararg{AbstractVector, N}}, nodal_derivs::_NodalDerivativesND{Tv, N, NP1},
            bcs::B, extraps::E, searches::P
        ) where {Tg, Tv, N, NP1, G, B, E, P}
        NP1 == N + 1 || throw(ArgumentError("NP1 must equal N+1"))
        # Per-axis ownership copy + element-type promotion. Outer
        # `quadratic_interp` already applied `_cache_axis` per axis, so each
        # grid is a wrapper carrying cached `h`/`inv_h` (sharing the user
        # buffer in `inner`). `_convert_copy(g, Tg)` is wrapper-preserving:
        # `Base.copy` for same eltype, single-pass rebuild for different
        # eltype. No separate `spacings` field — `_get_h(itp.grids[d], i)`
        # is the source of truth.
        grids_c = map(g -> _convert_copy(g, Tg), grids)
        return new{Tg, Tv, N, NP1, typeof(grids_c), B, E, P}(grids_c, nodal_derivs, bcs, extraps, searches)
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
@inline _bc(itp::QuadraticInterpolantND, ::Val{D}) where {D} = itp.bcs[D]
"""
    _extrap(itp, Val(d)) -> AbstractExtrap

Get extrapolation mode for dimension `d`.
"""
@inline _extrap(itp::QuadraticInterpolantND, ::Val{D}) where {D} = itp.extraps[D]
@inline _search(itp::QuadraticInterpolantND, ::Val{D}) where {D} = itp.searches[D]
