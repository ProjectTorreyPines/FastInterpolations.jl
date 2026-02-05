# ========================================
# ND Cubic Interpolation Types (Generic)
# ========================================
#
# Generic N-dimensional type definitions for cubic interpolation.
# This file contains types that work for ANY dimension N.
#
# 2D-specific types are in nd_types_2d.jl (temporary, will be deprecated).
#
# Type Parameters Convention:
# - Tg: Grid/coordinate type (AbstractFloat) - used for x/y/z coordinates, spacing
# - Tv: Value type - used for data values, coefficients (can be Complex{Tg}, Dual, etc.)
# - N:  Number of dimensions
#
# Design: Strategy pattern via AbstractCoeffStrategy for coefficient computation

# ========================================
# Coefficient Strategy Types
# ========================================

"""
    AbstractCoeffStrategy

Abstract supertype for coefficient computation strategies in ND interpolation.

# Implemented Strategies
- `PreCompute`: Precompute all partial derivatives at construction (O(1) query)
- `OnTheFly`: Compute coefficients lazily at query time (O(n) query)
"""
abstract type AbstractCoeffStrategy end

"""
    PreCompute <: AbstractCoeffStrategy

Precompute all partial derivatives at construction time.

For N-dimensional interpolation, stores 2^N partial derivatives per grid point:
- 2D: 4 partials (f, ∂f/∂x, ∂f/∂y, ∂²f/∂x∂y)
- 3D: 8 partials (f, ∂f/∂x, ∂f/∂y, ∂f/∂z, ∂²f/∂x∂y, ∂²f/∂x∂z, ∂²f/∂y∂z, ∂³f/∂x∂y∂z)

# Trade-offs
- **Memory**: O(2^N × n^N) - higher than OnTheFly
- **Construction**: O(N × n^N) - expensive (N passes of 1D spline solving)
- **Query**: O(1) - ultra-fast (just polynomial evaluation)

# Best for
- Static data with many queries
- Real-time applications requiring fast evaluation
- Scenarios where construction cost can be amortized
"""
struct PreCompute <: AbstractCoeffStrategy end

"""
    OnTheFly <: AbstractCoeffStrategy

Compute coefficients lazily at query time using tensor-product (separable) approach.

Stores only the original data; computes 1D spline solutions per-axis at each query.

# Trade-offs
- **Memory**: O(n^N) - minimal (only original data)
- **Construction**: O(1) - instant (just store data reference)
- **Query**: O(n) per axis - slower (must solve 1D systems)

# Best for
- Few queries on large datasets
- Memory-constrained environments
- Data that changes frequently
"""
struct OnTheFly <: AbstractCoeffStrategy end

# ========================================
# Generic ND Coefficient Storage
# ========================================
# NOTE: NodalDerivativesND has been moved to src/core/nd_utils.jl
# for shared use by both CubicInterpolantND and QuadraticInterpolantND.

# ========================================
# Generic ND Interpolant Type
# ========================================

"""
    CubicInterpolantND{Tg, Tv, N, NP1, G, S, B, E, P}

Generic N-dimensional cubic Hermite interpolant with precomputed partial derivatives.

Stores function values AND all partial derivatives at grid nodes, enabling
ultra-fast O(1) evaluation via tensor-product Hermite polynomials.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type (can be Tg, Complex{Tg}, or other Number)
- `N`: Number of dimensions
- `NP1`: N + 1 (partials array dimensionality)
- `G`: Tuple type for grids, `NTuple{N, <:AbstractVector{Tg}}`
- `S`: Tuple type for spacings, `NTuple{N, <:AbstractGridSpacing{Tg}}`
- `B`: Tuple type for boundary conditions, `NTuple{N, <:AbstractBC}`
- `E`: Tuple type for extrapolation modes, `NTuple{N, <:ExtrapVal}`
- `P`: Tuple type for search policies, `NTuple{N, <:AbstractSearchPolicy}`

# Fields
- `grids`: N-tuple of grid vectors for each dimension
- `spacings`: N-tuple of grid spacing info (for O(1) h lookup)
- `nodal_derivs`: NodalDerivativesND containing partial derivatives at grid nodes
- `bcs`: N-tuple of boundary conditions used for construction
- `extraps`: N-tuple of extrapolation modes
- `searches`: N-tuple of search policies

# Performance
- **Construction**: O(2^N × n^N) - computes all partial derivatives
- **Query**: O(1) - tensor-product Hermite polynomial evaluation
- **Memory**: 2^N × n^N values

# Thread-Safety
Immutable after construction; safe for concurrent read access.

# Example
```julia
x = range(0.0, 2π, 50)
y = range(0.0, π, 30)
z = range(0.0, 1.0, 20)
data = [sin(xi) * cos(yj) * zk for xi in x, yj in y, zk in z]

itp = cubic_interp((x, y, z), data)  # Returns CubicInterpolantND{..., 3, 4, ...}
itp((1.0, 0.5, 0.3))                  # Evaluate at (1.0, 0.5, 0.3)
```
"""
struct CubicInterpolantND{
    Tg<:AbstractFloat,
    Tv,
    N,
    NP1,
    G<:NTuple{N, AbstractVector{Tg}},
    S<:NTuple{N, AbstractGridSpacing{Tg}},
    B<:NTuple{N, AbstractBC},
    E<:NTuple{N, ExtrapVal},
    P<:NTuple{N, AbstractSearchPolicy},
} <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    spacings::S
    nodal_derivs::NodalDerivativesND{Tv, N, NP1}
    bcs::B
    extraps::E
    searches::P

    function CubicInterpolantND{Tg, Tv, N, NP1, G, S, B, E, P}(
        grids::G, spacings::S, nodal_derivs::NodalDerivativesND{Tv, N, NP1},
        bcs::B, extraps::E, searches::P
    ) where {Tg, Tv, N, NP1, G, S, B, E, P}
        NP1 == N + 1 || throw(ArgumentError("NP1 must equal N+1"))
        new{Tg, Tv, N, NP1, G, S, B, E, P}(grids, spacings, nodal_derivs, bcs, extraps, searches)
    end
end

# ========================================
# Type Introspection (Generic ND)
# ========================================

"""
    ndims(::CubicInterpolantND{..., N, ...}) -> Int

Return the number of dimensions.
"""
Base.ndims(::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = N

"""
    size(itp::CubicInterpolantND) -> NTuple{N, Int}

Return the grid sizes for all dimensions.
"""
Base.size(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    ntuple(d -> length(itp.grids[d]), Val(N))

"""
    axes(itp::CubicInterpolantND) -> NTuple{N, AbstractVector}

Return the grid vectors for all dimensions.
"""
Base.axes(itp::CubicInterpolantND) = itp.grids

# Convenience: number of partials per grid point
num_partials(::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = 1 << N  # 2^N
num_partials(::Type{<:CubicInterpolantND{Tg, Tv, N}}) where {Tg, Tv, N} = 1 << N

# ========================================
# Generic Accessors (Val{D} dispatch)
# ========================================
#
# Using Val{D} makes D a compile-time constant for type stability.

"""
    _grid(itp, Val(d)) -> AbstractVector{Tg}

Get grid for dimension `d`. Uses `Val{D}` for compile-time dispatch.
"""
@inline _grid(itp::CubicInterpolantND, ::Val{D}) where {D} = itp.grids[D]

"""
    _spacing(itp, Val(d)) -> AbstractGridSpacing{Tg}

Get spacing for dimension `d`.
"""
@inline _spacing(itp::CubicInterpolantND, ::Val{D}) where {D} = itp.spacings[D]

"""
    _bc(itp, Val(d)) -> AbstractBC

Get boundary condition for dimension `d`.
"""
@inline _bc(itp::CubicInterpolantND, ::Val{D}) where {D} = itp.bcs[D]

"""
    _extrap(itp, Val(d)) -> ExtrapVal

Get extrapolation mode for dimension `d`.
"""
@inline _extrap(itp::CubicInterpolantND, ::Val{D}) where {D} = itp.extraps[D]

"""
    _search(itp, Val(d)) -> AbstractSearchPolicy

Get search policy for dimension `d`.
"""
@inline _search(itp::CubicInterpolantND, ::Val{D}) where {D} = itp.searches[D]

# ========================================
# @generated Tuple Extractors (zero-allocation)
# ========================================

"""
    _get_grids(itp) -> NTuple{N, AbstractVector{Tg}}

Extract all grids as a tuple. Uses @generated for zero allocation.
"""
@inline @generated function _get_grids(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    exprs = [:(itp.grids[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

"""
    _get_spacings(itp) -> NTuple{N, AbstractGridSpacing{Tg}}

Extract all spacings as a tuple.
"""
@inline @generated function _get_spacings(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    exprs = [:(itp.spacings[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

"""
    _get_extraps(itp) -> NTuple{N, ExtrapVal}

Extract all extrapolation modes as a tuple.
"""
@inline @generated function _get_extraps(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    exprs = [:(itp.extraps[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

"""
    _get_searches(itp) -> NTuple{N, AbstractSearchPolicy}

Extract all search policies as a tuple.
"""
@inline @generated function _get_searches(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    exprs = [:(itp.searches[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

# ========================================
# @GENERATED VERSIONS (for performance testing)
# ========================================
#
# Note: _handle_all_extraps, _handle_axis_extrap, and _search_all_intervals
# are now in src/core/nd_utils.jl for shared use by constant/linear ND.

"""
@generated version of _handle_all_extraps - explicit unrolling instead of ntuple closure.
"""
@generated function _handle_all_extraps_gen(
    queries::NTuple{N, Tq}, grids::NTuple{N}, extraps::NTuple{N}
) where {N, Tq}
    exprs = [:(
        @inbounds _handle_axis_extrap(queries[$d], grids[$d], extraps[$d])
    ) for d in 1:N]
    return :(tuple($(exprs...)))
end

"""
@generated version of _search_all_intervals - explicit unrolling.
"""
@generated function _search_all_intervals_gen(
    q_evals::NTuple{N, Tg}, grids::NTuple{N}, spacings::NTuple{N}, searches::NTuple{N}
) where {N, Tg}
    # Generate search calls - inline searcher creation to avoid variable naming issues
    search_exprs = [:(
        @inbounds search_interval(_to_searcher(searches[$d]), grids[$d], spacings[$d], q_evals[$d])
    ) for d in 1:N]

    # Generate restructuring
    idx_exprs = [:(results[$d][1]) for d in 1:N]
    L_exprs = [:(results[$d][2]) for d in 1:N]
    R_exprs = [:(results[$d][3]) for d in 1:N]

    return quote
        results = tuple($(search_exprs...))
        indices = tuple($(idx_exprs...))
        Ls = tuple($(L_exprs...))
        Rs = tuple($(R_exprs...))
        return (indices, Ls, Rs)
    end
end

"""
@generated version of _compute_all_local_params - explicit unrolling.
"""
@generated function _compute_all_local_params_gen(
    q_evals::NTuple{N, Tg}, spacings::NTuple{N}, indices::NTuple{N, Int}, Ls::NTuple{N, Tg}
) where {N, Tg}
    h_exprs = [:(@inbounds _get_h(spacings[$d], indices[$d])) for d in 1:N]
    inv_h_exprs = [:(@inbounds _get_inv_h(spacings[$d], indices[$d])) for d in 1:N]
    dL_exprs = [:(@inbounds q_evals[$d] - Ls[$d]) for d in 1:N]

    return quote
        hs = tuple($(h_exprs...))
        inv_hs = tuple($(inv_h_exprs...))
        dLs = tuple($(dL_exprs...))
        return (hs, inv_hs, dLs)
    end
end

# Note: _search_all_intervals is now in src/core/nd_utils.jl

# NOTE: _compute_all_local_params has been moved to src/core/nd_utils.jl
# for shared use by CubicInterpolantND and QuadraticInterpolantND.
