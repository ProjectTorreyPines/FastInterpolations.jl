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

"""
    NodalDerivativesND{Tv, N, NP1}

Storage for precomputed N-dimensional partial derivatives at grid nodes.

This structure stores the function value f and all mixed partial derivatives
(∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁∂x₂, etc.) at each grid node, enabling O(1) evaluation
via tensor-product Hermite polynomials.

# Type Parameters
- `Tv`: Value type (Float64, ComplexF64, etc.)
- `N`: Number of dimensions
- `NP1`: N + 1 (array dimensionality, Julia can't compute N+1 in type definition)

# Fields
- `partials::Array{Tv, NP1}`: Partial derivatives array of shape (2^N, n₁, n₂, ..., nₙ)

# Partials Indexing Convention (bit-encoding of derivatives)
The first index `p` encodes which partial derivative via binary representation:
- Bit d set → differentiate with respect to dimension d
- p=1 (binary 0...0): f (no derivatives)
- p=2 (binary 0...1): ∂f/∂x₁
- p=3 (binary 0..10): ∂f/∂x₂
- p=4 (binary 0..11): ∂²f/∂x₁∂x₂
- etc.

# Examples
For N=2 (4 partials per point):
- `partials[1, i, j]` = f(xᵢ, yⱼ)         (p=1, binary 00)
- `partials[2, i, j]` = ∂f/∂x             (p=2, binary 01)
- `partials[3, i, j]` = ∂f/∂y             (p=3, binary 10)
- `partials[4, i, j]` = ∂²f/∂x∂y          (p=4, binary 11)

For N=3 (8 partials per point):
- `partials[1, i, j, k]` = f              (p=1, binary 000)
- `partials[2, i, j, k]` = ∂f/∂x          (p=2, binary 001)
- `partials[3, i, j, k]` = ∂f/∂y          (p=3, binary 010)
- `partials[4, i, j, k]` = ∂²f/∂x∂y       (p=4, binary 011)
- `partials[5, i, j, k]` = ∂f/∂z          (p=5, binary 100)
- `partials[6, i, j, k]` = ∂²f/∂x∂z       (p=6, binary 101)
- `partials[7, i, j, k]` = ∂²f/∂y∂z       (p=7, binary 110)
- `partials[8, i, j, k]` = ∂³f/∂x∂y∂z     (p=8, binary 111)
"""
struct NodalDerivativesND{Tv, N, NP1}
    partials::Array{Tv, NP1}

    function NodalDerivativesND{Tv, N, NP1}(partials::Array{Tv, NP1}) where {Tv, N, NP1}
        NP1 == N + 1 || throw(ArgumentError("NP1 must equal N+1, got NP1=$NP1, N=$N"))
        size(partials, 1) == (1 << N) || throw(DimensionMismatch(
            "First dimension must be 2^N=$(1 << N), got $(size(partials, 1))"
        ))
        new{Tv, N, NP1}(partials)
    end
end

# Convenience constructor that computes NP1 automatically
function NodalDerivativesND{Tv, N}(partials::Array{Tv, NP1}) where {Tv, N, NP1}
    NodalDerivativesND{Tv, N, NP1}(partials)
end

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
} <: AbstractInterpolant{Tg, Tv}
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
@generated function _get_grids(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    exprs = [:(itp.grids[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

"""
    _get_spacings(itp) -> NTuple{N, AbstractGridSpacing{Tg}}

Extract all spacings as a tuple.
"""
@generated function _get_spacings(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    exprs = [:(itp.spacings[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

"""
    _get_extraps(itp) -> NTuple{N, ExtrapVal}

Extract all extrapolation modes as a tuple.
"""
@generated function _get_extraps(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    exprs = [:(itp.extraps[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

"""
    _get_searches(itp) -> NTuple{N, AbstractSearchPolicy}

Extract all search policies as a tuple.
"""
@generated function _get_searches(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    exprs = [:(itp.searches[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

# ========================================
# Generic Per-Axis Helpers (allocation-free)
# ========================================

"""
    _handle_all_extraps(queries, grids, extraps) -> NTuple{N, Tg}

Apply extrapolation handling to all query coordinates.
Returns tuple of processed query values ready for interpolation.
"""
@inline function _handle_all_extraps(
    queries::NTuple{N, Tq}, grids::NTuple{N}, extraps::NTuple{N}
) where {N, Tq}
    ntuple(Val(N)) do d
        @inbounds _handle_axis_extrap(queries[d], grids[d], extraps[d])
    end
end

# Extrapolation handlers (defined here for generic use, 2D versions may override)
@inline function _handle_axis_extrap(q::Tq, axis::AbstractVector{Tg}, ::Val{:none}) where {Tq, Tg}
    @boundscheck _check_domain(axis, q, Val(:none))
    return Tg(q)
end

@inline function _handle_axis_extrap(q::Tq, axis::AbstractVector{Tg}, ::Val{:constant}) where {Tq, Tg}
    return clamp(Tg(q), first(axis), last(axis))
end

@inline function _handle_axis_extrap(q::Tq, axis::AbstractVector{Tg}, ::Val{:wrap}) where {Tq, Tg}
    return _wrap_to_domain(Tg(q), first(axis), last(axis))
end

# ========================================
# @GENERATED VERSIONS (for performance testing)
# ========================================

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

"""
    _search_all_intervals(q_evals, grids, spacings, searches) -> (indices, Ls, Rs)

Perform interval search on all axes.
Returns tuples of: indices (cell index), Ls (left bounds), Rs (right bounds).
"""
@inline function _search_all_intervals(
    q_evals::NTuple{N, Tg}, grids::NTuple{N}, spacings::NTuple{N}, searches::NTuple{N}
) where {N, Tg}
    results = ntuple(Val(N)) do d
        searcher = @inbounds _to_searcher(searches[d])
        @inbounds search_interval(searcher, grids[d], spacings[d], q_evals[d])
    end
    # Restructure (idx, L, R) tuples into separate tuples
    indices = ntuple(d -> @inbounds(results[d][1]), Val(N))
    Ls = ntuple(d -> @inbounds(results[d][2]), Val(N))
    Rs = ntuple(d -> @inbounds(results[d][3]), Val(N))
    return (indices, Ls, Rs)
end

"""
    _compute_all_local_params(q_evals, spacings, indices, Ls) -> (hs, inv_hs, dLs)

Compute local cell parameters for all axes.
Returns tuples of: hs (cell widths), inv_hs (reciprocals), dLs (left deltas).
"""
@inline function _compute_all_local_params(
    q_evals::NTuple{N, Tg}, spacings::NTuple{N}, indices::NTuple{N, Int}, Ls::NTuple{N, Tg}
) where {N, Tg}
    hs = ntuple(Val(N)) do d
        @inbounds _get_h(spacings[d], indices[d])
    end
    inv_hs = ntuple(Val(N)) do d
        @inbounds _get_inv_h(spacings[d], indices[d])
    end
    dLs = ntuple(Val(N)) do d
        @inbounds q_evals[d] - Ls[d]
    end
    return (hs, inv_hs, dLs)
end
