# ========================================
# ND Constant Adjoint Types
# ========================================
#
# Type definitions for the N-dimensional constant adjoint operator.
# ConstantAdjointND computes f̄ = Wᵀȳ where W is the implicit ND constant
# interpolation matrix.
#
# Key difference from LinearAdjointND:
# - No weights needed — constant selects exactly 1 grid point per query
# - Reuses _ConstantAnchoredQuery per axis (no new anchor type)
# - Side mode determines which endpoint per axis

# ========================================
# ND Constant Adjoint Operator
# ========================================

"""
    ConstantAdjointND{Tg, N, G, S, EP, SD}

Adjoint (transpose) operator for N-dimensional constant interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND constant interpolation weight matrix.

Constructed from grids and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`:  Grid float type (Float32 or Float64)
- `N`:   Number of dimensions
- `G`:   Grid tuple type
- `S`:   Spacing tuple type
- `EP`:  Extrapolation tuple type
- `SD`:  Side selection tuple type

# Architecture
Pure single-point scatter operator — no weights, no caches, no solve step.
Each query scatters to exactly 1 grid point (vs 2^N for linear, 4^N for cubic).
Per-axis anchors reuse `_ConstantAnchoredQuery` with offset computed at scatter time.

# Usage
```julia
adj = constant_adjoint((x, y), (xq, yq); side=NearestSide())
f_bar = adj(y_bar)              # allocating
adj(f_bar, y_bar)               # in-place (zero-allocation)
```
"""
struct ConstantAdjointND{
        Tg <: AbstractFloat,
        N,
        G <: NTuple{N, AbstractVector{Tg}},
        S <: NTuple{N, AbstractGridSpacing{Tg}},
        EP <: Tuple{Vararg{AbstractExtrap, N}},
        SD <: Tuple{Vararg{AbstractSide, N}},
    } <: AbstractAdjointND{Tg, N}
    grids::G
    spacings::S
    extraps::EP
    sides::SD
    anchors::Vector{NTuple{N, _ConstantAnchoredQuery{Tg}}}
    grid_size::NTuple{N, Int}
end

# ========================================
# AbstractAdjointND Interface
# ========================================

@inline _n_queries(adj::ConstantAdjointND) = length(adj.anchors)
@inline _grid_size(adj::ConstantAdjointND) = adj.grid_size

# Constant has no periodic BCs — return non-periodic sentinel tuple.
@inline _adjoint_bcs(adj::ConstantAdjointND{Tg, N}) where {Tg, N} =
    ntuple(_ -> NoExtrap(), Val(N))

@inline _adjoint_nd_apply!(f_bar, adj::ConstantAdjointND, y_bar, ops) =
    _constant_adjoint_nd_apply!(f_bar, adj, y_bar, ops)

# ========================================
# Type Introspection
# ========================================

Base.ndims(::ConstantAdjointND{Tg, N}) where {Tg, N} = N + 1
function Base.size(adj::ConstantAdjointND{Tg, N}) where {Tg, N}
    out_size = _adjoint_output_size(adj)
    return (out_size..., _n_queries(adj))
end
Base.size(adj::ConstantAdjointND, d::Integer) = size(adj)[d]
