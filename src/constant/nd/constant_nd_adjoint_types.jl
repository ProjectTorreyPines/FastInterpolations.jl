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
    ConstantAdjointND{Tg, N, G, B, EP, SD}

Adjoint (transpose) operator for N-dimensional constant interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND constant interpolation weight matrix.

Constructed from grids and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`: Grid type (unconstrained — supports duck types)
- `N`:   Number of dimensions
- `G`:   Grid tuple type — wrapped axes (`_CachedRange` / `_CachedVector` /
         `_ExclusivePeriodicAxis`) carry cached `h`/`inv_h`; no separate
         `spacings` field needed.
- `B`:   Per-axis BC tuple type. Stored so the ND adjoint protocol can detect
         `PeriodicBC{:exclusive}` axes and apply post-apply seam fold + trim.
- `EP`:  Extrapolation tuple type
- `SD`:  Side selection tuple type

# Architecture
Pure single-point scatter operator — no weights, no caches, no solve step.
Each query scatters to exactly 1 grid point (vs 2^N for linear, 4^N for cubic).
Per-axis anchors reuse `_ConstantAnchoredQuery` with offset computed at scatter time.
PeriodicBC support is wrapper-based (same as LinearAdjointND).

# Usage
```julia
adj = constant_adjoint((x, y), (xq, yq); side=NearestSide())
f_bar = adj(y_bar)              # allocating
adj(f_bar, y_bar)               # in-place (zero-allocation)
```
"""
struct ConstantAdjointND{
        Tg,
        N,
        G <: NTuple{N, AbstractVector{Tg}},
        B <: NTuple{N, AbstractBC},
        EP <: Tuple{Vararg{AbstractExtrap, N}},
        SD <: Tuple{Vararg{AbstractSide, N}},
    } <: AbstractAdjointND{Tg, N}
    grids::G
    bcs::B
    extraps::EP
    sides::SD
    anchors::Vector{NTuple{N, _ConstantAnchoredQuery{Tg, Tg}}}
    grid_size::NTuple{N, Int}

    # Inner ctor: ownership copy via wrapper-aware `_convert_copy`,
    # idempotent `_cache_axis` insurance for direct ctor calls.
    function ConstantAdjointND(
            grids::Tuple{Vararg{AbstractVector{Tg}, N}}, bcs::B, extraps::EP, sides::SD,
            anchors::Vector{NTuple{N, _ConstantAnchoredQuery{Tg, Tg}}}, grid_size::NTuple{N, Int}
        ) where {
            Tg, N, B <: NTuple{N, AbstractBC},
            EP <: Tuple{Vararg{AbstractExtrap, N}}, SD <: Tuple{Vararg{AbstractSide, N}},
        }
        grids_c = map((g, bc) -> _convert_copy(_cache_axis(g, bc, Tg), Tg), grids, bcs)
        return new{Tg, N, typeof(grids_c), B, EP, SD}(grids_c, bcs, extraps, sides, anchors, grid_size)
    end
end

# ========================================
# AbstractAdjointND Interface
# ========================================

@inline _n_queries(adj::ConstantAdjointND) = length(adj.anchors)
@inline _grid_size(adj::ConstantAdjointND) = adj.grid_size

# Per-axis BCs from struct field. The ND adjoint protocol checks
# `bcs[d] isa PeriodicBC{:exclusive}` for output sizing and post-apply
# seam fold (`_adjoint_apply_exclusive_nd!`).
@inline _adjoint_bcs(adj::ConstantAdjointND) = adj.bcs

@inline _adjoint_nd_apply!(f_bar, adj::ConstantAdjointND, y_bar, ops) =
    _constant_adjoint_nd_apply!(f_bar, adj, y_bar, ops)

# ========================================
# Type Introspection
# ========================================

Base.ndims(::ConstantAdjointND{Tg, N}) where {Tg, N} = N + 1
function Base.size(adj::ConstantAdjointND)
    out_size = _adjoint_output_size(adj)
    return (out_size..., _n_queries(adj))
end
Base.size(adj::ConstantAdjointND, d::Integer) = size(adj)[d]
