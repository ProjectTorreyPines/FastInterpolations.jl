# ========================================
# ND Linear Adjoint Types
# ========================================
#
# Type definitions for the N-dimensional linear adjoint operator.
# LinearAdjointND computes f̄ = Wᵀȳ where W is the implicit ND linear interpolation matrix.
#
# Key difference from CubicAdjointND:
# - No caches, no BCs, no solve step — pure scatter
# - 2 weights per axis (value + deriv) instead of 4

# ========================================
# Per-Query Anchor (Pre-Baked Weights)
# ========================================

"""
    _LinearNDAdjointAnchor{Tg, N}

Precomputed per-query data for ND linear adjoint scatter.
Stores cell location and per-axis weights for value and derivative evaluation.

# Fields
- `indices`: Per-axis cell index (left node of containing interval)
- `w_value`: EvalValue weights per axis — `(1-α, α)` tuple per dimension
- `w_deriv`: EvalDeriv1 weights per axis — `(-inv_h, inv_h)` tuple per dimension

# Weight Convention
For each axis d with normalized position `α = (xq - xL) / h`:
- `w_value[d] = (1-α, α)` — linear interpolation weights
- `w_deriv[d] = (-inv_h, inv_h)` — first derivative weights
- 2nd+ derivatives are identically zero (handled by early return)
"""
struct _LinearNDAdjointAnchor{Tg, N}
    indices::NTuple{N, Int}
    w_value::NTuple{N, NTuple{2, Tg}}
    w_deriv::NTuple{N, NTuple{2, Tg}}
end

# ========================================
# ND Linear Adjoint Operator
# ========================================

"""
    LinearAdjointND{Tg, N, G, B, EP}

Adjoint (transpose) operator for N-dimensional linear interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND linear interpolation weight matrix.

Constructed from grids and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`:  Grid float type (Float32 or Float64)
- `N`:   Number of dimensions
- `G`:   Grid tuple type — wrapped axes (`_CachedRange` / `_CachedVector` /
         `_ExclusivePeriodicAxis`) carry cached `h`/`inv_h`; no separate
         `spacings` field needed.
- `B`:   Per-axis BC tuple type. Stored so the ND adjoint protocol can detect
         seam-folded periodic axes (`_is_periodic_seam_folded(bc)` — covers
         both `:exclusive` and `:extended`) and apply post-apply seam fold + trim.
- `EP`:  Extrapolation tuple type

# Architecture
Pure scatter operator — no caches, no tridiagonal solve. The only per-axis
state is pre-baked weights in the anchors. PeriodicBC support is wrapper-
based: `_ExclusivePeriodicAxis` makes search return `idx_R = 1` for the seam
cell and `_get_h` returns the correct seam-cell width.

# Usage
```julia
adj = linear_adjoint((x, y), (xq, yq))
f_bar = adj(y_bar)              # allocating
adj(f_bar, y_bar)               # in-place (zero-allocation)
```
"""
struct LinearAdjointND{
        Tg,
        N,
        G <: NTuple{N, AbstractVector{Tg}},
        B <: NTuple{N, AbstractBC},
        EP <: Tuple{Vararg{AbstractExtrap, N}},
    } <: AbstractAdjointND{Tg, N}
    grids::G
    bcs::B
    extraps::EP
    anchors::Vector{_LinearNDAdjointAnchor{Tg, N}}
    grid_size::NTuple{N, Int}

    # Inner constructor: ownership copy via wrapper-aware `_convert_copy`,
    # idempotent `_cache_axis` insurance (already wrapped by outer API, but
    # this guards direct ctor calls from tests/external code).
    function LinearAdjointND(
            grids::Tuple{Vararg{AbstractVector{Tg}, N}}, bcs::B, extraps::EP,
            anchors::Vector{_LinearNDAdjointAnchor{Tg, N}}, grid_size::NTuple{N, Int}
        ) where {
            Tg, N, B <: NTuple{N, AbstractBC},
            EP <: Tuple{Vararg{AbstractExtrap, N}},
        }
        grids_c = map((g, bc, T) -> _convert_copy(_cache_axis(g, bc, T), T), grids, bcs, ntuple(_ -> Tg, Val(N)))
        return new{Tg, N, typeof(grids_c), B, EP}(grids_c, bcs, extraps, anchors, grid_size)
    end
end

# ========================================
# AbstractAdjointND Interface
# ========================================

@inline _n_queries(adj::LinearAdjointND) = length(adj.anchors)
@inline _grid_size(adj::LinearAdjointND) = adj.grid_size

# Per-axis BCs from struct field. The ND adjoint protocol checks
# `_is_periodic_seam_folded(bcs[d])` (covers `:exclusive` and `:extended`)
# for output sizing and post-apply seam fold (`_adjoint_apply_exclusive_nd!`).
@inline _adjoint_bcs(adj::LinearAdjointND) = adj.bcs

@inline _adjoint_nd_apply!(f_bar, adj::LinearAdjointND, y_bar, ops) =
    _linear_adjoint_nd_apply!(f_bar, adj, y_bar, ops)

# ========================================
# Type Introspection
# ========================================

Base.ndims(::LinearAdjointND{Tg, N}) where {Tg, N} = N + 1
function Base.size(adj::LinearAdjointND)
    out_size = _adjoint_output_size(adj)
    return (out_size..., _n_queries(adj))
end
Base.size(adj::LinearAdjointND, d::Integer) = size(adj)[d]
