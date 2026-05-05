# ========================================
# QuadraticAdjointND: Types & Protocol Accessors
# ========================================
#
# Reuses _NDAdjointAnchor{Tg, N} from cubic — quadratic weights are stored
# in 4-tuple format with the 4th element (dfR) always zero.
# This enables direct reuse of _scatter_nd! and _scatter_nd_codegen.
#
# No caches (unlike CubicAdjointND) — quadratic recurrence is O(n), no LU.

# ========================================
# ND Quadratic Weight Computation
# ========================================

"""
    _compute_nd_quadratic_anchor_weights(t, h, inv_h) -> (w0, w1, w2, w3)

Compute per-axis quadratic adjoint weights in 4-tuple format compatible with
`_NDAdjointAnchor`. The 4th element (dfR weight) is always zero — quadratic
uses 3 nodal values (fL, fR, dfL) per axis, not 4.

Index mapping: `w[1 + corner + 2*deriv]`
  - (c=0, d=0) → w_fL,  (c=1, d=0) → w_fR
  - (c=0, d=1) → w_d,   (c=1, d=1) → 0 (no dfR in quadratic)
"""
@inline function _compute_nd_quadratic_anchor_weights(t::Tg, h::Tg, inv_h::Tg) where {Tg}
    w0_3, w1_3, w2_3 = _compute_quadratic_adjoint_weights(t, h, inv_h)
    z = zero(Tg)
    w0 = (w0_3[1], w0_3[2], w0_3[3], z)
    w1 = (w1_3[1], w1_3[2], w1_3[3], z)
    w2 = (w2_3[1], w2_3[2], w2_3[3], z)
    w3 = (z, z, z, z)  # EvalDeriv3 = 0 for quadratic
    return (w0, w1, w2, w3)
end

# ========================================
# QuadraticAdjointND Struct
# ========================================

"""
    QuadraticAdjointND{Tg, N, G, BP}

Adjoint (transpose) operator for N-dimensional quadratic spline interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND interpolation weight matrix.

# Type Parameters
- `Tg`: Grid type (unconstrained — supports duck types like ForwardDiff.Dual)
- `N`: Number of dimensions
- `G`: Grid tuple type — wrapped axes (`_CachedRange` / `_CachedVector`)
       carry cached `h`/`inv_h`; no separate `spacings` field needed.
- `BP`: Boundary condition tuple type. PeriodicBC is not supported
        (half-integer slope offset is incompatible with seam wrap, by design).

# Fields
- `mincurv_Cs`: Precomputed `inv(Σ inv_h)` per axis, avoiding O(n) recomputation
  per slice for MinCurvFit BCs. Unused (but harmless) for other BC types.

# Usage
```julia
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 15)
xq = rand(100)
yq = rand(100)

adj = quadratic_adjoint((x, y), (xq, yq))
f_bar = adj(y_bar)   # returns 20×15 matrix
```
"""
struct QuadraticAdjointND{
        Tg,
        N,
        G <: NTuple{N, AbstractVector{Tg}},
        BP <: NTuple{N, AbstractBC},
    } <: AbstractAdjointND{Tg, N}
    grids::G
    bcs::BP
    anchors::Vector{_NDAdjointAnchor{Tg, N}}
    grid_size::NTuple{N, Int}
    mincurv_Cs::NTuple{N, Tg}  # Precomputed inv(Σ inv_h) per axis; only used for MinCurvFit BCs

    # Inner ctor: ownership copy via wrapper-aware `_convert_copy`,
    # idempotent `_cache_axis` insurance for direct ctor calls.
    function QuadraticAdjointND(
            grids::Tuple{Vararg{AbstractVector{Tg}, N}}, bcs::BP,
            anchors::Vector{_NDAdjointAnchor{Tg, N}}, grid_size::NTuple{N, Int},
            mincurv_Cs::NTuple{N, Tg}
        ) where {
            Tg, N, BP <: NTuple{N, AbstractBC},
        }
        grids_c = map((g, bc) -> _convert_copy(_cache_axis(g, bc, Tg), Tg), grids, bcs)
        return new{Tg, N, typeof(grids_c), BP}(grids_c, bcs, anchors, grid_size, mincurv_Cs)
    end
end

# ========================================
# ND Adjoint Protocol Accessors
# ========================================

@inline _n_queries(adj::QuadraticAdjointND) = length(adj.anchors)
@inline _grid_size(adj::QuadraticAdjointND) = adj.grid_size
@inline _adjoint_bcs(adj::QuadraticAdjointND) = adj.bcs

@inline _adjoint_nd_apply!(f_bar, adj::QuadraticAdjointND, y_bar, ops) =
    _quadratic_adjoint_nd_apply!(f_bar, adj, y_bar, ops)

# ========================================
# Type Introspection
# ========================================

Base.ndims(::QuadraticAdjointND{Tg, N}) where {Tg, N} = N + 1
function Base.size(adj::QuadraticAdjointND)
    out_size = _adjoint_output_size(adj)
    return (out_size..., _n_queries(adj))
end
Base.size(adj::QuadraticAdjointND, d::Integer) = size(adj)[d]
