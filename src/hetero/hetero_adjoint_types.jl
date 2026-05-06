# ========================================
# ND Heterogeneous Adjoint Types
# ========================================
#
# Type definitions for the N-dimensional heterogeneous adjoint operator.
# HeteroAdjointND computes f̄ = Wᵀȳ where W is the implicit ND heterogeneous
# interpolation weight matrix with per-axis method dispatch.
#
# Uses compact mixed-radix storage: prod(sizes) partials instead of 2^N,
# where sizes[d] = _deriv_size(methods[d]) (2 for Cubic/Quadratic, 1 for Linear/Constant).
#
# Reuses _NDAdjointAnchor{Tg, N} (from core/nd_adjoint_scatter.jl) with
# 4-tuple weights per axis — Linear/Constant axes zero-pad unused entries.

# ========================================
# Per-Axis Anchor Weight Functions
# ========================================

"""
    _compute_nd_linear_anchor_weights(alpha, inv_h) -> (w0, w1, w2, w3)

Compute per-axis linear adjoint weights in 4-tuple format compatible with
`_NDAdjointAnchor`. Elements 3-4 (derivative weights) are always zero —
linear interpolation uses 2 nodal values (fL, fR) per axis.

Index mapping: `w[1 + corner + 2*deriv]`
  - (c=0, d=0) → 1-α,  (c=1, d=0) → α
  - (c=0, d=1) → 0,    (c=1, d=1) → 0  (no derivatives in linear)
"""
@inline function _compute_nd_linear_anchor_weights(alpha::Tg, inv_h::Tg) where {Tg}
    z = zero(Tg)
    w0 = (one(Tg) - alpha, alpha, z, z)
    w1 = (-inv_h, inv_h, z, z)
    w2 = (z, z, z, z)
    w3 = (z, z, z, z)
    return (w0, w1, w2, w3)
end

"""
    _compute_nd_constant_anchor_weights(dL, h, side) -> (w0, w1, w2, w3)

Compute per-axis constant adjoint weights in 4-tuple format compatible with
`_NDAdjointAnchor`. All derivative weights are zero — constant interpolation
has zero derivative everywhere.

The value weight selects exactly one node based on the side convention:
- LeftSide: always fL → w0 = (1, 0, 0, 0)
- RightSide: fL at grid point (dL==0), fR otherwise → w0 = (1-offset, offset, 0, 0)
- NearestSide: fL if dL ≤ h/2, fR otherwise → w0 = (1-offset, offset, 0, 0)
"""
@inline function _compute_nd_constant_anchor_weights(dL::Tg, h::Tg, side::AbstractSide) where {Tg}
    z = zero(Tg)
    offset = _compute_single_offset(side, h, dL)
    w_L = offset == 0 ? one(Tg) : z
    w_R = offset == 0 ? z : one(Tg)
    w0 = (w_L, w_R, z, z)
    return (w0, (z, z, z, z), (z, z, z, z), (z, z, z, z))
end

"""
    _compute_hetero_anchor_weights(t, h, inv_h, dL, method) -> (w0, w1, w2, w3)

Per-method dispatch for computing 4-tuple adjoint weights at a query point.
Dispatches to the type-specific weight function based on the axis's interpolation method.
"""
@inline _compute_hetero_anchor_weights(t::Tg, h::Tg, inv_h::Tg, ::Tg, ::CubicInterp) where {Tg} =
    _compute_nd_anchor_weights(t, h, inv_h)
@inline _compute_hetero_anchor_weights(t::Tg, h::Tg, inv_h::Tg, ::Tg, ::QuadraticInterp) where {Tg} =
    _compute_nd_quadratic_anchor_weights(t, h, inv_h)
@inline _compute_hetero_anchor_weights(::Tg, h::Tg, ::Tg, dL::Tg, m::ConstantInterp) where {Tg} =
    _compute_nd_constant_anchor_weights(dL, h, m.side)
@inline function _compute_hetero_anchor_weights(t::Tg, ::Tg, inv_h::Tg, ::Tg, ::LinearInterp) where {Tg}
    return _compute_nd_linear_anchor_weights(t, inv_h)
end

# ========================================
# HeteroAdjointND Struct
# ========================================

"""
    HeteroAdjointND{Tg, N, M, G, C, MC, BP, MBP}

Adjoint (transpose) operator for N-dimensional heterogeneous interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND interpolation weight matrix
with per-axis method dispatch.

Constructed from grids, query points, and per-axis methods (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`:  Grid float type (unconstrained — supports duck types like ForwardDiff.Dual)
- `N`:   Number of dimensions
- `M`:   Per-axis methods tuple type
- `G`:   Grid tuple type — wrapped axes carry cached `h`/`inv_h` directly;
         no separate `spacings` field needed.
- `C`:   Per-axis cache tuple (CubicSplineCache for cubic, Nothing for others)
- `MC`:  Per-axis mixed-partial cache tuple
- `BP`:  Per-axis adjoint-BC tuple. For cubic axes: `BCPair`/`PeriodicBC`;
         for quadratic axes: normalized `AbstractBC` (e.g. `Left`, `Right`, `MinCurvFit`);
         for non-derivative axes (Linear/Constant): `nothing` sentinel.
- `MBP`: Per-axis mixed-BC tuple, analogous to `BP` but for mixed-partial
         directions; `nothing` when no mixed-partial BC is required.

# Architecture — Mixed-Radix Compact Storage
Uses `prod(sizes)` partials instead of `2^N`, where `sizes[d] = _deriv_size(methods[d])`.
The scatter kernel writes to compact partial indices using mixed-radix decomposition.
The build adjoint step processes only derivative axes (sizes[d]=2) in reverse order,
dispatching to existing per-method 1D adjoint functions (Thomas solve for cubic,
slope recurrence for quadratic).

# Usage
```julia
adj = hetero_adjoint((x, y), (xq, yq); methods=(CubicInterp(), LinearInterp()))
f_bar = adj(y_bar)              # allocating
adj(f_bar, y_bar)               # in-place (zero-allocation)
```
"""
struct HeteroAdjointND{
        Tg,
        N,
        M <: Tuple{Vararg{AbstractInterpMethod, N}},
        G <: Tuple{Vararg{AbstractVector, N}},
        C,     # Heterogeneous cache tuple (CubicSplineCache or Nothing per axis)
        MC,    # Mixed-partial cache tuple
        BP,    # Per-axis BC placeholder tuple
        MBP,   # Per-axis mixed BC tuple
    } <: AbstractAdjointND{Tg, N}
    methods::M
    grids::G
    caches::C
    mixed_caches::MC
    bcs::BP
    mixed_bcs::MBP
    anchors::Vector{_NDAdjointAnchor{Tg, N}}
    grid_size::NTuple{N, Int}
    mincurv_Cs::NTuple{N, Tg}

    # Defensive copy of grids to prevent mutation safety issues
    # (matches LinearAdjointND / QuadraticAdjointND / ConstantAdjointND convention)
    function HeteroAdjointND{Tg, N, M, G, C, MC, BP, MBP}(
            methods, grids, caches, mixed_caches,
            bcs, mixed_bcs, anchors, grid_size, mincurv_Cs
        ) where {Tg, N, M, G, C, MC, BP, MBP}
        grids_c = map(copy, grids)
        return new{Tg, N, M, typeof(grids_c), C, MC, BP, MBP}(
            methods, grids_c, caches, mixed_caches,
            bcs, mixed_bcs, anchors, grid_size, mincurv_Cs
        )
    end
end

# ========================================
# AbstractAdjointND Interface
# ========================================

@inline _n_queries(adj::HeteroAdjointND) = length(adj.anchors)
@inline _grid_size(adj::HeteroAdjointND) = adj.grid_size

# For periodic finalization: return per-axis BC for derivative methods,
# nothing sentinel for non-derivative methods (never matches PeriodicBC{:exclusive}).
@inline _adjoint_bcs(adj::HeteroAdjointND) = adj.bcs

@inline _adjoint_nd_apply!(f_bar, adj::HeteroAdjointND, y_bar, ops) =
    _hetero_adjoint_nd_apply!(f_bar, adj, y_bar, ops)

# ========================================
# Type Introspection
# ========================================

Base.ndims(::HeteroAdjointND{Tg, N}) where {Tg, N} = N + 1
function Base.size(adj::HeteroAdjointND)
    out_size = _adjoint_output_size(adj)
    return (out_size..., _n_queries(adj))
end
Base.size(adj::HeteroAdjointND, d::Integer) = size(adj)[d]
