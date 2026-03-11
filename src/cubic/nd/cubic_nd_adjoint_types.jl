# ========================================
# ND Cubic Adjoint Types
# ========================================
#
# Type definitions for the N-dimensional cubic adjoint operator.
# CubicAdjointND computes f̄ = Wᵀȳ where W is the implicit ND cubic interpolation matrix.
#
# Key difference from CubicInterpolantND:
# - Forward discards LU caches after precomputing nodal_derivs (all partials baked)
# - Adjoint RETAINS per-axis LU caches (transpose Thomas solve needed at every call)

# ========================================
# Per-Query Anchor (Baked Hermite Weights)
# ========================================

"""
    _NDAdjointAnchor{Tg, N}

Precomputed per-query data for ND cubic adjoint scatter.
Stores cell location and per-axis Hermite basis weights for all derivative orders.

All 4 weight sets are pre-baked at construction time, matching the 1D `_CubicAnchoredQuery`
pattern. The adjoint is constructed once and called repeatedly with different `ȳ` vectors,
so pre-computing is optimal (pay once, reuse many times).

# Fields
- `indices`: Per-axis cell index (left node of containing interval)
- `w0`: EvalValue weights per axis — `(h00, h01, h10·h, h11·h)`
- `w1`: EvalDeriv1 weights per axis — `(h00'·inv_h, h01'·inv_h, h10', h11')`
- `w2`: EvalDeriv2 weights per axis — `(h00''·inv_h², h01''·inv_h², h10''·inv_h, h11''·inv_h)`
- `w3`: EvalDeriv3 weights per axis — `(12·inv_h³, -12·inv_h³, 6·inv_h², 6·inv_h²)`

# Weight Convention
For each axis d with normalized position `t = dL / h`:
- Each `wk[d]` is a 4-tuple `(w_fL, w_fR, w_dyL, w_dyR)` for the k-th derivative
- Forward: `dᵏP/dxᵏ = Σ_j wk_j · (fL, fR, dyL, dyR)_j` (tensor-product per axis)
- All 4 weights are non-zero for all k (ND Hermite form, unlike 1D moment form)
"""
struct _NDAdjointAnchor{Tg <: AbstractFloat, N}
    indices::NTuple{N, Int}
    w0::NTuple{N, NTuple{4, Tg}}
    w1::NTuple{N, NTuple{4, Tg}}
    w2::NTuple{N, NTuple{4, Tg}}
    w3::NTuple{N, NTuple{4, Tg}}
end

# ========================================
# ND Cubic Adjoint Operator
# ========================================

"""
    CubicAdjointND{Tg, N, G, S, C, MC, BP, MBP}

Adjoint (transpose) operator for N-dimensional cubic Hermite interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND cubic interpolation weight matrix.

Constructed from grids and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`:  Grid float type (Float32 or Float64)
- `N`:   Number of dimensions
- `G`:   Grid tuple type
- `S`:   Spacing tuple type
- `C`:   Cache tuple type for user BC (per-axis CubicSplineCache)
- `MC`:  Cache tuple type for mixed-partial BC (CubicFit internally)
- `BP`:  User BC pairs — `BCPair` for non-periodic, `PeriodicBC` for periodic axes
- `MBP`: Mixed-partial BC pairs — same convention as `BP`

# Architecture — Dual Cache/BC Structure
The adjoint retains per-axis Thomas LU caches for transpose solves at every `adj(ȳ)`
call. The ND build pipeline processes `(d, p_src)` pairs, where the BC used depends
on whether the partial is pure or mixed:

- **Pure derivatives** (`p_src == 1`): use the user's BC → `caches` + `bcs`
- **Mixed partials** (`p_src > 1`): internally always use CubicFit →
  `mixed_caches` + `mixed_bcs`

For **periodic axes**, `_get_effective_bc(PeriodicBC(), p_src, grid)` returns `PeriodicBC()`
for ALL `p_src`, so `caches[d] == mixed_caches[d]` (same pool entry) and both bcs
entries are `PeriodicBC`. The periodic build adjoint uses Sherman-Morrison transpose
solves instead of standard Thomas.

`CubicSplineCache` is parameterized on `BCPair{L,R}` or `PeriodicData{Tg}`, so `C` and
`MC` are generally different types (e.g., `BCPair{Deriv2,Deriv2}` vs `BCPair{Deriv1,Deriv1}`).
When the user's BC is already CubicFit, `C == MC` and `BP == MBP`, so Julia
optimizes the branch away entirely.

PolyFit stencil coefficients are computed on the fly from `BCPair` + grid (O(D) per
axis, negligible), matching the forward `compute_rhs!` pattern.

# Usage
```julia
adj = cubic_adjoint((x, y), (xq, yq); bc=CubicFit())
f_bar = adj(y_bar)              # allocating
adj(f_bar, y_bar)               # in-place (zero-allocation)

# Periodic
adj = cubic_adjoint((x, y), (xq, yq); bc=PeriodicBC())
f_bar = adj(y_bar)              # inclusive periodic

# Mixed: periodic × non-periodic
adj = cubic_adjoint((x, y), (xq, yq); bc=(PeriodicBC(), CubicFit()))
```
"""
struct CubicAdjointND{
        Tg <: AbstractFloat,
        N,
        G <: NTuple{N, AbstractVector{Tg}},
        S <: NTuple{N, AbstractGridSpacing{Tg}},
        C <: NTuple{N, CubicSplineCache{Tg}},
        MC <: NTuple{N, CubicSplineCache{Tg}},
        BP <: NTuple{N, Union{BCPair, PeriodicBC}},
        MBP <: NTuple{N, Union{BCPair, PeriodicBC}},
    } <: AbstractAdjointND{Tg, N}
    grids::G
    spacings::S
    caches::C
    mixed_caches::MC
    bcs::BP
    mixed_bcs::MBP
    anchors::Vector{_NDAdjointAnchor{Tg, N}}
    grid_size::NTuple{N, Int}
end

# ========================================
# AbstractAdjointND Interface
# ========================================

@inline _n_queries(adj::CubicAdjointND) = length(adj.anchors)
@inline _grid_size(adj::CubicAdjointND) = adj.grid_size
@inline _adjoint_bcs(adj::CubicAdjointND) = adj.bcs
@inline _adjoint_nd_apply!(f_bar, adj::CubicAdjointND, y_bar, ops) =
    _cubic_adjoint_nd_apply!(f_bar, adj, y_bar, ops)

# ========================================
# Type Introspection
# ========================================

Base.ndims(::CubicAdjointND{Tg, N}) where {Tg, N} = N + 1
function Base.size(adj::CubicAdjointND{Tg, N}) where {Tg, N}
    out_size = _adjoint_output_size(adj)
    return (out_size..., _n_queries(adj))
end
Base.size(adj::CubicAdjointND, d::Integer) = size(adj)[d]
