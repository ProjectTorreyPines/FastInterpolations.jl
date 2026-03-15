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
#
# _NDAdjointAnchor{Tg, N} is defined in core/nd_adjoint_scatter.jl (shared with quadratic)

# ========================================
# ND Cubic Adjoint Operator
# ========================================

"""
    CubicAdjointND{Tg, N, S, C, MC, BP, MBP}

Adjoint (transpose) operator for N-dimensional cubic Hermite interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND cubic interpolation weight matrix.

Constructed from grids and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`:  Grid float type (Float32 or Float64)
- `N`:   Number of dimensions
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

PolyFit stencil coefficients are computed on the fly from `BCPair` + `caches[d].x` (O(D) per
axis, negligible), matching the forward `compute_rhs!` pattern. Grids are NOT stored
separately — each per-axis `CubicSplineCache` already owns a mutation-safe copy of its grid
via `cache.x` (inner constructor `copy()` + `typeof(xc)` rebinding).

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
        S <: NTuple{N, AbstractGridSpacing{Tg}},
        C <: NTuple{N, CubicSplineCache{Tg}},
        MC <: NTuple{N, CubicSplineCache{Tg}},
        BP <: NTuple{N, Union{BCPair, PeriodicBC}},
        MBP <: NTuple{N, Union{BCPair, PeriodicBC}},
    } <: AbstractAdjointND{Tg, N}
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
