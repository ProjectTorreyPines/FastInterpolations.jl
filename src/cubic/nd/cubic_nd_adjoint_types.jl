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
Stores cell location and per-axis Hermite basis weights.

# Fields
- `indices`: Per-axis cell index (left node of containing interval)
- `weights`: Per-axis `(w_fL, w_fR, w_dyL, w_dyR)` Hermite weights

# Weight Convention
For normalized position `t = dL / h` on each axis:
- `w_fL  = h00(t)`     — weight on function value at left node
- `w_fR  = h01(t)`     — weight on function value at right node
- `w_dyL = h10(t) * h` — weight on derivative at left node (h-scaled)
- `w_dyR = h11(t) * h` — weight on derivative at right node (h-scaled)

The h-scaling on derivative weights matches `_hermite_kernel_1d` (cubic_nd_math.jl).
"""
struct _NDAdjointAnchor{Tg <: AbstractFloat, N}
    indices::NTuple{N, Int}
    weights::NTuple{N, NTuple{4, Tg}}
end

# ========================================
# ND Cubic Adjoint Operator
# ========================================

"""
    CubicAdjointND{Tg, N, G, S, C, B}

Adjoint (transpose) operator for N-dimensional cubic Hermite interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND cubic interpolation weight matrix.

Constructed from grids and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `N`:  Number of dimensions
- `G`:  Grid tuple type
- `S`:  Spacing tuple type
- `C`:  Cache tuple type (per-axis CubicSplineCache)
- `B`:  BC tuple type (per-axis boundary conditions)

# Architecture
Unlike `CubicInterpolantND` which precomputes all partial derivatives and discards
the Thomas LU factorizations, the adjoint **retains per-axis LU caches** because
it must solve transpose Thomas systems at every `adj(ȳ)` call.

# Usage
```julia
adj = cubic_adjoint((x, y), (xq, yq); bc=CubicFit())
f_bar = adj(y_bar)              # allocating
adj(f_bar, y_bar)               # in-place (zero-allocation)
```

# Mathematical Background
Forward: `y = W·f` where `W = Eval ∘ Build`
  - Build: per-axis `f → moments → derivatives` (involves A⁻¹·R per axis)
  - Eval: tensor-product Hermite collapse

Adjoint: `f̄ = Wᵀȳ = Buildᵀ · Evalᵀ · ȳ`
  - Evalᵀ: scatter ȳ to partials_bar via Hermite weight products
  - Buildᵀ: per-axis (reverse order) moments-to-deriv adjoint + Aᵀ\\ solve + Rᵀ scatter
"""
struct CubicAdjointND{
        Tg <: AbstractFloat,
        N,
        G <: NTuple{N, AbstractVector{Tg}},
        S <: NTuple{N, AbstractGridSpacing{Tg}},
        C <: NTuple{N, CubicSplineCache{Tg}},
        B <: NTuple{N, AbstractBC},
    } <: AbstractAdjoint{Tg}
    grids::G
    spacings::S
    caches::C
    bcs::B
    anchors::Vector{_NDAdjointAnchor{Tg, N}}
    grid_size::NTuple{N, Int}
end

# ========================================
# Type Introspection
# ========================================

Base.ndims(::CubicAdjointND{Tg, N}) where {Tg, N} = N
Base.size(adj::CubicAdjointND) = (adj.grid_size, length(adj.anchors))
Base.size(adj::CubicAdjointND, d::Integer) = size(adj)[d]
