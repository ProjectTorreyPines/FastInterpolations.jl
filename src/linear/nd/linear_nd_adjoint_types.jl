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
struct _LinearNDAdjointAnchor{Tg <: AbstractFloat, N}
    indices::NTuple{N, Int}
    w_value::NTuple{N, NTuple{2, Tg}}
    w_deriv::NTuple{N, NTuple{2, Tg}}
end

# ========================================
# ND Linear Adjoint Operator
# ========================================

"""
    LinearAdjointND{Tg, N, G, S, EP}

Adjoint (transpose) operator for N-dimensional linear interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward ND linear interpolation weight matrix.

Constructed from grids and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`:  Grid float type (Float32 or Float64)
- `N`:   Number of dimensions
- `G`:   Grid tuple type
- `S`:   Spacing tuple type
- `EP`:  Extrapolation tuple type

# Architecture
Pure scatter operator — no caches, no boundary conditions, no tridiagonal solve.
The only per-axis state is pre-baked weights in the anchors.

# Usage
```julia
adj = linear_adjoint((x, y), (xq, yq))
f_bar = adj(y_bar)              # allocating
adj(f_bar, y_bar)               # in-place (zero-allocation)
```
"""
struct LinearAdjointND{
        Tg <: AbstractFloat,
        N,
        G <: NTuple{N, AbstractVector{Tg}},
        S <: NTuple{N, AbstractGridSpacing{Tg}},
        EP <: Tuple{Vararg{AbstractExtrap, N}},
    } <: AbstractAdjointND{Tg, N}
    grids::G
    spacings::S
    extraps::EP
    anchors::Vector{_LinearNDAdjointAnchor{Tg, N}}
    grid_size::NTuple{N, Int}
end

# ========================================
# AbstractAdjointND Interface
# ========================================

@inline _n_queries(adj::LinearAdjointND) = length(adj.anchors)
@inline _grid_size(adj::LinearAdjointND) = adj.grid_size

# Linear has no periodic BCs — return dummy non-periodic tuple
# (protocol never hits periodic path for linear)
@inline _adjoint_bcs(adj::LinearAdjointND{Tg, N}) where {Tg, N} =
    ntuple(_ -> NoExtrap(), Val(N))

@inline _adjoint_nd_apply!(f_bar, adj::LinearAdjointND, y_bar, ops) =
    _linear_adjoint_nd_apply!(f_bar, adj, y_bar, ops)

# ========================================
# Type Introspection
# ========================================

Base.ndims(::LinearAdjointND{Tg, N}) where {Tg, N} = N + 1
function Base.size(adj::LinearAdjointND{Tg, N}) where {Tg, N}
    out_size = _adjoint_output_size(adj)
    return (out_size..., _n_queries(adj))
end
Base.size(adj::LinearAdjointND, d::Integer) = size(adj)[d]

# ========================================
# Deriv zero-fill trait
# ========================================

# Linear has zero 2nd+ derivative — same check as forward eval
@inline _deriv_zero_fill(::LinearAdjointND, ops::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N} =
    _has_second_or_higher_derivative(ops, Val(N))
