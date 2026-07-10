# ========================================
# Constant Adjoint Operator (1D)
# ========================================
#
# Computes f̄ = Wᵀȳ where W is the constant interpolation weight matrix.
# Pure single-point scatter (simpler than linear's 2-point scatter).
#
# Reuses _ConstantAnchoredQuery from constant_anchor.jl:
# - idxL  → left cell index (used directly; this path never sees seam pairs)
# - idxR  → right cell index (unused here; present on the shared struct for periodic-exclusive callers)
# - h     → interval width (for side-offset computation)
# - dL    → offset from left boundary (for side-offset computation)
# - state → OOB detection: IN_DOMAIN, OOB_LEFT, OOB_RIGHT
# - xq    → query point (for right-boundary special case)
#
# All derivatives of constant interpolation are identically zero.

# ========================================
# ConstantAdjoint Type
# ========================================

"""
    ConstantAdjoint{Tg, Tq, BC, SD, EP, I}

Adjoint (transpose) operator for 1D constant interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward constant interpolation weight matrix.

Constructed from grid and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`: Grid type (unconstrained — supports duck types like ForwardDiff.Dual)
- `Tq`: Anchor query coordinate type, `promote_type(Tg, eltype(x_query))`
- `BC`: Boundary condition type
- `SD`: Side selection mode (`NearestSide`, `LeftSide`, `RightSide`)
- `EP`: Extrapolation policy type (`NoExtrap`, `ExtendExtrap`, `ClampExtrap`, `FillExtrap`, `WrapExtrap`)
- `I <: _AbstractIndices{2}`: Interval representation — `_ContiguousIndices{2}` (one Int) for
  ordinary grids, `_ExplicitIndices{2}` (two Int) for periodic-exclusive seam cells

# Fields
- `anchors`: Pre-computed `_ConstantAnchoredQuery` per query point
- `grid_size`: Number of grid points (for output allocation)
- `x_hi`: Right domain boundary (for right-boundary special case)
- `side`: Side selection mode (for compile-time offset dispatch)
- `extrap`: Extrapolation policy (for compile-time OOB dispatch in scatter)

# Usage
```julia
x = collect(range(0, 1, 50))
xq = sort(rand(30))
f = randn(50)
y_bar = randn(30)

adj = constant_adjoint(x, xq; side=NearestSide())
f_bar = adj(y_bar)                      # value adjoint
adj(f_bar, y_bar)                       # in-place

# Dot-product identity: ⟨W·f, ȳ⟩ = ⟨f, Wᵀȳ⟩
itp = constant_interp(x, f; side=NearestSide())
@assert dot(itp.(xq), y_bar) ≈ dot(f, adj(y_bar))
```
"""
struct ConstantAdjoint{Tg, Tq, BC <: AbstractBC, SD <: AbstractSide, EP <: AbstractExtrap, I <: _AbstractIndices{2}} <: AbstractAdjoint1D{Tg}
    anchors::Vector{_ConstantAnchoredQuery{Tg, Tq, I}}
    grid_size::Int  # internal length: n+1 for PeriodicBC{:exclusive}, n otherwise
    x_hi::Tg
    bc::BC
    side::SD
    extrap::EP
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================
# Callables (6 overloads), Base.size, Base.Matrix, and exclusive-periodic
# in-place seam fold are inherited from AbstractAdjoint1D via
# src/core/adjoint_protocol.jl.

@inline _n_queries(adj::ConstantAdjoint) = length(adj.anchors)

@inline _adjoint_1d_apply!(f_bar, adj::ConstantAdjoint, y_bar, deriv) =
    _constant_adjoint_apply!(f_bar, adj, y_bar, deriv)

# ========================================
# Core Apply Function
# ========================================

"""
    _constant_adjoint_apply!(f_bar, adj, y_bar, deriv)

Core scatter: accumulate `Wᵀȳ` into `f_bar` using pre-baked anchor data.
Dispatches on `deriv` for compile-time derivative handling.
"""
@inline function _constant_adjoint_apply!(
        f_bar::AbstractVector,
        adj::ConstantAdjoint,
        y_bar,
        deriv::DerivOp
    )
    _scatter_constant_adjoint!(f_bar, adj.anchors, y_bar, deriv, adj.side, adj.extrap, adj.grid_size, adj.x_hi)
    return nothing
end

# ========================================
# Scatter Functions
# ========================================

# ── EvalValue scatter ─────────────────────────────────────────────────────
# Forward: y_q = data[idx + offset]  (offset from side mode)
# Adjoint: f̄[idx + offset] += ȳ_q

@inline function _scatter_constant_adjoint!(
        f_bar::AbstractVector,
        anchors::Vector{<:_ConstantAnchoredQuery},
        y_bar,
        ::EvalValue,
        side::AbstractSide,
        extrap::AbstractExtrap,
        grid_size::Int,
        x_hi
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        _is_oob_skip(aq.state, extrap) && continue
        # Right-boundary special case: xq == x_max → always f_bar[end]
        if aq.xq == x_hi
            f_bar[grid_size] += y_bar[q]
        else
            offset = _compute_single_offset(side, aq.h, aq.dL)
            f_bar[aq.idxL + offset] += y_bar[q]
        end
    end
    return nothing
end

# ── All derivative scatters ───────────────────────────────────────────────
# Constant interpolation has zero derivative at all orders → no scatter needed

@inline function _scatter_constant_adjoint!(
        ::AbstractVector,
        ::Vector{<:_ConstantAnchoredQuery},
        ::Any,
        ::Union{EvalDeriv1, EvalDeriv2, EvalDeriv3},
        ::AbstractSide,
        ::AbstractExtrap,
        ::Int,
        ::Any
    )
    return nothing
end

# Generic fallback: any derivative of constant is zero → no scatter
@inline function _scatter_constant_adjoint!(
        ::AbstractVector, ::Vector{<:_ConstantAnchoredQuery}, ::Any,
        ::DerivOp{N}, ::AbstractSide, ::AbstractExtrap, ::Int, ::Any
    ) where {N}
    return nothing
end

# ========================================
# ClampExtrap / FillExtrap OOB Side Fixup
# ========================================

"""
    _bake_constant_clampfill_anchors(x, xq) -> Vector{_ConstantAnchoredQuery}

Single-pass ClampExtrap/FillExtrap/ExtendExtrap adjoint anchor builder
(`ExtendExtrap == ClampExtrap` for constant interp, slope 0).

Clamps each query to the actual grid endpoints (`_clamp_to_grid`) for valid
boundary geometry, anchors it, then restores the OOB side flag from the widened
(`_oob_state`) classification for genuinely-OOB queries (scatter skips/keeps per
extrap). In-domain and endpoint-sliver queries keep the anchor as built.

Fuses the former clamp-broadcast + `_anchor_query` + state-fixup three passes
into one loop — no transient clamped-query array. Construction-time only.
"""
function _bake_constant_clampfill_anchors(
        x::AbstractVector{Tg},
        xq::AbstractVector{Tq},
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {Tg, Tq <: Real, P <: Searcher}
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)
    output = Vector{_ConstantAnchoredQuery{Tg, promote_type(Tg, Tq), _interval_type(x)}}(undef, length(xq))
    @inbounds for k in eachindex(xq)
        xq_raw = xq[k]
        aq = _constant_anchor_query_impl(x, _clamp_to_grid(xq_raw, x), false, searcher_resolved)
        state = _oob_state(x, xq_raw)
        output[k] = state == IN_DOMAIN ? aq :
            typeof(aq)(aq.interval, aq.xq, state, aq.h, aq.dL)
    end
    return output
end

# ========================================
# Constructor
# ========================================

"""
    constant_adjoint(x, x_query; side=NearestSide(), extrap=NoExtrap()) -> ConstantAdjoint

Create a constant interpolation adjoint operator (query-baked, data-free).

Computes `f̄ = Wᵀȳ` where `W` is the forward constant interpolation weight matrix.
Maps query-space sensitivities back to grid-space sensitivities.

# Arguments
- `x::AbstractVector`: Grid points (must be sorted)
- `x_query::AbstractVector`: Query points (baked into the operator)
- `side::AbstractSide`: Side selection mode (default: `NearestSide()`).
  Controls which grid point is selected per interval.
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`).

# Example
```julia
using LinearAlgebra
x = collect(range(0, 1, 50))
xq = sort(rand(30))
f = randn(50)
y_bar = randn(30)

itp = constant_interp(x, f; side=NearestSide())
adj = constant_adjoint(x, xq; side=NearestSide())
f_bar = adj(y_bar)

# Dot-product identity: ⟨W·f, ȳ⟩ = ⟨f, Wᵀȳ⟩
@assert dot(itp.(xq), y_bar) ≈ dot(f, adj(y_bar))
```

# Notes
- All derivative adjoints always return zero (constant has no slope).
- `NoExtrap` validates all queries are in-domain at construction time.
"""
function constant_adjoint(
        x::AbstractVector{Tg},
        x_query::AbstractVector;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
    ) where {Tg}
    # Grid stays raw `Tg` (no `_promote_adjoint_inputs` Float widening).
    # Adjoint buffer eltype comes from the protocol's `_promote_eltype`.
    x_p = x
    xq_p = _promote_query_typed(x_query, Tg)

    length(x_p) >= 2 || _throw_adjoint_grid_too_small(length(x_p))

    # BC-aware axis wrap: `:exclusive` periodic → `_ExclusivePeriodicAxis` with
    # logical length n+1. Anchors at the seam cell store interval = (n, n+1);
    # the protocol's exclusive-periodic in-place callable folds f_work[1] +=
    # f_work[n+1] before trim. Right-boundary special case `xq == x_hi` also
    # writes to f_bar[n+1] (the virtual seam endpoint), so it folds correctly.
    x_axis = _cache_axis(x_p, bc, Tg)
    extrap_eff = _resolve_extrap(extrap, bc, x_axis)
    x_hi = last(x_axis)

    # NoExtrap: validate all queries in-domain (uses x_axis bounds, which include
    # the virtual seam endpoint for `:exclusive`).
    if extrap_eff isa NoExtrap
        _validate_domain(x_axis, xq_p)
    end

    # Build anchored queries with extrap-specific preprocessing
    wrap = extrap_eff isa WrapExtrap
    if extrap_eff isa _ClampOrFill || extrap_eff isa ExtendExtrap
        # For constant interp, ExtendExtrap == ClampExtrap (slope=0). OOB queries
        # get actual-endpoint geometry; the widened (`_oob_state`) classification
        # restores side flags. Single fused pass (no temp array).
        anchors = _bake_constant_clampfill_anchors(x_axis, xq_p)
    else
        # WrapExtrap: wraps to domain (covers periodic auto-promotion).
        # NoExtrap: already validated in-domain above.
        anchors = _anchor_query(x_axis, xq_p, Val(:constant), wrap)
    end

    Tq = eltype(anchors).parameters[2]
    return ConstantAdjoint{Tg, Tq, typeof(bc), typeof(side), typeof(extrap_eff), _interval_type(x_axis)}(
        anchors, length(x_axis), x_hi, bc, side, extrap_eff
    )
end

# Scalar query convenience
function constant_adjoint(
        x::AbstractVector,
        x_query::Real;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
    )
    return constant_adjoint(x, [x_query]; bc = bc, side = side, extrap = extrap)
end

# Matrix materialization inherited from AbstractAdjoint (adjoint_protocol.jl)
