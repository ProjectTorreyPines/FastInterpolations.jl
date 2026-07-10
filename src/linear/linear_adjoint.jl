# ========================================
# Linear Adjoint Operator (1D)
# ========================================
#
# Computes f̄ = Wᵀȳ where W is the linear interpolation weight matrix.
# Pure scatter operation (no solve step, unlike cubic).
#
# Reuses _LinearAnchoredQuery from linear_anchor.jl:
# - alpha  → EvalValue weights: (1-α, α)
# - inv_h  → EvalDeriv1 weights: (-inv_h, inv_h)
# - state  → OOB detection: IN_DOMAIN, OOB_LEFT, OOB_RIGHT
#
# 2nd+ derivatives of linear interpolation are identically zero.

# ========================================
# LinearAdjoint Type
# ========================================

"""
    LinearAdjoint{Tg, EP}

Adjoint (transpose) operator for 1D linear interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward linear interpolation weight matrix.

Constructed from grid and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector.

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `EP`: Extrapolation policy type (`NoExtrap`, `ExtendExtrap`, `ClampExtrap`, `FillExtrap`, `WrapExtrap`)

# Fields
- `anchors`: Pre-computed `_LinearAnchoredQuery` per query point
- `grid_size`: Number of grid points (for output allocation)
- `extrap`: Extrapolation policy (for compile-time OOB dispatch in scatter)

# Usage
```julia
x = collect(range(0, 1, 50))
xq = sort(rand(30))
f = randn(50)
y_bar = randn(30)

adj = linear_adjoint(x, xq)
f_bar = adj(y_bar)                      # value adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))    # 1st derivative adjoint
adj(f_bar, y_bar; deriv=DerivOp(1))     # in-place

# Dot-product identity: ⟨W·f, ȳ⟩ = ⟨f, Wᵀȳ⟩
itp = linear_interp(x, f)
@assert dot(itp.(xq), y_bar) ≈ dot(f, adj(y_bar))
```
"""
struct LinearAdjoint{Tg, BC <: AbstractBC, EP <: AbstractExtrap} <: AbstractAdjoint1D{Tg}
    # Coordinate type is grid-pinned (`Tc = Tg`), not the canonical `_coord_eltype(Tq, Tg)`:
    # the adjoint operates on baked coefficients, so AD-through-adjoint is unsupported. The
    # forward Dual-grid contract is satisfied independently.
    anchors::Vector{_LinearAnchoredQuery{Tg, Tg}}
    grid_size::Int  # internal length: n+1 for PeriodicBC{:exclusive}, n otherwise
    bc::BC
    extrap::EP
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================
# Callables (6 overloads), Base.size, Base.Matrix, and exclusive-periodic
# in-place seam fold are inherited from AbstractAdjoint1D via
# src/core/adjoint_protocol.jl.

@inline _n_queries(adj::LinearAdjoint) = length(adj.anchors)

@inline _adjoint_1d_apply!(f_bar, adj::LinearAdjoint, y_bar, deriv) =
    _linear_adjoint_apply!(f_bar, adj, y_bar, deriv)

# ========================================
# Core Apply Function
# ========================================

"""
    _linear_adjoint_apply!(f_bar, adj, y_bar, deriv)

Core scatter: accumulate `Wᵀȳ` into `f_bar` using pre-baked anchor weights.
Dispatches on `deriv` and `adj.extrap` for compile-time OOB handling.
"""
@inline function _linear_adjoint_apply!(
        f_bar::AbstractVector,
        adj::LinearAdjoint,
        y_bar,
        deriv::DerivOp
    )
    _scatter_linear_adjoint!(f_bar, adj.anchors, y_bar, deriv, adj.extrap)
    return nothing
end

# ========================================
# Scatter Functions
# ========================================

# ── EvalValue scatter ─────────────────────────────────────────────────────
# Forward: y_q = (1-α)·f[i] + α·f[i+1]
# Adjoint: f̄[i] += (1-α)·ȳ_q, f̄[i+1] += α·ȳ_q

@inline function _scatter_linear_adjoint!(
        f_bar::AbstractVector,
        anchors::Vector{<:_LinearAnchoredQuery},
        y_bar,
        ::EvalValue,
        extrap::AbstractExtrap
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        _is_oob_skip(aq.state, extrap) && continue
        yb = y_bar[q]
        f_bar[aq.idxL] += (one(aq.alpha) - aq.alpha) * yb
        f_bar[aq.idxR] += aq.alpha * yb
    end
    return nothing
end

# ── EvalDeriv1 scatter ────────────────────────────────────────────────────
# Forward: dy/dx = (f[i+1] - f[i]) / h = f[i+1]·inv_h - f[i]·inv_h
# Adjoint: f̄[i] += (-inv_h)·ȳ_q, f̄[i+1] += inv_h·ȳ_q

@inline function _scatter_linear_adjoint!(
        f_bar::AbstractVector,
        anchors::Vector{<:_LinearAnchoredQuery},
        y_bar,
        ::EvalDeriv1,
        extrap::AbstractExtrap
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        _is_oob_skip_deriv(aq.state, extrap) && continue
        yb = y_bar[q]
        f_bar[aq.idxL] -= aq.inv_h * yb
        f_bar[aq.idxR] += aq.inv_h * yb
    end
    return nothing
end

# ── EvalDeriv2 / EvalDeriv3 scatter ──────────────────────────────────────
# Linear interpolation has zero 2nd+ derivative → no scatter needed

@inline function _scatter_linear_adjoint!(
        ::AbstractVector,
        ::Vector{<:_LinearAnchoredQuery},
        ::Any,
        ::Union{EvalDeriv2, EvalDeriv3},
        ::AbstractExtrap
    )
    return nothing
end

# Generic fallback: 2nd+ derivative of linear is zero → no scatter
@inline function _scatter_linear_adjoint!(
        ::AbstractVector, ::Vector{<:_LinearAnchoredQuery}, ::Any,
        ::DerivOp{N}, ::AbstractExtrap
    ) where {N}
    return nothing
end

# OOB skip helpers (_is_oob_skip, _is_oob_skip_deriv) are defined in
# src/core/adjoint_protocol.jl and shared by all adjoint types.

# ========================================
# ClampExtrap / FillExtrap OOB Side Fixup
# ========================================

"""
    _bake_linear_clampfill_anchors(x, xq) -> Vector{_LinearAnchoredQuery}

Single-pass ClampExtrap/FillExtrap adjoint anchor builder.

Each query is clamped to the *actual* grid endpoints (`_clamp_to_grid`) for
valid boundary-cell geometry, anchored, then genuinely-OOB queries get their
side flag restored from the widened (`_oob_state`) classification so scatter
skips (FillExtrap) or keeps (ClampExtrap) the boundary weight per extrap. An
in-domain or endpoint-sliver query keeps the anchor as built.

Fuses the former clamp-broadcast + `_anchor_query` + state-fixup three passes
into one loop, dropping the transient clamped-query array. Construction-time
only; the apply path consumes the baked anchors unchanged.
"""
function _bake_linear_clampfill_anchors(
        x::AbstractVector{Tg},
        xq::AbstractVector{Tq},
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {Tg, Tq <: Real, P <: Searcher}
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)
    output = Vector{_LinearAnchoredQuery{Tg, promote_type(Tq, Tg)}}(undef, length(xq))
    @inbounds for k in eachindex(xq)
        xq_raw = xq[k]
        aq = _linear_anchor_query_impl(x, _clamp_to_grid(xq_raw, x), false, searcher_resolved)
        state = _oob_state(x, xq_raw)
        output[k] = state == IN_DOMAIN ? aq :
            typeof(aq)(aq.interval, aq.xq, state, aq.xL, aq.h, aq.inv_h, aq.alpha)
    end
    return output
end

# ========================================
# Constructor
# ========================================

"""
    linear_adjoint(x, x_query; extrap=NoExtrap()) -> LinearAdjoint

Create a linear interpolation adjoint operator (query-baked, data-free).

Computes `f̄ = Wᵀȳ` where `W` is the forward linear interpolation weight matrix.
Maps query-space sensitivities back to grid-space sensitivities.

# Arguments
- `x::AbstractVector`: Grid points (must be sorted)
- `x_query::AbstractVector`: Query points (baked into the operator)
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`).
  For `WrapExtrap()`, queries are wrapped to the domain before anchoring.
  For `ClampExtrap()`/`FillExtrap()`, OOB queries use boundary intervals with
  correct `side` flags for adjoint scatter skipping.

# Example
```julia
using LinearAlgebra
x = collect(range(0, 1, 50))
xq = sort(rand(30))
f = randn(50)
y_bar = randn(30)

itp = linear_interp(x, f)
adj = linear_adjoint(x, xq)
f_bar = adj(y_bar)

# Dot-product identity: ⟨W·f, ȳ⟩ = ⟨f, Wᵀȳ⟩
@assert dot(itp.(xq), y_bar) ≈ dot(f, adj(y_bar))
```

# Notes
- No boundary conditions needed (linear interpolation has none).
- No caching infrastructure (no tridiagonal solve).
- 2nd+ derivative adjoints always return zero.
- `NoExtrap` validates all queries are in-domain at construction time.
"""
function linear_adjoint(
        x::AbstractVector,
        x_query::AbstractVector;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
    )
    x_p, xq_p, Tg = _promote_adjoint_inputs(x, x_query)

    length(x_p) >= 2 || _throw_adjoint_grid_too_small(length(x_p))

    # BC-aware axis wrap: `:exclusive` periodic → `_ExclusivePeriodicAxis` with
    # logical length n+1 (virtual seam endpoint `inner[1] + period`). Anchors
    # at the seam cell store interval = (n, n+1); the protocol's exclusive-
    # periodic in-place callable folds f_work[1] += f_work[n+1] before trim.
    x_axis = _cache_axis(x_p, bc, Tg)
    # Periodic BCs auto-promote `extrap` to `WrapExtrap` against the wrapped axis;
    # non-periodic BCs are passthrough.
    extrap_eff = _resolve_extrap(extrap, bc, x_axis)

    # NoExtrap: validate all queries in-domain (uses x_axis bounds, which include
    # the virtual seam endpoint for `:exclusive`). Use primal for Dual grid boundaries.
    if extrap_eff isa NoExtrap
        _validate_domain(x_axis, xq_p)
    end

    # Build anchored queries with extrap-specific preprocessing
    wrap = extrap_eff isa WrapExtrap
    if extrap_eff isa _ClampOrFill
        # OOB queries get actual-endpoint geometry; the widened (`_oob_state`)
        # classification restores side flags. Single fused pass (no temp array).
        anchors = _bake_linear_clampfill_anchors(x_axis, xq_p)
    else
        # ExtendExtrap: OOB uses boundary interval with extrapolated alpha (correct)
        # WrapExtrap: wraps to domain (correct; covers periodic auto-promotion)
        # NoExtrap: already validated in-domain above
        anchors = _anchor_query(x_axis, xq_p, Val(:linear), wrap)
    end

    return LinearAdjoint{Tg, typeof(bc), typeof(extrap_eff)}(
        anchors, length(x_axis), bc, extrap_eff
    )
end

# Scalar query convenience
function linear_adjoint(
        x::AbstractVector,
        x_query::Real;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
    )
    return linear_adjoint(x, [x_query]; bc = bc, extrap = extrap)
end

# Matrix materialization inherited from AbstractAdjoint (adjoint_protocol.jl)
