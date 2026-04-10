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
struct LinearAdjoint{Tg, EP <: AbstractExtrap} <: AbstractAdjoint1D{Tg}
    anchors::Vector{_LinearAnchoredQuery{Tg, Tg}}
    grid_size::Int
    extrap::EP
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================
# Callables (6 overloads), Base.size, and Base.Matrix are inherited
# from AbstractAdjoint via src/core/adjoint_protocol.jl.

@inline _n_queries(adj::LinearAdjoint) = length(adj.anchors)
@inline _adjoint_output_length(adj::LinearAdjoint) = adj.grid_size

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
        f_bar[aq.idx] += (one(aq.alpha) - aq.alpha) * yb
        f_bar[aq.idx + 1] += aq.alpha * yb
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
        f_bar[aq.idx] -= aq.inv_h * yb
        f_bar[aq.idx + 1] += aq.inv_h * yb
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
Restore `state` flags for anchors built with clamped query positions.

When ClampExtrap/FillExtrap queries are clamped before anchoring, the anchor
gets `state=IN_DOMAIN` (inside). This restores the correct OOB state flag based on the
original query position, so scatter can skip OOB contributions.
"""
function _fixup_linear_anchor_state!(
        anchors::Vector{_LinearAnchoredQuery{Tg, Tg}},
        xq_original::AbstractVector{Tg},
        x_lo::Tg, x_hi::Tg
    ) where {Tg}
    @inbounds for i in eachindex(anchors)
        xq_i = xq_original[i]
        (x_lo <= xq_i <= x_hi) && continue
        state = xq_i < x_lo ? OOB_LEFT : OOB_RIGHT
        aq = anchors[i]
        anchors[i] = _LinearAnchoredQuery{Tg, Tg}(
            aq.idx, aq.xq, state, aq.xL, aq.h, aq.inv_h, aq.alpha
        )
    end
    return nothing
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
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    Tg = _promote_grid_float(eltype(x), eltype(x_query))
    x_p = _to_float(x, Tg)
    # Query normalization: convert to the grid's float base type, not to Tg itself.
    # When Tg is a duck type (e.g. Dual), queries stay plain Float — they don't
    # carry grid-parameter derivatives. The anchor outer constructor handles
    # widening xq to match alpha's arithmetic type.
    Tq_float = Tg <: AbstractFloat ? Tg : float(eltype(x_query))
    xq_p = _to_float(x_query, Tq_float)

    length(x_p) >= 2 || _throw_adjoint_grid_too_small(length(x_p))

    # NoExtrap: validate all queries in-domain
    if extrap isa NoExtrap
        x_lo, x_hi = first(x_p), last(x_p)
        @inbounds for i in eachindex(xq_p)
            xq_i = xq_p[i]
            (x_lo <= xq_i <= x_hi) || throw(
                DomainError(xq_i, "query point outside domain [$x_lo, $x_hi]")
            )
        end
    end

    # Build anchored queries with extrap-specific preprocessing
    wrap = extrap isa WrapExtrap
    if extrap isa _ClampOrFill
        # Clamp OOB queries to boundary for correct anchor weights (alpha ∈ [0,1]).
        # Then restore side flags so scatter can skip OOB contributions.
        x_lo, x_hi = first(x_p), last(x_p)
        xq_clamped = clamp.(xq_p, x_lo, x_hi)
        anchors = _anchor_query(x_p, xq_clamped, Val(:linear), false)
        _fixup_linear_anchor_state!(anchors, xq_p, x_lo, x_hi)
    else
        # ExtendExtrap: OOB uses boundary interval with extrapolated alpha (correct)
        # WrapExtrap: wraps to domain (correct)
        # NoExtrap: already validated in-domain above
        anchors = _anchor_query(x_p, xq_p, Val(:linear), wrap)
    end

    return LinearAdjoint{Tg, typeof(extrap)}(anchors, length(x_p), extrap)
end

# Scalar query convenience
function linear_adjoint(
        x::AbstractVector,
        x_query::Real;
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    return linear_adjoint(x, [x_query]; extrap = extrap)
end

# Matrix materialization inherited from AbstractAdjoint (adjoint_protocol.jl)
