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
# - side   → OOB detection: 0x00=inside, 0x01=below, 0x02=above
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
struct LinearAdjoint{Tg <: AbstractFloat, EP <: AbstractExtrap} <: AbstractAdjoint{Tg}
    anchors::Vector{_LinearAnchoredQuery{Tg, Tg}}
    grid_size::Int
    extrap::EP
end

# ========================================
# Size / Introspection
# ========================================

Base.size(adj::LinearAdjoint) = (adj.grid_size, length(adj.anchors))
Base.size(adj::LinearAdjoint, d::Integer) = size(adj)[d]

# ========================================
# Callable Methods
# ========================================

"""
    (adj::LinearAdjoint)(y_bar; deriv=EvalValue()) -> f_bar

Apply the adjoint operator: `f̄ = W_dᵀȳ`. Allocating version.

The `deriv` keyword selects which forward operator's adjoint to compute:
- `EvalValue()` (default): adjoint of value interpolation
- `EvalDeriv1()` / `DerivOp(1)`: adjoint of first derivative
- `EvalDeriv2()`/`EvalDeriv3()`: always returns zero vector (linear has no 2nd+ derivative)
"""
function (adj::LinearAdjoint{Tg})(y_bar::AbstractVector; deriv::DerivOp = EvalValue(), _extra...) where {Tg}
    length(y_bar) == length(adj.anchors) || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), length(adj.anchors))
    Tv = promote_type(eltype(y_bar), Tg)
    f_bar = zeros(Tv, adj.grid_size)
    _linear_adjoint_apply!(f_bar, adj, y_bar, deriv)
    return f_bar
end

"""
    (adj::LinearAdjoint)(f_bar, y_bar; deriv=EvalValue()) -> f_bar

Apply the adjoint operator in-place: `f̄ = W_dᵀȳ`.
Zeros `f_bar` before accumulating. See allocating version for `deriv` options.
"""
function (adj::LinearAdjoint{Tg})(f_bar::AbstractVector, y_bar::AbstractVector; deriv::DerivOp = EvalValue(), _extra...) where {Tg}
    length(f_bar) == adj.grid_size || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), adj.grid_size)
    length(y_bar) == length(adj.anchors) || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), length(adj.anchors))
    fill!(f_bar, zero(eltype(f_bar)))
    _linear_adjoint_apply!(f_bar, adj, y_bar, deriv)
    return f_bar
end

# ── Scalar / Tuple callables ─────────────────────────────────────────────

function (adj::LinearAdjoint{Tg})(y_bar::Real; deriv::DerivOp = EvalValue(), _extra...) where {Tg}
    length(adj.anchors) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, length(adj.anchors))
    Tv = promote_type(typeof(y_bar), Tg)
    f_bar = zeros(Tv, adj.grid_size)
    _linear_adjoint_apply!(f_bar, adj, (y_bar,), deriv)
    return f_bar
end

function (adj::LinearAdjoint{Tg})(y_bar::Tuple{Vararg{Real}}; deriv::DerivOp = EvalValue(), _extra...) where {Tg}
    length(y_bar) == length(adj.anchors) || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), length(adj.anchors))
    Tv = promote_type(eltype(y_bar), Tg)
    f_bar = zeros(Tv, adj.grid_size)
    _linear_adjoint_apply!(f_bar, adj, y_bar, deriv)
    return f_bar
end

function (adj::LinearAdjoint{Tg})(f_bar::AbstractVector, y_bar::Real; deriv::DerivOp = EvalValue(), _extra...) where {Tg}
    length(f_bar) == adj.grid_size || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), adj.grid_size)
    length(adj.anchors) == 1 || _throw_adjoint_dim_mismatch("y_bar", 1, length(adj.anchors))
    fill!(f_bar, zero(eltype(f_bar)))
    _linear_adjoint_apply!(f_bar, adj, (y_bar,), deriv)
    return f_bar
end

function (adj::LinearAdjoint{Tg})(f_bar::AbstractVector, y_bar::Tuple{Vararg{Real}}; deriv::DerivOp = EvalValue(), _extra...) where {Tg}
    length(f_bar) == adj.grid_size || _throw_adjoint_dim_mismatch("f_bar", length(f_bar), adj.grid_size)
    length(y_bar) == length(adj.anchors) || _throw_adjoint_dim_mismatch("y_bar", length(y_bar), length(adj.anchors))
    fill!(f_bar, zero(eltype(f_bar)))
    _linear_adjoint_apply!(f_bar, adj, y_bar, deriv)
    return f_bar
end

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

function _scatter_linear_adjoint!(
        f_bar::AbstractVector,
        anchors::Vector{<:_LinearAnchoredQuery},
        y_bar,
        ::EvalValue,
        extrap::AbstractExtrap
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        _is_oob_skip(aq.side, extrap) && continue
        yb = y_bar[q]
        f_bar[aq.idx] += (one(aq.alpha) - aq.alpha) * yb
        f_bar[aq.idx + 1] += aq.alpha * yb
    end
    return nothing
end

# ── EvalDeriv1 scatter ────────────────────────────────────────────────────
# Forward: dy/dx = (f[i+1] - f[i]) / h = f[i+1]·inv_h - f[i]·inv_h
# Adjoint: f̄[i] += (-inv_h)·ȳ_q, f̄[i+1] += inv_h·ȳ_q

function _scatter_linear_adjoint!(
        f_bar::AbstractVector,
        anchors::Vector{<:_LinearAnchoredQuery},
        y_bar,
        ::EvalDeriv1,
        extrap::AbstractExtrap
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        _is_oob_skip_deriv(aq.side, extrap) && continue
        yb = y_bar[q]
        f_bar[aq.idx] -= aq.inv_h * yb
        f_bar[aq.idx + 1] += aq.inv_h * yb
    end
    return nothing
end

# ── EvalDeriv2 / EvalDeriv3 scatter ──────────────────────────────────────
# Linear interpolation has zero 2nd+ derivative → no scatter needed

function _scatter_linear_adjoint!(
        ::AbstractVector,
        ::Vector{<:_LinearAnchoredQuery},
        ::Any,
        ::Union{EvalDeriv2, EvalDeriv3},
        ::AbstractExtrap
    )
    return nothing
end

# ========================================
# OOB Skip Helpers (Compile-Time Dispatch)
# ========================================
#
# These provide zero-cost OOB handling by dispatching on extrap type:
# - FillExtrap: OOB → filled constant independent of f → zero gradient
# - ClampExtrap: OOB → f[boundary], but derivative is zero for constant extrap
# - NoExtrap/ExtendExtrap/WrapExtrap: no OOB skip (NoExtrap throws at construction)

# EvalValue OOB skip: only FillExtrap needs skip (fill value not function of f)
@inline _is_oob_skip(side::UInt8, ::FillExtrap) = side != 0x00
@inline _is_oob_skip(::UInt8, ::AbstractExtrap) = false

# EvalDeriv1 OOB skip: both Clamp and Fill have zero derivative outside domain
@inline _is_oob_skip_deriv(side::UInt8, ::_ClampOrFill) = side != 0x00
@inline _is_oob_skip_deriv(::UInt8, ::AbstractExtrap) = false

# ========================================
# ClampExtrap / FillExtrap OOB Side Fixup
# ========================================

"""
Restore `side` flags for anchors built with clamped query positions.

When ClampExtrap/FillExtrap queries are clamped before anchoring, the anchor
gets `side=0x00` (inside). This restores the correct OOB side flag based on the
original query position, so scatter can skip OOB contributions.
"""
function _fixup_linear_anchor_sides!(
        anchors::Vector{_LinearAnchoredQuery{Tg, Tg}},
        xq_original::AbstractVector{Tg},
        x_lo::Tg, x_hi::Tg
    ) where {Tg}
    @inbounds for i in eachindex(anchors)
        xq_i = xq_original[i]
        (x_lo <= xq_i <= x_hi) && continue
        side = xq_i < x_lo ? 0x01 : 0x02
        aq = anchors[i]
        anchors[i] = _LinearAnchoredQuery{Tg, Tg}(
            aq.idx, aq.xq, side, aq.h, aq.inv_h, aq.alpha
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
    xq_p = _to_float(x_query, Tg)

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
        _fixup_linear_anchor_sides!(anchors, xq_p, x_lo, x_hi)
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

# ========================================
# Matrix Materialization
# ========================================

"""
    Matrix(adj::LinearAdjoint; deriv=EvalValue()) -> Matrix{Tg}

Materialize the adjoint as a dense matrix `Wᵀ` of size `(n_grid, n_query)`.

# Example
```julia
adj = linear_adjoint(x, xq)
Wᵀ = Matrix(adj)                          # (n_grid × n_query)
W  = Matrix(adj)'                          # (n_query × n_grid)

@assert Wᵀ * y_bar ≈ adj(y_bar)           # matrix-vector == operator
@assert W * f ≈ itp.(xq)                  # forward matrix works too
```
"""
function Base.Matrix(adj::LinearAdjoint{Tg}; deriv::DerivOp = EvalValue()) where {Tg}
    n_out, n_query = size(adj)
    W_T = zeros(Tg, n_out, n_query)
    e_q = zeros(Tg, n_query)
    @inbounds for q in 1:n_query
        e_q[q] = one(Tg)
        adj(view(W_T, :, q), e_q; deriv = deriv)
        e_q[q] = zero(Tg)
    end
    return W_T
end
