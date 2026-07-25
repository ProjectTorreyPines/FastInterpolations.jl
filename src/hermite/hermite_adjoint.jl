# ========================================
# Cubic Hermite Adjoint: Shared Core + HermiteAdjoint1D
# ========================================
#
# Shared adjoint infrastructure for ALL cubic Hermite family:
# - HermiteAdjoint1D  (user-supplied slopes, y-only scatter — this file)
# - PchipAdjoint1D    (PCHIP, full scatter — later)
# - CardinalAdjoint1D (Cardinal/CatmullRom, full scatter — later)
# - AkimaAdjoint1D    (Akima, full scatter — later)
#
# Forward Hermite kernel:
#   P(t) = h₀₀(t)·yL + h₁₀(t)·h·dyL + h₀₁(t)·yR + h₁₁(t)·h·dyR
#
# The adjoint scatters ȳ_q back using the transpose of the kernel weights.
# HermiteAdjoint1D only scatters y-weights (h₀₀, h₀₁) because user-supplied
# slopes are treated as fixed data, not optimized quantities.

# ========================================
# Anchor Type
# ========================================

"""
    _HermiteAdjointAnchor1D{Tg}

Precomputed per-query weights for Hermite adjoint scatter.
Each weight 4-tuple represents: `(w_yL, w_yR, w_dyL, w_dyR)` — contributions
to `f_bar[idx]`, `f_bar[idx+1]`, `dy_bar[idx]`, `dy_bar[idx+1]` respectively.

Weight tuples for derivative orders 0–3 are pre-baked at construction time.
"""
struct _HermiteAdjointAnchor1D{Tg}
    idx::Int              # Interval index
    w0::NTuple{4, Tg}     # (w_yL, w_yR, w_dyL, w_dyR) for EvalValue
    w1::NTuple{4, Tg}     # for EvalDeriv1
    w2::NTuple{4, Tg}     # for EvalDeriv2
    w3::NTuple{4, Tg}     # for EvalDeriv3
end

# ========================================
# Weight Computation
# ========================================

"""
    _compute_hermite_adjoint_weights(t, h, inv_h) -> (w0, w1, w2, w3)

Compute all 4 derivative-order weight tuples for one query at normalized position `t`.
Each tuple is `(w_yL, w_yR, w_dyL, w_dyR)`.

# Derivation
EvalValue:  P  = h₀₀(t)·yL + h₀₁(t)·yR + h₁₀(t)·h·dyL + h₁₁(t)·h·dyR
EvalDeriv1: P' = h₀₀'(t)/h·yL + h₀₁'(t)/h·yR + h₁₀'(t)·dyL + h₁₁'(t)·dyR
EvalDeriv2: P''= h₀₀''(t)/h²·yL + h₀₁''(t)/h²·yR + h₁₀''(t)/h·dyL + h₁₁''(t)/h·dyR
EvalDeriv3: P'''= 12/h³·yL - 12/h³·yR + 6/h²·dyL + 6/h²·dyR
"""
@inline function _compute_hermite_adjoint_weights(t::Tg, h::Tg, inv_h::Tg) where {Tg}
    t2 = t * t
    t3 = t2 * t

    # k=0: EvalValue — basis functions × h for slope terms
    h00 = muladd(Tg(2), t3, muladd(Tg(-3), t2, one(Tg)))    # 2t³ - 3t² + 1
    h01 = muladd(Tg(-2), t3, Tg(3) * t2)                     # -2t³ + 3t²
    h10_h = h * muladd(t3, one(Tg), muladd(Tg(-2), t2, t))   # h·(t³ - 2t² + t)
    h11_h = h * (t3 - t2)                                     # h·(t³ - t²)
    w0 = (h00, h01, h10_h, h11_h)

    # k=1: EvalDeriv1 — d/dt basis × inv_h for value terms, d/dt basis for slope terms
    dh00 = muladd(Tg(6), t2, Tg(-6) * t)                     # 6t² - 6t
    dh01 = muladd(Tg(-6), t2, Tg(6) * t)                     # -6t² + 6t
    dh10 = muladd(Tg(3), t2, muladd(Tg(-4), t, one(Tg)))    # 3t² - 4t + 1
    dh11 = muladd(Tg(3), t2, Tg(-2) * t)                     # 3t² - 2t
    w1 = (dh00 * inv_h, dh01 * inv_h, dh10, dh11)

    # k=2: EvalDeriv2 — d²/dt² basis × inv_h² for value, × inv_h for slope
    inv_h2 = inv_h * inv_h
    d2h00 = muladd(Tg(12), t, Tg(-6))                        # 12t - 6
    d2h01 = muladd(Tg(-12), t, Tg(6))                        # -12t + 6
    d2h10 = muladd(Tg(6), t, Tg(-4))                         # 6t - 4
    d2h11 = muladd(Tg(6), t, Tg(-2))                         # 6t - 2
    w2 = (d2h00 * inv_h2, d2h01 * inv_h2, d2h10 * inv_h, d2h11 * inv_h)

    # k=3: EvalDeriv3 — constant (d³h/dt³ are 12, -12, 6, 6)
    inv_h3 = inv_h2 * inv_h
    w3 = (Tg(12) * inv_h3, Tg(-12) * inv_h3, Tg(6) * inv_h2, Tg(6) * inv_h2)

    return w0, w1, w2, w3
end

# ========================================
# Anchor Baking
# ========================================

"""
    _bake_hermite_adjoint_anchors(x, xq, extrap) -> Vector{_HermiteAdjointAnchor1D}

Precompute cell indices and all derivative-order weights for each query point.
OOB handling baked into weights at construction time (no runtime OOB checks).
Axis-as-truth: `x` is the wrapped axis carrying cached h/inv_h.
"""
function _bake_hermite_adjoint_anchors(
        x::AbstractVector{Tg},
        xq::AbstractVector,
        extrap::AbstractExtrap
    ) where {Tg}
    # Classification (`_is_inbounds`, widened bounds) and clamp geometry
    # (`_clamp_to_grid`, actual endpoints) stay in separate helpers; the wrap
    # routes through the 2-arg axis-aware form, so the widened bracket never leaks.
    need_clamp = extrap isa Union{ClampExtrap, FillExtrap}
    wrap = extrap isa WrapExtrap

    anchors = Vector{_HermiteAdjointAnchor1D{Tg}}(undef, length(xq))
    @inbounds for q in eachindex(xq)
        xq_raw = xq[q]

        # Preprocess query point based on extrap mode
        xq_eval = if need_clamp
            _clamp_to_grid(xq_raw, x)
        elseif wrap
            _wrap_to_domain(xq_raw, x)
        else
            xq_raw
        end

        # Search interval
        idx, _, xL, _ = search_interval(DEFAULT_SEARCHER, x, xq_eval)
        h = _get_h(x, idx)
        inv_h = _get_inv_h(x, idx)
        t = (xq_eval - xL) * inv_h

        # Compute all weight sets
        w0, w1, w2, w3 = _compute_hermite_adjoint_weights(t, h, inv_h)

        # OOB weight fixup (bake into weights at construction)
        is_oob = !_is_inbounds(x, xq_raw)
        if is_oob
            z = zero(Tg)
            z4 = (z, z, z, z)
            if extrap isa FillExtrap
                # Fill: all weights zero (fill constant independent of f)
                w0 = z4
                w1 = z4
                w2 = z4
                w3 = z4
            elseif extrap isa ClampExtrap
                # Clamp: value weights OK, deriv weights zero
                w1 = z4
                w2 = z4
                w3 = z4
            end
            # ExtendExtrap/WrapExtrap: weights unchanged (correct behavior)
        end

        anchors[q] = _HermiteAdjointAnchor1D{Tg}(idx, w0, w1, w2, w3)
    end
    return anchors
end

# ========================================
# Scatter Functions — Full (f_bar + dy_bar)
# ========================================
# Used by PCHIP/Cardinal/Akima adjoints where slopes are computed from data.

# ── EvalValue scatter ────────────────────────────────────────────────────
@inline function _scatter_hermite_adjoint!(
        f_bar::AbstractVector, dy_bar::AbstractVector,
        anchors::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::EvalValue
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        w_yL, w_yR, w_dyL, w_dyR = aq.w0
        f_bar[aq.idx] += w_yL * yb
        f_bar[aq.idx + 1] += w_yR * yb
        dy_bar[aq.idx] += w_dyL * yb
        dy_bar[aq.idx + 1] += w_dyR * yb
    end
    return nothing
end

# ── EvalDeriv1 scatter ───────────────────────────────────────────────────
@inline function _scatter_hermite_adjoint!(
        f_bar::AbstractVector, dy_bar::AbstractVector,
        anchors::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::EvalDeriv1
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        w_yL, w_yR, w_dyL, w_dyR = aq.w1
        f_bar[aq.idx] += w_yL * yb
        f_bar[aq.idx + 1] += w_yR * yb
        dy_bar[aq.idx] += w_dyL * yb
        dy_bar[aq.idx + 1] += w_dyR * yb
    end
    return nothing
end

# ── EvalDeriv2 scatter ───────────────────────────────────────────────────
@inline function _scatter_hermite_adjoint!(
        f_bar::AbstractVector, dy_bar::AbstractVector,
        anchors::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::EvalDeriv2
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        w_yL, w_yR, w_dyL, w_dyR = aq.w2
        f_bar[aq.idx] += w_yL * yb
        f_bar[aq.idx + 1] += w_yR * yb
        dy_bar[aq.idx] += w_dyL * yb
        dy_bar[aq.idx + 1] += w_dyR * yb
    end
    return nothing
end

# ── EvalDeriv3 scatter ───────────────────────────────────────────────────
@inline function _scatter_hermite_adjoint!(
        f_bar::AbstractVector, dy_bar::AbstractVector,
        anchors::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::EvalDeriv3
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        w_yL, w_yR, w_dyL, w_dyR = aq.w3
        f_bar[aq.idx] += w_yL * yb
        f_bar[aq.idx + 1] += w_yR * yb
        dy_bar[aq.idx] += w_dyL * yb
        dy_bar[aq.idx + 1] += w_dyR * yb
    end
    return nothing
end

# ── Generic fallback: 4th+ derivative of cubic is zero ───────────────────
@inline function _scatter_hermite_adjoint!(
        ::AbstractVector, ::AbstractVector,
        ::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::DerivOp{N}
    ) where {N}
    return nothing
end

# ========================================
# Scatter Functions — Y-Only (f_bar only)
# ========================================
# Used by HermiteAdjoint1D where user-supplied slopes are fixed.

# ── EvalValue scatter ────────────────────────────────────────────────────
@inline function _scatter_hermite_adjoint_y_only!(
        f_bar::AbstractVector,
        anchors::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::EvalValue
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        w_yL, w_yR, _, _ = aq.w0
        f_bar[aq.idx] += w_yL * yb
        f_bar[aq.idx + 1] += w_yR * yb
    end
    return nothing
end

# ── EvalDeriv1 scatter ───────────────────────────────────────────────────
@inline function _scatter_hermite_adjoint_y_only!(
        f_bar::AbstractVector,
        anchors::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::EvalDeriv1
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        w_yL, w_yR, _, _ = aq.w1
        f_bar[aq.idx] += w_yL * yb
        f_bar[aq.idx + 1] += w_yR * yb
    end
    return nothing
end

# ── EvalDeriv2 scatter ───────────────────────────────────────────────────
@inline function _scatter_hermite_adjoint_y_only!(
        f_bar::AbstractVector,
        anchors::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::EvalDeriv2
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        w_yL, w_yR, _, _ = aq.w2
        f_bar[aq.idx] += w_yL * yb
        f_bar[aq.idx + 1] += w_yR * yb
    end
    return nothing
end

# ── EvalDeriv3 scatter ───────────────────────────────────────────────────
@inline function _scatter_hermite_adjoint_y_only!(
        f_bar::AbstractVector,
        anchors::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::EvalDeriv3
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        w_yL, w_yR, _, _ = aq.w3
        f_bar[aq.idx] += w_yL * yb
        f_bar[aq.idx + 1] += w_yR * yb
    end
    return nothing
end

# ── Generic fallback: 4th+ derivative of cubic is zero ───────────────────
@inline function _scatter_hermite_adjoint_y_only!(
        ::AbstractVector,
        ::Vector{<:_HermiteAdjointAnchor1D}, y_bar,
        ::DerivOp{N}
    ) where {N}
    return nothing
end

# ========================================
# HermiteAdjoint1D Struct
# ========================================

"""
    HermiteAdjoint1D{Tg, EP}

Adjoint (transpose) operator for cubic Hermite interpolation with user-supplied slopes.
Computes `f̄ = Wᵀȳ` where `W` is the forward Hermite interpolation weight matrix,
considering only the `y`-dependence (slopes `dy` are treated as fixed).

Constructed from a grid and query points (query-baked, data-free).

# Type Parameters
- `Tg`: Grid type — normally Float32/Float64, unconstrained for duck-typed grids (e.g. ForwardDiff.Dual)
- `EP`: Extrapolation policy type (`NoExtrap`, `ExtendExtrap`, `ClampExtrap`, `FillExtrap`, `WrapExtrap`)

# Usage
```julia
adj = hermite_adjoint(x, xq)

# Value adjoint (default)
f_bar = adj(y_bar)

# Derivative adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))

# In-place
adj(f_bar, y_bar; deriv=DerivOp(1))

# Dot-product identity: ⟨W·y, ȳ⟩ = ⟨y, Wᵀȳ⟩
@assert dot(itp.(xq), y_bar) ≈ dot(y, adj(y_bar))
```
"""
struct HermiteAdjoint1D{Tg, EP <: AbstractExtrap} <: AbstractAdjoint1D{Tg}
    anchors::Vector{_HermiteAdjointAnchor1D{Tg}}
    grid_size::Int
    extrap::EP
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================

@inline _n_queries(adj::HermiteAdjoint1D) = length(adj.anchors)
@inline _adjoint_output_length(adj::HermiteAdjoint1D) = adj.grid_size

# HermiteAdjoint1D has no `bc` field (user-supplied slopes; periodic BC is
# meaningless without slope handling). Override the protocol defaults that
# read `adj.bc` to avoid a FieldError on the allocating callable's finalize.
@inline _adjoint_1d_has_seam_fold(::HermiteAdjoint1D) = false
@inline _adjoint_1d_finalize(f_bar::AbstractVector, ::HermiteAdjoint1D) = f_bar

@inline function _adjoint_1d_apply!(f_bar, adj::HermiteAdjoint1D, y_bar, deriv)
    _scatter_hermite_adjoint_y_only!(f_bar, adj.anchors, y_bar, deriv)
    return nothing
end

# ========================================
# Constructor
# ========================================

"""
    hermite_adjoint(x, x_query; extrap=NoExtrap()) -> HermiteAdjoint1D

Create a cubic Hermite adjoint operator for user-supplied slopes (query-baked, data-free).

Computes `f̄ = Wᵀȳ` where `W` is the forward Hermite interpolation weight matrix
restricted to the `y`-dependence. User-supplied slopes `dy` are treated as fixed
and do not receive gradient contributions.

This adjoint satisfies the dot-product identity:

    dot(hermite_interp(x, y, dy, xq), ȳ) ≈ dot(y, hermite_adjoint(x, xq)(ȳ))

For PCHIP/Cardinal/Akima where slopes are computed from `y`, use their dedicated
adjoint constructors instead (which scatter to both `y` and `dy`).

# Arguments
- `x::AbstractVector`: Grid points (must be sorted)
- `x_query::AbstractVector`: Query points (baked into the operator)
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`)

# Example
```julia
using LinearAlgebra
x = collect(range(0, 1, 50))
xq = sort(rand(30))
y  = sin.(x)
dy = cos.(x)
y_bar = randn(30)

itp = hermite_interp(x, y, dy)
adj = hermite_adjoint(x, xq)
f_bar = adj(y_bar)

# Dot-product identity: ⟨W·y, ȳ⟩ = ⟨y, Wᵀȳ⟩
@assert dot(itp.(xq), y_bar) ≈ dot(y, adj(y_bar))
```

# Notes
- No boundary conditions needed (Hermite uses supplied slopes directly).
- Pure scatter operation (no tridiagonal solve).
- 4th+ derivative adjoints always return zero (cubic polynomial).
"""
function hermite_adjoint(
        x::AbstractVector,
        x_query::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
    )
    x_p, xq_p, Tg = _promote_adjoint_inputs(x, x_query)

    length(x_p) >= 2 || _throw_adjoint_grid_too_small(length(x_p))

    # NoExtrap: validate all queries in-domain
    if extrap isa NoExtrap
        _validate_domain(x_p, xq_p)
    end

    # Caching wrap (transient — used only for anchor baking). Range from
    # `_promote_adjoint_inputs` arrives as `_CachedRange` (idempotent
    # passthrough); plain `Vector` becomes `_CachedVector` for cached h/inv_h.
    x_axis = _cache_axis(x_p, NoBC())
    anchors = _bake_hermite_adjoint_anchors(x_axis, xq_p, extrap)

    return HermiteAdjoint1D{Tg, typeof(extrap)}(anchors, length(x_axis), extrap)
end

# Scalar query convenience
function hermite_adjoint(
        x::AbstractVector,
        x_query::Number;
        extrap::AbstractExtrap = NoExtrap(),
    )
    return hermite_adjoint(x, [x_query]; extrap = extrap)
end

# ========================================
# Full Hermite Pullback (for AD — returns both ∂y and ∂dy)
# ========================================

"""
    _hermite_full_pullback(adj, y_bar, deriv) -> (f̄_y, f̄_dy)

Compute gradients w.r.t. both `y` and `dy` for `hermite_interp(x, y, dy, xq)`.

Uses the full `_scatter_hermite_adjoint!` (not y-only) to populate both vectors.
This is called by the ChainRulesCore rrule for the Hermite wrapper path.
"""
function _hermite_full_pullback(
        adj::HermiteAdjoint1D{Tg},
        y_bar,
        deriv::DerivOp = EvalValue()
    ) where {Tg}
    Tv = y_bar isa AbstractVector ? promote_type(eltype(y_bar), Tg) : promote_type(typeof(y_bar), Tg)
    n = adj.grid_size
    f_bar_y = zeros(Tv, n)
    f_bar_dy = zeros(Tv, n)
    _scatter_hermite_adjoint!(f_bar_y, f_bar_dy, adj.anchors, y_bar, deriv)
    return f_bar_y, f_bar_dy
end

# Scalar y_bar: wrap to vector, unpack
function _hermite_full_pullback(
        adj::HermiteAdjoint1D,
        y_bar::Number,
        deriv::DerivOp = EvalValue()
    )
    return _hermite_full_pullback(adj, [y_bar], deriv)
end
