# ========================================
# QuadraticAdjoint: Adjoint (Transpose) Operator
# ========================================
#
# Computes f̄ = Wᵀ · ȳ where W is the implicit forward interpolation matrix.
# The adjoint pipeline reverses the forward: scatter → recurrence⁻ᵀ → secantᵀ.
#
# Key difference from CubicAdjoint:
# - No Thomas LU solve — uses simple O(n) recurrence adjoint instead
# - 3-point stencil per query: (fL, fR, dfL) vs cubic's 4-point (fL, fR, dzL, dzR)
# - No dual cache — effective BC computed on the fly
#
# Dependencies (already included before this file):
# - QuadraticBC, _fill_slopes!, _compute_quadratic_secants! (quadratic_solver.jl)
# - _quadratic_kernel (quadratic_kernels.jl)
# - _anchor_query, _QuadraticAnchoredQuery (quadratic_anchor.jl)
# - QuadraticBC (quadratic_solver.jl)
# - _get_h, _get_inv_h, _create_spacing (grid_spacing.jl)
# - _precompute_polyfit_coeffs, _AdjointPolyfitData (cubic_adjoint.jl)
# - AbstractAdjoint1D, _is_oob_skip, _is_oob_skip_deriv (adjoint_protocol.jl)

# ========================================
# Anchor Type
# ========================================

"""
    _QuadraticAdjointAnchor1D{Tg}

Precomputed per-query weights for quadratic adjoint scatter.
Each weight 3-tuple represents: (w_fL, w_fR, w_d) — contributions to
`f_bar[idx]`, `f_bar[idx+1]`, and `d_bar[idx]` respectively.

Forward eval: S(x) = a·dL² + d·dL + y  where a = (s-d)/h, s = (y_{R}-y_{L})/h.
Rewritten:  S = y_L·(1-t²) + y_R·t² + d·h·t·(1-t)  where t = dL/h.
"""
struct _QuadraticAdjointAnchor1D{Tg <: AbstractFloat}
    idx::Int              # Interval index
    w0::NTuple{3, Tg}     # (w_fL, w_fR, w_d) for EvalValue
    w1::NTuple{3, Tg}     # for EvalDeriv1
    w2::NTuple{3, Tg}     # for EvalDeriv2
end

# ========================================
# Weight Computation
# ========================================

"""
    _compute_quadratic_adjoint_weights(t, h, inv_h) -> (w0, w1, w2)

Compute all 3 derivative-order weight tuples for one query at normalized position `t`.
Each tuple is `(w_fL, w_fR, w_d)`.

# Derivation
EvalValue:  S  = y_L·(1-t²) + y_R·t²   + d·h·t·(1-t)
EvalDeriv1: S' = y_L·(-2t/h) + y_R·(2t/h) + d·(1-2t)
EvalDeriv2: S''= y_L·(-2/h²) + y_R·(2/h²) + d·(-2/h)
"""
@inline function _compute_quadratic_adjoint_weights(t::Tg, h::Tg, inv_h::Tg) where {Tg}
    t_sq = t * t

    # k=0: EvalValue
    w0 = (one(Tg) - t_sq, t_sq, h * t * (one(Tg) - t))

    # k=1: EvalDeriv1
    two_t_inv_h = 2 * t * inv_h
    w1 = (-two_t_inv_h, two_t_inv_h, one(Tg) - 2 * t)

    # k=2: EvalDeriv2
    inv_h2 = inv_h * inv_h
    two_inv_h2 = 2 * inv_h2
    w2 = (-two_inv_h2, two_inv_h2, -2 * inv_h)

    return w0, w1, w2
end

# ========================================
# Anchor Baking
# ========================================

"""
    _bake_quadratic_adjoint_anchors(x, spacing, xq, extrap) -> Vector{_QuadraticAdjointAnchor1D}

Precompute cell indices and all derivative-order weights for each query point.
OOB handling baked into weights at construction time (no runtime OOB checks).
"""
function _bake_quadratic_adjoint_anchors(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        xq::AbstractVector{Tg},
        extrap::AbstractExtrap
    ) where {Tg <: AbstractFloat}
    x_lo, x_hi = first(x), last(x)

    # For ClampExtrap/FillExtrap, clamp queries to domain before anchoring
    need_clamp = extrap isa Union{ClampExtrap, FillExtrap}
    wrap = extrap isa WrapExtrap

    anchors = Vector{_QuadraticAdjointAnchor1D{Tg}}(undef, length(xq))
    @inbounds for q in eachindex(xq)
        xq_raw = xq[q]

        # Preprocess query point based on extrap mode
        xq_eval = if need_clamp
            clamp(xq_raw, x_lo, x_hi)
        elseif wrap
            _wrap_to_domain(xq_raw, x_lo, x_hi)
        else
            xq_raw
        end

        # Search interval
        idx, xL, _ = search_interval(DEFAULT_SEARCHER, x, spacing, xq_eval)
        h = _get_h(spacing, idx)
        inv_h = _get_inv_h(spacing, idx)
        t = (xq_eval - xL) * inv_h

        # Compute all weight sets
        w0, w1, w2 = _compute_quadratic_adjoint_weights(t, h, inv_h)

        # OOB weight fixup (bake into weights at construction)
        is_oob = xq_raw < x_lo || xq_raw > x_hi
        if is_oob
            z = zero(Tg)
            z3 = (z, z, z)
            if extrap isa FillExtrap
                # Fill: all weights zero (fill constant independent of f)
                w0 = z3
                w1 = z3
                w2 = z3
            elseif extrap isa ClampExtrap
                # Clamp: value weights OK (clamped gives correct boundary value), deriv weights zero
                w1 = z3
                w2 = z3
            end
            # ExtendExtrap/WrapExtrap: weights unchanged (correct behavior)
        end

        anchors[q] = _QuadraticAdjointAnchor1D{Tg}(idx, w0, w1, w2)
    end
    return anchors
end

# ========================================
# QuadraticAdjoint Struct
# ========================================

"""
    QuadraticAdjoint{Tg, S, BC, X}

Adjoint (transpose) operator for quadratic spline interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward interpolation weight matrix.

Constructed from a grid and query points (query-baked, data-free).

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `S`: Grid spacing type (for fast inv_h access)
- `X`: Grid vector type (after copy for mutation safety)
- `BC`: Boundary condition type (Left, Right, or MinCurvFit)

# Usage
```julia
adj = quadratic_adjoint(x_grid, x_query; bc=Left(QuadraticFit()))

# Value adjoint (default)
f_bar = adj(y_bar)

# Derivative adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))

# In-place
adj(f_bar, y_bar; deriv=DerivOp(1))

# Dot-product identity: ⟨W·f, ȳ⟩ = ⟨f, Wᵀȳ⟩
@assert dot(itp.(xq), y_bar) ≈ dot(f, adj(y_bar))
```

# Mathematical Background
Forward: `y = W·f` where `W` encodes: secants → slope recurrence → eval.
Adjoint pipeline reverses: scatter → recurrence adjoint → secant adjoint.

No tridiagonal solve needed (unlike cubic) — the slope recurrence
`d[i+1] = 2s[i] - d[i]` has a simple O(n) adjoint.
"""
struct QuadraticAdjoint{
        Tg <: AbstractFloat,
        S <: AbstractGridSpacing{Tg},
        BC <: QuadraticBC,
        X <: AbstractVector{Tg},
    } <: AbstractAdjoint1D{Tg}
    spacing::S
    anchors::Vector{_QuadraticAdjointAnchor1D{Tg}}
    bc::BC
    grid_size::Int
    grid::X  # Needed for PolyFit BC adjoint
    mincurv_C::Tg  # Precomputed inv(Σ inv_h); only used for MinCurvFit BC

    # Inner constructor: copy() for mutation safety.
    # copy() on immutable Range types is a no-op (zero allocation).
    # typeof() rebinds X after copy (e.g. SubArray → Vector).
    function QuadraticAdjoint(
            spacing::S, anchors::Vector{_QuadraticAdjointAnchor1D{Tg}},
            bc::BC, grid_size::Int, grid::AbstractVector{Tg}
        ) where {Tg <: AbstractFloat, S <: AbstractGridSpacing{Tg}, BC <: QuadraticBC}
        gc = copy(grid)
        C = bc isa MinCurvFit ? _compute_mincurv_C(spacing, grid_size) : zero(Tg)
        return new{Tg, S, BC, typeof(gc)}(spacing, anchors, bc, grid_size, gc, C)
    end
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================

@inline _n_queries(adj::QuadraticAdjoint) = length(adj.anchors)
@inline _adjoint_output_length(adj::QuadraticAdjoint) = adj.grid_size

@inline _adjoint_1d_apply!(f_bar, adj::QuadraticAdjoint, y_bar, deriv) =
    _quadratic_adjoint_apply!(f_bar, adj, y_bar, deriv)

# ========================================
# Step 1: Evaluation Adjoint Scatter
# ========================================

# ── EvalValue scatter ─────────────────────────────────────────────────────
# Forward: S = y_L·(1-t²) + y_R·t² + d·h·t·(1-t)
# Adjoint: f̄[idx] += w_fL·ȳ, f̄[idx+1] += w_fR·ȳ, d̄[idx] += w_d·ȳ

@inline function _scatter_eval_adjoint_quadratic!(
        f_bar::AbstractVector, d_bar::AbstractVector,
        anchors::Vector{<:_QuadraticAdjointAnchor1D}, y_bar,
        ::EvalValue
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wfL, wfR, wd = aq.w0
        f_bar[aq.idx] += wfL * yb
        f_bar[aq.idx + 1] += wfR * yb
        d_bar[aq.idx] += wd * yb
    end
    return nothing
end

# ── EvalDeriv1 scatter ────────────────────────────────────────────────────
@inline function _scatter_eval_adjoint_quadratic!(
        f_bar::AbstractVector, d_bar::AbstractVector,
        anchors::Vector{<:_QuadraticAdjointAnchor1D}, y_bar,
        ::EvalDeriv1
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wfL, wfR, wd = aq.w1
        f_bar[aq.idx] += wfL * yb
        f_bar[aq.idx + 1] += wfR * yb
        d_bar[aq.idx] += wd * yb
    end
    return nothing
end

# ── EvalDeriv2 scatter ────────────────────────────────────────────────────
@inline function _scatter_eval_adjoint_quadratic!(
        f_bar::AbstractVector, d_bar::AbstractVector,
        anchors::Vector{<:_QuadraticAdjointAnchor1D}, y_bar,
        ::EvalDeriv2
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wfL, wfR, wd = aq.w2
        f_bar[aq.idx] += wfL * yb
        f_bar[aq.idx + 1] += wfR * yb
        d_bar[aq.idx] += wd * yb
    end
    return nothing
end

# ── EvalDeriv3 scatter ────────────────────────────────────────────────────
# 3rd derivative of quadratic is always zero → no scatter needed
@inline function _scatter_eval_adjoint_quadratic!(
        ::AbstractVector, ::AbstractVector,
        ::Vector{<:_QuadraticAdjointAnchor1D}, ::Any,
        ::EvalDeriv3
    )
    return nothing
end

# ========================================
# Step 2: Recurrence Adjoint
# ========================================
#
# Reverses the slope recurrence to propagate d̄ → s̄.
# Also handles BC endpoint adjoint (PolyFit, Deriv2, MinCurvFit).
#
# After this step, s_bar contains all secant sensitivities and
# f_bar may be updated for PolyFit BC endpoints.

# ── Precomputed MinCurvFit constant ───────────────────────────────────────
# C = inv(Σ inv_h[i]) is a grid-only constant used in MinCurvFit endpoint adjoint.
# Precomputed once at construction time, avoiding O(n) recomputation per call.

@inline function _compute_mincurv_C(spacing::AbstractGridSpacing{Tg}, n::Int) where {Tg}
    inv_h_sum = zero(Tg)
    @inbounds for i in 1:(n - 1)
        inv_h_sum += _get_inv_h(spacing, i)
    end
    return inv(inv_h_sum)
end

# ── Shared sweep helpers ──────────────────────────────────────────────────
# Factor the hot loops that are common across all Left/Right BC variants.

"""
Reverse sweep for Left BC (forward recurrence d[i+1] = 2s[i] - d[i]).
After this, d̄[1] holds the residual for BC endpoint adjoint.
"""
@inline function _recurrence_sweep_left!(s_bar, d_bar)
    n = length(d_bar)
    @inbounds for i in (n - 1):-1:1
        s_bar[i] += 2 * d_bar[i + 1]
        d_bar[i] -= d_bar[i + 1]
    end
    return nothing
end

"""
Forward sweep for Right BC (backward recurrence d[i] = 2s[i] - d[i+1]).
After this, d̄[n] holds the residual for BC endpoint adjoint.
"""
@inline function _recurrence_sweep_right!(s_bar, d_bar)
    n = length(d_bar)
    @inbounds for i in 1:(n - 1)
        s_bar[i] += 2 * d_bar[i]
        d_bar[i + 1] -= d_bar[i]
    end
    return nothing
end

# ── Left BC: forward recurrence d[i+1] = 2s[i] - d[i] ───────────────────
# Adjoint reverse sweep, then dispatch on BC type for residual d̄[1].

@inline function _recurrence_adjoint!(
        s_bar::AbstractVector{Tv},
        d_bar::AbstractVector{Tv},
        f_bar::AbstractVector{Tv},
        bc::Left{<:Deriv1},
        spacing::AbstractGridSpacing,
        grid::AbstractVector
    ) where {Tv}
    _recurrence_sweep_left!(s_bar, d_bar)
    # d̄[1] remains: Deriv1 → d[1] = constant → discard d̄[1]
    return nothing
end

@inline function _recurrence_adjoint!(
        s_bar::AbstractVector{Tv},
        d_bar::AbstractVector{Tv},
        f_bar::AbstractVector{Tv},
        bc::Left{<:Deriv2},
        spacing::AbstractGridSpacing,
        grid::AbstractVector
    ) where {Tv}
    _recurrence_sweep_left!(s_bar, d_bar)
    # d̄[1] remains: Deriv2 → d[1] = s[1] - κ·h[1]/2 → s̄[1] += d̄[1]
    @inbounds s_bar[1] += d_bar[1]
    return nothing
end

@inline function _recurrence_adjoint!(
        s_bar::AbstractVector{Tv},
        d_bar::AbstractVector{Tv},
        f_bar::AbstractVector{Tv},
        bc::Left{<:PolyFit{D}},
        spacing::AbstractGridSpacing{Tg},
        grid::AbstractVector{Tg}
    ) where {Tv, Tg, D}
    _recurrence_sweep_left!(s_bar, d_bar)
    # d̄[1] remains: PolyFit → d[1] = Σ coeffs[k]·y[k]
    coeffs = _precompute_polyfit_coeffs(bc.bc, grid, LeftSide())
    db1 = @inbounds d_bar[1]
    @inbounds for k in eachindex(coeffs)
        f_bar[k] += coeffs[k] * db1
    end
    return nothing
end

# ── Right BC: backward recurrence d[i] = 2s[i] - d[i+1] ─────────────────
# Adjoint forward sweep, then dispatch on BC type for residual d̄[n].

@inline function _recurrence_adjoint!(
        s_bar::AbstractVector{Tv},
        d_bar::AbstractVector{Tv},
        f_bar::AbstractVector{Tv},
        bc::Right{<:Deriv1},
        spacing::AbstractGridSpacing,
        grid::AbstractVector
    ) where {Tv}
    _recurrence_sweep_right!(s_bar, d_bar)
    # d̄[n] remains: Deriv1 → d[n] = constant → discard d̄[n]
    return nothing
end

@inline function _recurrence_adjoint!(
        s_bar::AbstractVector{Tv},
        d_bar::AbstractVector{Tv},
        f_bar::AbstractVector{Tv},
        bc::Right{<:Deriv2},
        spacing::AbstractGridSpacing,
        grid::AbstractVector
    ) where {Tv}
    _recurrence_sweep_right!(s_bar, d_bar)
    n = length(d_bar)
    # d̄[n] remains: Deriv2 → d[n] = s[n-1] + κ·h[n-1]/2 → s̄[n-1] += d̄[n]
    @inbounds s_bar[n - 1] += d_bar[n]
    return nothing
end

@inline function _recurrence_adjoint!(
        s_bar::AbstractVector{Tv},
        d_bar::AbstractVector{Tv},
        f_bar::AbstractVector{Tv},
        bc::Right{<:PolyFit{D}},
        spacing::AbstractGridSpacing{Tg},
        grid::AbstractVector{Tg}
    ) where {Tv, Tg, D}
    _recurrence_sweep_right!(s_bar, d_bar)
    n = length(d_bar)
    # d̄[n] remains: PolyFit → d[n] = Σ coeffs[k]·y[stencil_k]
    coeffs = _precompute_polyfit_coeffs(bc.bc, grid, RightSide())
    dbn = @inbounds d_bar[n]
    n_pts = n
    n_stencil = length(coeffs)
    @inbounds for k in eachindex(coeffs)
        f_bar[n_pts - n_stencil + k] += coeffs[k] * dbn
    end
    return nothing
end

# ── Dispatch wrapper ──────────────────────────────────────────────────────
# Routes the precomputed C to MinCurvFit; other BCs ignore it.

@inline function _call_recurrence_adjoint!(s_bar, d_bar, f_bar, bc, spacing, grid, _C)
    _recurrence_adjoint!(s_bar, d_bar, f_bar, bc, spacing, grid)
    return nothing
end

@inline function _call_recurrence_adjoint!(s_bar, d_bar, f_bar, bc::MinCurvFit, spacing, grid, C)
    _recurrence_adjoint!(s_bar, d_bar, f_bar, bc, spacing, grid, C)
    return nothing
end

# ── MinCurvFit BC ─────────────────────────────────────────────────────────
# Forward: d[1] = [Σ αᵢ(sᵢ - βᵢ)·inv_hᵢ] · inv(Σ inv_hᵢ)
#          then forward recurrence from d[1].
#
# Adjoint: 1) Forward recurrence adjoint (same as Left) → residual d̄[1]
#          2) MinCurvFit d[1] adjoint: d̄[1] → s̄ contributions

@inline function _recurrence_adjoint!(
        s_bar::AbstractVector{Tv},
        d_bar::AbstractVector{Tv},
        f_bar::AbstractVector{Tv},
        ::MinCurvFit,
        spacing::AbstractGridSpacing{Tg},
        grid::AbstractVector{Tg},
        C::Tg = _compute_mincurv_C(spacing, length(d_bar))
    ) where {Tv, Tg}
    n = length(d_bar)
    nm1 = n - 1

    # Part 1: Forward recurrence adjoint (same sweep as Left BCs)
    _recurrence_sweep_left!(s_bar, d_bar)

    # Part 2: MinCurvFit d[1] adjoint
    # Forward: d[1] = numerator * C where C = inv(inv_h_sum)
    # numerator = Σ α[i] * (s[i] - β[i]) * inv_h[i]
    # β[1]=0, β[i+1] = 2*s[i] - β[i], α[i] = (-1)^(i+1)

    # numerator_bar = d̄[1] * C
    numerator_bar = @inbounds d_bar[1] * C

    # Reverse sweep through the MinCurvFit forward loop
    # α[i] = (-1)^(i+1): at i=nm1, α = (-1)^(nm1+1)
    sign = isodd(nm1) ? one(Tg) : -one(Tg)
    beta_bar = zero(Tv)

    @inbounds for i in nm1:-1:1
        # Adjoint of β[i+1] = 2*s[i] - β[i]
        s_bar[i] += 2 * beta_bar
        beta_bar = -beta_bar

        # Adjoint of numerator += α[i] * (s[i] - β[i]) * inv_h[i]
        inv_h_i = _get_inv_h(spacing, i)
        contrib = sign * inv_h_i * numerator_bar
        s_bar[i] += contrib
        beta_bar -= contrib

        sign = -sign
    end

    return nothing
end

# ========================================
# Step 3: Secant Adjoint
# ========================================
#
# Reverse: s[i] = (y[i+1] - y[i]) * inv_h[i]
# Adjoint: f̄[i] -= s̄[i] * inv_h[i], f̄[i+1] += s̄[i] * inv_h[i]

@inline function _secant_adjoint!(
        f_bar::AbstractVector{Tv},
        s_bar::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg}
    ) where {Tv, Tg}
    @inbounds for i in eachindex(s_bar)
        inv_h = _get_inv_h(spacing, i)
        c = s_bar[i] * inv_h
        f_bar[i] -= c
        f_bar[i + 1] += c
    end
    return nothing
end

# ========================================
# Core Apply Function
# ========================================

@with_pool pool function _quadratic_adjoint_apply!(
        f_bar::AbstractVector{Tv},
        adj::QuadraticAdjoint{Tg},
        y_bar,
        deriv::DerivOp = EvalValue()
    ) where {Tv, Tg}
    n = adj.grid_size
    nm1 = n - 1

    # Pool-allocate work buffers
    d_bar = zeros!(pool, Tv, n)     # Slope sensitivities
    s_bar = zeros!(pool, Tv, nm1)   # Secant sensitivities

    # Step 1: Eval adjoint scatter → f̄, d̄
    _scatter_eval_adjoint_quadratic!(f_bar, d_bar, adj.anchors, y_bar, deriv)

    # Step 2: Recurrence adjoint → s̄ (+ BC endpoint contribution to f̄)
    _call_recurrence_adjoint!(s_bar, d_bar, f_bar, adj.bc, adj.spacing, adj.grid, adj.mincurv_C)

    # Step 3: Secant adjoint → f̄ update
    _secant_adjoint!(f_bar, s_bar, adj.spacing)

    return f_bar
end

# No BC normalization needed — constructor accepts QuadraticBC directly.
# The adjoint never uses BC values (Deriv1.v, Deriv2.κ), only dispatches on type.

# ========================================
# Constructor
# ========================================

"""
    quadratic_adjoint(x, x_query; bc=Left(QuadraticFit()), extrap=NoExtrap()) -> QuadraticAdjoint

Create a quadratic spline adjoint operator (query-baked, data-free).

Computes `f̄ = Wᵀȳ` where `W` is the forward interpolation weight matrix.
Maps query-space sensitivities back to grid-space sensitivities.

# Arguments
- `x::AbstractVector`: Grid points (must be sorted)
- `x_query::AbstractVector`: Query points (baked into the operator)
- `bc::QuadraticBC`: Boundary condition (default: `Left(QuadraticFit())`)
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`)

# Example
```julia
using LinearAlgebra
x = collect(range(0, 1, 50))
xq = sort(rand(30))
f = randn(50)
y_bar = randn(30)

# Forward
itp = quadratic_interp(x, f; bc=Left(QuadraticFit()))

# Adjoint
adj = quadratic_adjoint(x, xq; bc=Left(QuadraticFit()))
f_bar = adj(y_bar)                      # value adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))    # 1st derivative adjoint
adj(f_bar, y_bar; deriv=DerivOp(1))     # in-place

# Dot-product identity: ⟨W·f, ȳ⟩ = ⟨f, Wᵀȳ⟩
@assert dot(itp.(xq), y_bar) ≈ dot(f, adj(y_bar))
```
"""
function quadratic_adjoint(
        x::AbstractVector,
        x_query::AbstractVector;
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    Tg = _promote_grid_float(eltype(x), eltype(x_query))
    x_p = _to_float(x, Tg)
    xq_p = _to_float(x_query, Tg)

    length(x_p) >= 2 || _throw_adjoint_grid_too_small(length(x_p))

    # Validate PolyFit{D} point requirements
    validate_polyfit_points(bc, length(x_p))

    # Validate domain for NoExtrap
    if extrap isa NoExtrap
        x_lo, x_hi = first(x_p), last(x_p)
        for i in eachindex(xq_p)
            xq_i = xq_p[i]
            (x_lo <= xq_i <= x_hi) || throw(
                DomainError(xq_i, "query point outside domain [$x_lo, $x_hi]")
            )
        end
    end

    # Build spacing and anchors
    spacing = _create_spacing(x_p)
    anchors = _bake_quadratic_adjoint_anchors(x_p, spacing, xq_p, extrap)

    return QuadraticAdjoint(spacing, anchors, bc, length(x_p), x_p)
end

# Scalar query convenience
function quadratic_adjoint(
        x::AbstractVector,
        x_query::Real;
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    return quadratic_adjoint(x, [x_query]; bc = bc, extrap = extrap)
end

# ========================================
# Matrix(itp, xq) convenience — forward matrix from interpolant
# ========================================

"""
    Matrix(itp::QuadraticInterpolant, xq; deriv=EvalValue()) -> Matrix

Materialize the forward interpolation operator as a dense matrix `W` of size
`(n_query, n_grid)`, such that `W * f ≈ itp.(xq; deriv=deriv)` for the linear part.

Internally constructs the adjoint and transposes: `W = Matrix(adj; deriv)'`.

# Example
```julia
itp = quadratic_interp(x, f; bc=Left(QuadraticFit()))
W = Matrix(itp, xq)                       # (n_query × n_grid)
@assert W * f ≈ itp.(xq)                  # for zero-valued BCs
```
"""
function Base.Matrix(
        itp::QuadraticInterpolant, xq::AbstractVector;
        deriv::DerivOp = EvalValue()
    )
    adj = quadratic_adjoint(itp.x, xq; bc = itp.bc, extrap = itp.extrap)
    return Matrix(adj; deriv = deriv)'
end
