# ========================================
# CubicAdjoint: Adjoint (Transpose) Operator
# ========================================
#
# Computes f̄ = Wᵀ · ȳ where W is the implicit forward interpolation matrix.
# The adjoint pipeline reverses the forward: scatter → transpose solve → Rᵀ.
#
# Dependencies (already included before this file):
# - CubicSplineCache, _CubicAnchoredQuery (cubic_types.jl, cubic_anchor.jl)
# - _anchor_query, _fill_anchors! (cubic_anchor.jl)
# - _get_cubic_cache (cubic_cache_pool.jl)
# - _ldiv_tridiagonal_transpose! (thomas_lu_solver.jl)
# - _compute_deriv1_coeffs, _extract_stencil_values (polyfit_kernels.jl)
# - _normalize_bc, _is_periodic_bc (bc_types.jl)
# - _get_h, _get_inv_h (grid_spacing.jl)

# ========================================
# PolyFit Precomputed Data
# ========================================

"""
    _AdjointPolyfitData{L, R}

Precomputed polynomial fit coefficients for the Rᵀ boundary scatter.
Grid-only data, computed once at `CubicAdjoint` construction time.

- `left::L`: `Nothing` for non-PolyFit, or `NTuple{D+1, Tg}` for `PolyFit{D}`
- `right::R`: same
"""
struct _AdjointPolyfitData{L, R}
    left::L
    right::R
end

@inline function _precompute_polyfit_coeffs(::PolyFit{D}, x::AbstractVector{Tg}, ::LeftSide) where {D, Tg}
    x_stencil = _extract_stencil_values(x, LeftSide(), Val(D + 1))
    return _compute_deriv1_coeffs(PolyFit{D}(), LeftSide(), x_stencil)
end

@inline function _precompute_polyfit_coeffs(::PolyFit{D}, x::AbstractVector{Tg}, ::RightSide) where {D, Tg}
    x_stencil = _extract_stencil_values(x, RightSide(), Val(D + 1))
    return _compute_deriv1_coeffs(PolyFit{D}(), RightSide(), x_stencil)
end

@inline _precompute_polyfit_coeffs(::PointBC, ::AbstractVector, ::AbstractSide) = nothing

@inline function _build_polyfit_data(bc::BCPair, x::AbstractVector)
    left = _precompute_polyfit_coeffs(bc.left, x, LeftSide())
    right = _precompute_polyfit_coeffs(bc.right, x, RightSide())
    return _AdjointPolyfitData(left, right)
end

# ========================================
# CubicAdjoint Struct
# ========================================

"""
    CubicAdjoint{Tg, C, BC}

Adjoint (transpose) operator for cubic spline interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward interpolation weight matrix.

Constructed from a grid and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector regardless of value type.

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `C`: `CubicSplineCache` type (reused from forward interpolation)
- `BC`: `BCPair` or `PeriodicBC` (normalized boundary condition)

# Usage
```julia
adj = cubic_adjoint(x_grid, x_query; bc=CubicFit())

# Value adjoint (default)
f_bar = adj(y_bar)

# Derivative adjoint: adjoint of the d-th derivative operator
f_bar = adj(y_bar; deriv=DerivOp(1))   # adjoint of first derivative
f_bar = adj(y_bar; deriv=EvalDeriv2()) # adjoint of second derivative

# In-place
adj(f_bar, y_bar; deriv=DerivOp(1))

# Dimensions
size(adj)    # (n_grid, n_query)
```

# Mathematical Background
Forward: `y = W·f` where `W = Eᵧ + E_z·A⁻¹·R`
Adjoint: `f̄ = Wᵀȳ = Eᵧᵀȳ + Rᵀ·A⁻ᵀ·E_zᵀȳ`

Here `A` is the tridiagonal moment matrix, `R` the finite-difference RHS operator,
`Eᵧ` and `E_z` the evaluation weight matrices for y-values and z-moments respectively.

PolyFit stencil coefficients and periodic `q_transpose = A'^{-T}u` are computed on the
fly at each `adj(ȳ)` call (O(D) and O(n) respectively, negligible vs overall pipeline).
"""
struct CubicAdjoint{Tg <: AbstractFloat, C <: CubicSplineCache{Tg}, BC <: Union{BCPair, PeriodicBC}} <: AbstractAdjoint1D{Tg}
    cache::C
    anchors::Vector{_CubicAnchoredQuery{Tg, Tg}}
    bc::BC
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================
# Callables (6 overloads), Base.size, Base.Matrix, and exclusive periodic
# in-place are inherited from AbstractAdjoint1D via src/core/adjoint_protocol.jl.

@inline _adjoint_output_length(adj::CubicAdjoint) =
    adj.bc isa PeriodicBC{:exclusive} ? length(adj.cache.x) - 1 : length(adj.cache.x)

@inline _n_queries(adj::CubicAdjoint) = length(adj.anchors)

@inline _adjoint_internal_length(adj::CubicAdjoint) = length(adj.cache.x)

@inline _adjoint_1d_has_exclusive_periodic(adj::CubicAdjoint) =
    adj.bc isa PeriodicBC{:exclusive}

@inline _adjoint_1d_apply!(f_bar, adj::CubicAdjoint, y_bar, deriv) =
    _cubic_adjoint_apply!(f_bar, adj, y_bar, deriv)

function _adjoint_1d_finalize(f_bar::AbstractVector, adj::CubicAdjoint)
    n_internal = length(adj.cache.x)
    if adj.bc isa PeriodicBC{:exclusive}
        @inbounds f_bar[1] += f_bar[n_internal]
        return f_bar[1:(n_internal - 1)]
    end
    return f_bar
end

# ========================================
# Core Apply Function
# ========================================

@with_pool pool function _cubic_adjoint_apply!(
        f_bar::AbstractVector{Tv},
        adj::CubicAdjoint{Tg},
        y_bar,  # AbstractVector, Tuple, or scalar-in-tuple
        deriv::DerivOp = EvalValue()
    ) where {Tv, Tg}
    n = length(adj.cache.x)

    # Step 1: Evaluation adjoint scatter — E_dᵧᵀȳ → f̄, E_dzᵀȳ → z̄
    z_bar = zeros!(pool, Tv, n)
    _scatter_eval_adjoint!(f_bar, z_bar, adj.anchors, y_bar, deriv)

    # Step 2: Transpose solve — A⁻ᵀz̄ → r̄ (result in z_bar)
    _ldiv_tridiagonal_transpose!(z_bar, adj.cache.thomas)

    # Step 3: RHS adjoint — f̄ += Rᵀr̄
    # Compute polyfit stencil coefficients on the fly (O(D), grid-only)
    pf = _build_polyfit_data(adj.bc, adj.cache.x)
    _compute_rhs_adjoint!(f_bar, z_bar, adj.cache.spacing, adj.bc, pf)

    return f_bar
end

# ========================================
# Step 1: Evaluation Adjoint Scatter
# ========================================

"""
Scatter query-space sensitivities to grid-space using precomputed anchor weights.
Dispatches on `DerivOp{N}` to select the appropriate weight field from `_CubicAnchoredQuery`.

- `EvalValue`/`EvalDeriv1`: 4-weight scatter (wyL, wyR, wzL, wzR) → f_bar + z_bar
- `EvalDeriv2`/`EvalDeriv3`: 2-weight scatter (wzL, wzR) → z_bar only (y-weights are zero)
"""
function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar,  # AbstractVector or Tuple
        ::EvalValue
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wyL, wyR, wzL, wzR = aq.w0
        f_bar[aq.idx] += wyL * yb
        f_bar[aq.idx + 1] += wyR * yb
        z_bar[aq.idx] += wzL * yb
        z_bar[aq.idx + 1] += wzR * yb
    end
    return nothing
end

function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar,  # AbstractVector or Tuple
        ::EvalDeriv1
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wyL, wyR, wzL, wzR = aq.w1
        f_bar[aq.idx] += wyL * yb
        f_bar[aq.idx + 1] += wyR * yb
        z_bar[aq.idx] += wzL * yb
        z_bar[aq.idx + 1] += wzR * yb
    end
    return nothing
end

function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar,  # AbstractVector or Tuple
        ::EvalDeriv2
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wzL, wzR = aq.w2
        z_bar[aq.idx] += wzL * yb
        z_bar[aq.idx + 1] += wzR * yb
    end
    return nothing
end

function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar,  # AbstractVector or Tuple
        ::EvalDeriv3
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wzL, wzR = aq.w3
        z_bar[aq.idx] += wzL * yb
        z_bar[aq.idx + 1] += wzR * yb
    end
    return nothing
end

# ========================================
# Step 3: RHS Adjoint (Rᵀ)
# ========================================

"""
Apply Rᵀ to `r_bar` and accumulate into `f_bar`.

The forward RHS operator `R` maps `f → r` via finite differences + BC rows.
`Rᵀ` is its transpose: for each `r̄` entry, scatter to `f̄` using transposed stencil.
"""
function _compute_rhs_adjoint!(
        f_bar::AbstractVector, r_bar::AbstractVector,
        spacing::AbstractGridSpacing{Tg},
        bc::BCPair{L, R}, pf::_AdjointPolyfitData
    ) where {Tg, L <: PointBC, R <: PointBC}
    n = length(f_bar) - 1  # n intervals

    # Interior rows (i=2..n): R[i, i-1]=6/h[i-1], R[i,i]=-6(1/h[i-1]+1/h[i]), R[i,i+1]=6/h[i]
    @inbounds for i in 2:n
        c = 6 * r_bar[i]
        f_bar[i - 1] += c * _get_inv_h(spacing, i - 1)
        f_bar[i] -= c * (_get_inv_h(spacing, i - 1) + _get_inv_h(spacing, i))
        f_bar[i + 1] += c * _get_inv_h(spacing, i)
    end

    # Boundary rows — type-dispatched
    _rhs_adjoint_left!(f_bar, r_bar[1], spacing, bc.left, pf.left)
    _rhs_adjoint_right!(f_bar, r_bar[end], spacing, bc.right, pf.right, n)

    return nothing
end

# ----------------------------------------
# Left boundary Rᵀ dispatch
# ----------------------------------------

# Deriv1 left: r₁ = 6·((y₂-y₁)/h₁ - val), so R[1,1] = -6/h₁, R[1,2] = 6/h₁
@inline function _rhs_adjoint_left!(
        f_bar::AbstractVector, r1, spacing::AbstractGridSpacing{Tg}, ::Deriv1, ::Nothing
    ) where {Tg}
    inv_h1 = _get_inv_h(spacing, 1)
    c = 6 * r1
    @inbounds f_bar[1] -= c * inv_h1
    @inbounds f_bar[2] += c * inv_h1
    return nothing
end

# PolyFit left: Deriv1 terms + polyfit stencil. R[1,k] = -6·coeffs[k] for stencil k
@inline function _rhs_adjoint_left!(
        f_bar::AbstractVector, r1, spacing::AbstractGridSpacing{Tg},
        ::PolyFit{D}, coeffs::NTuple{N, Tg}
    ) where {D, N, Tg}
    inv_h1 = _get_inv_h(spacing, 1)
    c = 6 * r1
    # Standard Deriv1 terms
    @inbounds f_bar[1] -= c * inv_h1
    @inbounds f_bar[2] += c * inv_h1
    # PolyFit stencil: left side uses indices 1..D+1
    @inbounds for k in 1:N
        f_bar[k] -= coeffs[k] * c
    end
    return nothing
end

# Deriv2/Deriv3 left: RHS is constant (bc.val or h*bc.val), no f dependency
@inline _rhs_adjoint_left!(::AbstractVector, _, ::AbstractGridSpacing, ::Deriv2, ::Nothing) = nothing
@inline _rhs_adjoint_left!(::AbstractVector, _, ::AbstractGridSpacing, ::Deriv3, ::Nothing) = nothing

# ----------------------------------------
# Right boundary Rᵀ dispatch
# ----------------------------------------

# Deriv1 right: rₙ₊₁ = 6·(val - (yₙ₊₁-yₙ)/hₙ), so R[n+1,n] = 6/hₙ, R[n+1,n+1] = -6/hₙ
@inline function _rhs_adjoint_right!(
        f_bar::AbstractVector, rn1, spacing::AbstractGridSpacing{Tg}, ::Deriv1, ::Nothing, n::Int
    ) where {Tg}
    inv_hn = _get_inv_h(spacing, n)
    c = 6 * rn1
    @inbounds f_bar[n] += c * inv_hn
    @inbounds f_bar[n + 1] -= c * inv_hn
    return nothing
end

# PolyFit right: Deriv1 terms + polyfit stencil.
# Forward: rₙ₊₁ = 6·(d'_polyfit - slope), where d' = Σ coeffs[k]·f[stencil_k].
# ∂r/∂f[stencil_k] = 6·coeffs[k] (positive). R[n+1,stencil_k] = 6·coeffs[k].
@inline function _rhs_adjoint_right!(
        f_bar::AbstractVector, rn1, spacing::AbstractGridSpacing{Tg},
        ::PolyFit{D}, coeffs::NTuple{N, Tg}, n::Int
    ) where {D, N, Tg}
    inv_hn = _get_inv_h(spacing, n)
    c = 6 * rn1
    # Standard Deriv1 terms
    @inbounds f_bar[n] += c * inv_hn
    @inbounds f_bar[n + 1] -= c * inv_hn
    # PolyFit stencil: right side uses indices (n+1-D)...(n+1) = (n+2-N)...(n+1)
    n_pts = n + 1
    @inbounds for k in 1:N
        f_bar[n_pts - N + k] += coeffs[k] * c
    end
    return nothing
end

# Deriv2/Deriv3 right: no-op
@inline _rhs_adjoint_right!(::AbstractVector, _, ::AbstractGridSpacing, ::Deriv2, ::Nothing, ::Int) = nothing
@inline _rhs_adjoint_right!(::AbstractVector, _, ::AbstractGridSpacing, ::Deriv3, ::Nothing, ::Int) = nothing

# ========================================
# ClampExtrap / FillExtrap OOB Anchor Fixup
# ========================================

"""
Fix anchor weights for OOB queries under ClampExtrap or FillExtrap.

- **ClampExtrap**: forward returns `f[boundary]` for EvalValue, `0` for derivatives.
  Keep w0 (clamped to boundary gives correct [1,0,0,0] or [0,1,0,0] weights),
  zero out w1/w2/w3 (derivatives are zero for constant extrapolation).
- **FillExtrap**: forward returns fill constant → gradient w.r.t. f is zero.
  Zero out all weights (w0/w1/w2/w3).

Modifies `anchors` in-place by replacing OOB entries with corrected structs.
Only called at construction time; no runtime overhead.
"""
function _fixup_clampfill_anchors!(
        anchors::Vector{_CubicAnchoredQuery{Tg, Tg}},
        xq_original::AbstractVector{Tg},
        x_lo::Tg, x_hi::Tg,
        extrap::AbstractExtrap
    ) where {Tg}
    keep_w0 = extrap isa ClampExtrap
    z = zero(Tg)
    @inbounds for i in eachindex(anchors)
        (x_lo <= xq_original[i] <= x_hi) && continue
        aq = anchors[i]
        w0_new = keep_w0 ? aq.w0 : (z, z, z, z)
        anchors[i] = _CubicAnchoredQuery{Tg, Tg}(
            aq.idx, aq.xq, 0x00,
            w0_new, (z, z, z, z), (z, z), (z, z)
        )
    end
    return nothing
end

# ========================================
# Constructor
# ========================================

"""
    cubic_adjoint(x, x_query; bc=CubicFit(), autocache=true) -> CubicAdjoint

Create a cubic spline adjoint operator (query-baked, data-free).

Computes `f̄ = Wᵀȳ` where `W` is the forward interpolation weight matrix.
Maps query-space sensitivities back to grid-space sensitivities.

# Arguments
- `x::AbstractVector`: Grid points (must be sorted)
- `x_query::AbstractVector`: Query points (baked into the operator)
- `bc::AbstractBC`: Boundary condition (default: `CubicFit()`)
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`).
  For `WrapExtrap()`, queries are wrapped to the domain before anchoring.
  For `ClampExtrap()`/`FillExtrap()`, queries are clamped to the boundary.
- `autocache::Bool`: Enable automatic caching (default: `true`)

# Example
```julia
using LinearAlgebra
x = collect(range(0, 1, 50))
xq = sort(rand(30))
f = randn(50)

# Forward
itp = cubic_interp(x, f; bc=CubicFit())

# Adjoint
adj = cubic_adjoint(x, xq; bc=CubicFit())
f_bar = adj(y_bar)                      # value adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))    # 1st derivative adjoint
adj(f_bar, y_bar; deriv=DerivOp(1))     # in-place

# Dot-product identity: ⟨W_d·f, ȳ⟩ = ⟨f, W_dᵀȳ⟩
@assert dot(itp.(xq), y_bar) ≈ dot(f, adj(y_bar))
```

# Notes
- `PeriodicBC()` and `PeriodicBC(endpoint=:exclusive)` are supported.
  For periodic BCs, the adjoint uses Sherman-Morrison with the same symmetric factorization.
- The adjoint is the Jacobian transpose (∂y/∂f)ᵀ, NOT an inverse operator.
- For BCs with non-zero values (e.g., `Deriv1(0.5)`), the forward is affine:
  `y = W·f + c`. The adjoint computes `Wᵀȳ`, independent of the constant `c`.
"""
function cubic_adjoint(
        x::AbstractVector,
        x_query::AbstractVector;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        _extra...
    )
    # Promote grid and query to AbstractFloat (handles Integer, Rational, etc.)
    Tg = _promote_grid_float(eltype(x), eltype(x_query))
    x_p = _to_float(x, Tg)
    xq_p = _to_float(x_query, Tg)

    # Periodic path (Sherman-Morrison adjoint)
    if _is_periodic_bc(bc)
        return _build_cubic_adjoint_periodic(x_p, xq_p, bc, autocache)
    end

    # Normalize BC → BCPair (ZeroCurvBC → BCPair(Deriv2(0), Deriv2(0)), etc.)
    bc_pair = _normalize_bc(bc, Tg)

    # Get/build cache (reuses existing infrastructure + autocache)
    cache = _get_cubic_cache(x_p, bc_pair, autocache)

    # Build anchored queries with extrap-specific preprocessing
    wrap = extrap isa WrapExtrap
    if extrap isa Union{ClampExtrap, FillExtrap}
        # Clamp OOB queries to boundary for valid anchor indices, then fix weights
        x_lo, x_hi = first(cache.x), last(cache.x)
        xq_clamped = clamp.(xq_p, x_lo, x_hi)
        anchors = _anchor_query(cache.x, xq_clamped, Val(:cubic), false)
        _fixup_clampfill_anchors!(anchors, xq_p, x_lo, x_hi, extrap)
    else
        anchors = _anchor_query(cache.x, xq_p, Val(:cubic), wrap)
    end

    return CubicAdjoint(cache, anchors, bc_pair)
end

# Scalar query convenience: cubic_adjoint(x, 0.5; ...) → wraps to vector
function cubic_adjoint(
        x::AbstractVector,
        x_query::Real;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        _extra...
    )
    return cubic_adjoint(x, [x_query]; bc = bc, extrap = extrap, autocache = autocache)
end

# ========================================
# Periodic Constructor
# ========================================

function _build_cubic_adjoint_periodic(
        x::AbstractVector{Tg},
        xq::AbstractVector{Tg},
        bc::PeriodicBC,
        autocache::Bool
    ) where {Tg <: AbstractFloat}

    # Extend exclusive → inclusive grid (grid-only, no y-data needed)
    x_ext = if bc isa PeriodicBC{:exclusive}
        period = _resolve_exclusive_period(x, bc)
        x_end = first(x) + Tg(period)
        if x isa AbstractRange
            range(first(x), step = step(x), length = length(x) + 1)
        else
            vcat(x, x_end)
        end
    else
        x
    end

    # Get/build periodic cache (Thomas factorization + PeriodicData{q, period})
    cache = _get_cubic_cache(x_ext, PeriodicBC(), autocache)

    # Build anchored queries with wrapping (queries outside domain → wrap to [x[1], x[end]))
    anchors = _anchor_query(cache.x, xq, Val(:cubic), true)

    # Store resolved period in BC for display/introspection
    bc_display = _with_resolved_period(bc, cache.bc_config.period)

    return CubicAdjoint(cache, anchors, bc_display)
end

# ========================================
# Periodic Apply Pipeline
# ========================================

@with_pool pool function _cubic_adjoint_apply!(
        f_bar::AbstractVector{Tv},
        adj::CubicAdjoint{Tg, <:CubicSplineCache{Tg}, <:PeriodicBC},
        y_bar,  # AbstractVector, Tuple, or scalar-in-tuple
        deriv::DerivOp = EvalValue()
    ) where {Tv, Tg}
    n = length(adj.cache.x) - 1  # n intervals, n+1 grid points

    # Step 1: Evaluation adjoint scatter — E_dᵧᵀȳ → f̄[1..n+1], E_dzᵀȳ → z̄[1..n+1]
    # Reuse non-periodic scatter (n+1-element arrays, idx+1=n+1 is valid index)
    z_bar = zeros!(pool, Tv, n + 1)
    _scatter_eval_adjoint!(f_bar, z_bar, adj.anchors, y_bar, deriv)

    # Step 2: Fold z̄[n+1] → z̄[1] (forward has z[n+1]=z[1], adjoint sums contributions)
    @inbounds z_bar[1] += z_bar[n + 1]

    # Step 3: Sherman-Morrison adjoint solve — (A'+αuuᵀ)⁻ᵀz̄ → r̄
    # Compute q_t = A'^{-T} u on the fly (pool-allocated, zero-alloc)
    q_t = acquire!(pool, Tg, n)
    fill!(q_t, zero(Tg))
    @inbounds q_t[1] = one(Tg)
    @inbounds q_t[n] = one(Tg)
    _ldiv_tridiagonal_transpose!(q_t, adj.cache.thomas)

    _adjoint_periodic_solve!(z_bar, adj.cache, q_t, n)

    # Step 4: Rᵀ_circ r̄ → f̄ += Rᵀ·r̄ (accumulates into f_bar[1..n+1])
    _compute_rhs_adjoint_periodic!(f_bar, z_bar, adj.cache.spacing, n)

    return f_bar
end

# ========================================
# Periodic Step 3: Sherman-Morrison Adjoint Solve
# ========================================

"""
Adjoint Sherman-Morrison solve for periodic cubic spline.

The periodic matrix A_cyc = A' + α·u·uᵀ is NOT symmetric for non-uniform grids
(A'[i,i-1] = h[i-1] ≠ h[i] = A'[i,i+1]). The transpose inverse uses:
  A_cyc⁻ᵀ x = y_t - [α·uᵀy_t / (1 + α·uᵀq_t)] · q_t
where y_t = A'⁻ᵀx and q_t = A'⁻ᵀu (precomputed at construction).

Operates in-place on `z_bar[1:n]`; `z_bar[n+1]` is not touched.
"""
function _adjoint_periodic_solve!(
        z_bar::AbstractVector{Tv},
        cache::CubicSplineCache{Tg, X, F, PeriodicData{Tg}, S},
        q_t::AbstractVector,
        n::Int
    ) where {Tv, Tg <: AbstractFloat, X, F, S <: AbstractGridSpacing{Tg}}

    # Transpose Thomas solve on z_bar[1:n]
    _ldiv_tridiagonal_transpose!(z_bar, cache.thomas)

    # Sherman-Morrison correction with q_t = A'^{-T} u
    α = Tv(_get_h(cache.spacing, n))

    @inbounds begin
        vTz = α * (z_bar[1] + z_bar[n])
        vTq = α * (Tv(q_t[1]) + Tv(q_t[n]))
        factor = vTz * inv(one(Tv) + vTq)
        for i in 1:n
            z_bar[i] -= factor * Tv(q_t[i])
        end
    end

    return nothing
end

# ========================================
# Periodic Step 4: Rᵀ (Circulant Tridiagonal Transpose)
# ========================================

"""
Apply Rᵀ to `r_bar[1:n]` and accumulate into `f_bar[1:n+1]` for periodic BC.
R is n×(n+1) with wrapping entries at rows 1 and n.
"""
function _compute_rhs_adjoint_periodic!(
        f_bar::AbstractVector, r_bar::AbstractVector,
        spacing::AbstractGridSpacing{Tg}, n::Int
    ) where {Tg}

    # Interior rows (i=2..n-1): standard tridiagonal stencil
    @inbounds for i in 2:(n - 1)
        c = 6 * r_bar[i]
        f_bar[i - 1] += c * _get_inv_h(spacing, i - 1)
        f_bar[i] -= c * (_get_inv_h(spacing, i - 1) + _get_inv_h(spacing, i))
        f_bar[i + 1] += c * _get_inv_h(spacing, i)
    end

    # Row 1: R[1,n]=6/hₙ, R[1,1]=-6(1/hₙ+1/h₁), R[1,2]=6/h₁
    @inbounds begin
        c1 = 6 * r_bar[1]
        f_bar[n] += c1 * _get_inv_h(spacing, n)
        f_bar[1] -= c1 * (_get_inv_h(spacing, n) + _get_inv_h(spacing, 1))
        f_bar[2] += c1 * _get_inv_h(spacing, 1)
    end

    # Row n: R[n,n-1]=6/hₙ₋₁, R[n,n]=-6(1/hₙ₋₁+1/hₙ), R[n,n+1]=6/hₙ
    @inbounds begin
        cn = 6 * r_bar[n]
        f_bar[n - 1] += cn * _get_inv_h(spacing, n - 1)
        f_bar[n] -= cn * (_get_inv_h(spacing, n - 1) + _get_inv_h(spacing, n))
        f_bar[n + 1] += cn * _get_inv_h(spacing, n)
    end

    return nothing
end

# Matrix(adj::CubicAdjoint) inherited from AbstractAdjoint (adjoint_protocol.jl)

"""
    Matrix(itp::CubicInterpolant, xq; deriv=EvalValue()) -> Matrix

Materialize the forward interpolation operator as a dense matrix `W` of size
`(n_query, n_grid)`, such that `W * f ≈ itp.(xq; deriv=deriv)` for the linear part.

Internally constructs the adjoint and transposes: `W = Matrix(adj; deriv)'`.

# Example
```julia
itp = cubic_interp(x, f; bc=CubicFit())
W = Matrix(itp, xq)                       # (n_query × n_grid)
@assert W * f ≈ itp.(xq)                  # for zero-valued BCs
```
"""
function Base.Matrix(
        itp::CubicInterpolant, xq::AbstractVector;
        deriv::DerivOp = EvalValue()
    )
    adj = cubic_adjoint(itp.cache.x, xq; bc = itp.bc, extrap = itp.extrap)
    return Matrix(adj; deriv = deriv)'
end
