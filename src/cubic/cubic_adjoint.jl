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
struct CubicAdjoint{Tg, C <: CubicSplineCache{Tg}, BC <: Union{BCPair, PeriodicBC}, I <: _AbstractIndices{2}} <: AbstractAdjoint1D{Tg}
    cache::C
    # Coordinate type is grid-pinned (`Tc = Tg`), not the canonical `_coord_eltype(Tq, Tg)`:
    # the adjoint operates on baked coefficients, so AD-through-adjoint is unsupported. The
    # forward Dual-grid contract is satisfied independently.
    anchors::Vector{_CubicAnchoredQuery{Tg, Tg, I}}
    bc::BC
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================
# Callables (6 overloads), Base.size, Base.Matrix, and exclusive periodic
# in-place are inherited from AbstractAdjoint1D via src/core/adjoint_protocol.jl.

@inline _adjoint_output_length(adj::CubicAdjoint) =
    _is_periodic_seam_folded(adj.bc) ? length(adj.cache.x) - 1 : length(adj.cache.x)

@inline _n_queries(adj::CubicAdjoint) = length(adj.anchors)

@inline _adjoint_internal_length(adj::CubicAdjoint) = length(adj.cache.x)

@inline _adjoint_1d_has_seam_fold(adj::CubicAdjoint) =
    _is_periodic_seam_folded(adj.bc)

@inline _adjoint_1d_apply!(f_bar, adj::CubicAdjoint, y_bar, deriv) =
    _cubic_adjoint_apply!(f_bar, adj, y_bar, deriv)

# `_adjoint_1d_finalize` falls through to the protocol default, which dispatches
# on `adj.bc` and uses `_adjoint_internal_length(adj)` — CubicAdjoint's override
# of `_adjoint_internal_length` (`length(adj.cache.x)`) supplies the right size.

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
    _compute_rhs_adjoint!(f_bar, z_bar, adj.cache.x, adj.bc, pf)

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
@inline function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar,  # AbstractVector or Tuple
        ::EvalValue
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wyL, wyR, wzL, wzR = aq.w0
        idxL = aq.idxL
        idxR = aq.idxR
        f_bar[idxL] += wyL * yb
        f_bar[idxR] += wyR * yb
        z_bar[idxL] += wzL * yb
        z_bar[idxR] += wzR * yb
    end
    return nothing
end

@inline function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar,  # AbstractVector or Tuple
        ::EvalDeriv1
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wyL, wyR, wzL, wzR = aq.w1
        idxL = aq.idxL
        idxR = aq.idxR
        f_bar[idxL] += wyL * yb
        f_bar[idxR] += wyR * yb
        z_bar[idxL] += wzL * yb
        z_bar[idxR] += wzR * yb
    end
    return nothing
end

@inline function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar,  # AbstractVector or Tuple
        ::EvalDeriv2
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wzL, wzR = aq.w2
        z_bar[aq.idxL] += wzL * yb
        z_bar[aq.idxR] += wzR * yb
    end
    return nothing
end

@inline function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar,  # AbstractVector or Tuple
        ::EvalDeriv3
    )
    @inbounds for q in eachindex(y_bar)
        aq = anchors[q]
        yb = y_bar[q]
        wzL, wzR = aq.w3
        z_bar[aq.idxL] += wzL * yb
        z_bar[aq.idxR] += wzR * yb
    end
    return nothing
end

# Generic fallback: 4th+ derivative of cubic is zero → no scatter
@inline function _scatter_eval_adjoint!(
        ::AbstractVector, ::AbstractVector,
        ::Vector{<:_CubicAnchoredQuery}, ::Any,
        ::DerivOp{N}
    ) where {N}
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
@inline function _compute_rhs_adjoint!(
        f_bar::AbstractVector, r_bar::AbstractVector,
        x::AbstractVector{Tg},
        bc::BCPair{L, R}, pf::_AdjointPolyfitData
    ) where {Tg, L <: PointBC, R <: PointBC}
    n = length(f_bar) - 1  # n intervals

    # Interior rows (i=2..n): R[i, i-1]=6/h[i-1], R[i,i]=-6(1/h[i-1]+1/h[i]), R[i,i+1]=6/h[i]
    @inbounds for i in 2:n
        c = 6 * r_bar[i]
        f_bar[i - 1] += c * _get_inv_h(x, i - 1)
        f_bar[i] -= c * (_get_inv_h(x, i - 1) + _get_inv_h(x, i))
        f_bar[i + 1] += c * _get_inv_h(x, i)
    end

    # Boundary rows — type-dispatched
    _rhs_adjoint_left!(f_bar, r_bar[1], x, bc.left, pf.left)
    _rhs_adjoint_right!(f_bar, r_bar[end], x, bc.right, pf.right, n)

    return nothing
end

# ----------------------------------------
# Left boundary Rᵀ dispatch
# ----------------------------------------

# Deriv1 left: r₁ = 6·((y₂-y₁)/h₁ - val), so R[1,1] = -6/h₁, R[1,2] = 6/h₁
@inline function _rhs_adjoint_left!(
        f_bar::AbstractVector, r1, x::AbstractVector{Tg}, ::Deriv1, ::Nothing
    ) where {Tg}
    inv_h1 = _get_inv_h(x, 1)
    c = 6 * r1
    @inbounds f_bar[1] -= c * inv_h1
    @inbounds f_bar[2] += c * inv_h1
    return nothing
end

# PolyFit left: Deriv1 terms + polyfit stencil. R[1,k] = -6·coeffs[k] for stencil k
@inline function _rhs_adjoint_left!(
        f_bar::AbstractVector, r1, x::AbstractVector{Tg},
        ::PolyFit{D}, coeffs::NTuple{N, Tg}
    ) where {D, N, Tg}
    inv_h1 = _get_inv_h(x, 1)
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
@inline _rhs_adjoint_left!(::AbstractVector, _, ::AbstractVector, ::Deriv2, ::Nothing) = nothing
@inline _rhs_adjoint_left!(::AbstractVector, _, ::AbstractVector, ::Deriv3, ::Nothing) = nothing

# ----------------------------------------
# Right boundary Rᵀ dispatch
# ----------------------------------------

# Deriv1 right: rₙ₊₁ = 6·(val - (yₙ₊₁-yₙ)/hₙ), so R[n+1,n] = 6/hₙ, R[n+1,n+1] = -6/hₙ
@inline function _rhs_adjoint_right!(
        f_bar::AbstractVector, rn1, x::AbstractVector{Tg}, ::Deriv1, ::Nothing, n::Int
    ) where {Tg}
    inv_hn = _get_inv_h(x, n)
    c = 6 * rn1
    @inbounds f_bar[n] += c * inv_hn
    @inbounds f_bar[n + 1] -= c * inv_hn
    return nothing
end

# PolyFit right: Deriv1 terms + polyfit stencil.
# Forward: rₙ₊₁ = 6·(d'_polyfit - slope), where d' = Σ coeffs[k]·f[stencil_k].
# ∂r/∂f[stencil_k] = 6·coeffs[k] (positive). R[n+1,stencil_k] = 6·coeffs[k].
@inline function _rhs_adjoint_right!(
        f_bar::AbstractVector, rn1, x::AbstractVector{Tg},
        ::PolyFit{D}, coeffs::NTuple{N, Tg}, n::Int
    ) where {D, N, Tg}
    inv_hn = _get_inv_h(x, n)
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
@inline _rhs_adjoint_right!(::AbstractVector, _, ::AbstractVector, ::Deriv2, ::Nothing, ::Int) = nothing
@inline _rhs_adjoint_right!(::AbstractVector, _, ::AbstractVector, ::Deriv3, ::Nothing, ::Int) = nothing

# ========================================
# ClampExtrap / FillExtrap OOB Anchor Fixup
# ========================================

"""
    _bake_cubic_clampfill_anchors(x, xq, extrap) -> Vector{_CubicAnchoredQuery}

Single-pass ClampExtrap/FillExtrap adjoint anchor builder.

Clamps each query to the actual grid endpoints (`_clamp_to_grid`) for valid
boundary-cell geometry, anchors it, then bakes the OOB weight semantics for
genuinely-OOB queries (widened `_is_inbounds` classification):

- **ClampExtrap**: forward returns `f[boundary]` for EvalValue, `0` for
  derivatives. Keep w0 (clamped boundary weights = [1,0,0,0] or [0,1,0,0]),
  zero w1/w2/w3.
- **FillExtrap**: forward returns fill constant → gradient w.r.t. f is zero.
  Zero all weights.

In-domain and endpoint-sliver queries keep the anchor as built. Fuses the
former clamp-broadcast + `_anchor_query` + weight-fixup three passes into one
loop — no transient clamped-query array. Construction-time only.
"""
function _bake_cubic_clampfill_anchors(
        x::AbstractVector{T},
        xq::AbstractVector{S},
        extrap::AbstractExtrap,
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {T, S <: Real, P <: Searcher}
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)
    keep_w0 = extrap isa ClampExtrap
    z = zero(T)
    z4 = (z, z, z, z)
    output = Vector{_CubicAnchoredQuery{T, T, _interval_type(x)}}(undef, length(xq))
    @inbounds for k in eachindex(xq)
        xq_raw = xq[k]
        aq = _anchor_query_impl(x, _promote_coord(_clamp_to_grid(xq_raw, x), T), false, searcher_resolved)
        if _is_inbounds(x, xq_raw)
            output[k] = aq
        else
            w0_new = keep_w0 ? aq.w0 : z4
            output[k] = _CubicAnchoredQuery{T, T, _interval_type(x)}(
                getfield(aq, :interval), aq.xq, IN_DOMAIN,
                w0_new, z4, (z, z), (z, z)
            )
        end
    end
    return output
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
    )
    x_p, xq_p, Tg = _promote_adjoint_inputs(x, x_query)

    # Periodic path (Sherman-Morrison adjoint)
    if _is_periodic_bc(bc)
        return _build_cubic_adjoint_periodic(x_p, xq_p, bc, autocache)
    end

    # Normalize BC → BCPair (ZeroCurvBC → BCPair(Deriv2(0), Deriv2(0)), etc.)
    bc_pair = _normalize_bc(bc)

    # Get/build cache (reuses existing infrastructure + autocache)
    cache = _get_cubic_cache(x_p, bc_pair, _effective_autocache(autocache, eltype(x_p)))

    # Build anchored queries with extrap-specific preprocessing
    wrap = extrap isa WrapExtrap
    if extrap isa Union{ClampExtrap, FillExtrap}
        # OOB queries get actual-endpoint geometry; the widened (`_is_inbounds`)
        # classification bakes OOB weights. Single fused pass (no temp array).
        anchors = _bake_cubic_clampfill_anchors(cache.x, xq_p, extrap)
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
    )
    return cubic_adjoint(x, [x_query]; bc = bc, extrap = extrap, autocache = autocache)
end

# ========================================
# Periodic Constructor
# ========================================

function _build_cubic_adjoint_periodic(
        x::AbstractVector{Tg},
        xq::AbstractVector{Tq},
        bc::PeriodicBC,
        autocache::Bool
    ) where {Tg, Tq <: Real}

    # Extend exclusive → inclusive grid (grid-only, no y-data needed)
    x_ext = if bc isa PeriodicBC{:exclusive}
        period = _resolve_exclusive_period(x, bc)
        x_end = first(x) + Tg(period)
        if x isa AbstractRange
            _to_float_adding_endpoint(x, Tg)
        else
            vcat(x, x_end)
        end
    else
        x
    end

    # Cache built with `_bc_after_extend(bc)` → cache.bc is :extended for
    # promoted :exclusive, :inclusive for direct user input.
    cache = _get_cubic_cache(x_ext, _bc_after_extend(bc), _effective_autocache(autocache, Tg))

    # Build anchored queries with wrapping (queries outside closed domain → wrap to [x[1], x[end]])
    anchors = _anchor_query(cache.x, xq, Val(:cubic), true)

    return CubicAdjoint(cache, anchors, cache.bc)
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
    _compute_rhs_adjoint_periodic!(f_bar, z_bar, adj.cache.x, n)

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
        cache::CubicSplineCache{Tg, X, F, <:PeriodicBC},
        q_t::AbstractVector,
        n::Int
    ) where {Tv, Tg, X, F}

    # Transpose Thomas solve on z_bar[1:n]
    _ldiv_tridiagonal_transpose!(z_bar, cache.thomas)

    # Sherman-Morrison correction with q_t = A'^{-T} u
    α = Tv(_get_h(cache.x, n))

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
@inline function _compute_rhs_adjoint_periodic!(
        f_bar::AbstractVector, r_bar::AbstractVector,
        x::AbstractVector{Tg}, n::Int
    ) where {Tg}

    # Interior rows (i=2..n-1): standard tridiagonal stencil
    @inbounds for i in 2:(n - 1)
        c = 6 * r_bar[i]
        f_bar[i - 1] += c * _get_inv_h(x, i - 1)
        f_bar[i] -= c * (_get_inv_h(x, i - 1) + _get_inv_h(x, i))
        f_bar[i + 1] += c * _get_inv_h(x, i)
    end

    # Row 1: R[1,n]=6/hₙ, R[1,1]=-6(1/hₙ+1/h₁), R[1,2]=6/h₁
    @inbounds begin
        c1 = 6 * r_bar[1]
        f_bar[n] += c1 * _get_inv_h(x, n)
        f_bar[1] -= c1 * (_get_inv_h(x, n) + _get_inv_h(x, 1))
        f_bar[2] += c1 * _get_inv_h(x, 1)
    end

    # Row n: R[n,n-1]=6/hₙ₋₁, R[n,n]=-6(1/hₙ₋₁+1/hₙ), R[n,n+1]=6/hₙ
    @inbounds begin
        cn = 6 * r_bar[n]
        f_bar[n - 1] += cn * _get_inv_h(x, n - 1)
        f_bar[n] -= cn * (_get_inv_h(x, n - 1) + _get_inv_h(x, n))
        f_bar[n + 1] += cn * _get_inv_h(x, n)
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
