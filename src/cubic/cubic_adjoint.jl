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
# - _ldiv_tridiagonal_nopiv!, _ldiv_tridiagonal_transpose! (thomas_lu_solver.jl)
# - _compute_deriv1_coeffs, _extract_stencil_values (polyfit_kernels.jl)
# - _normalize_bc, _is_periodic_bc (bc_types.jl)
# - _get_h, _get_inv_h (grid_spacing.jl)

# ========================================
# Thomas Matrix Symmetry Trait
# ========================================

# Deriv1 and PolyFit produce boundary rows (2h₁, h₁) / (hₙ, 2hₙ)
# that are symmetric with the interior tridiagonal pattern.
# Deriv2 produces (1, 0) / (0, 1) and Deriv3 produces (-1, 1) / (-1, 1),
# which break symmetry.

@inline _is_symmetric_thomas(::Deriv1) = true
@inline _is_symmetric_thomas(::PolyFit) = true
@inline _is_symmetric_thomas(::Deriv2) = false
@inline _is_symmetric_thomas(::Deriv3) = false
@inline _is_symmetric_thomas(bc::BCPair) = _is_symmetric_thomas(bc.left) && _is_symmetric_thomas(bc.right)

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
    CubicAdjoint{Tg, C, BC, Sym, PF}

Adjoint (transpose) operator for cubic spline interpolation.
Computes `f̄ = Wᵀȳ` where `W` is the forward interpolation weight matrix.

Constructed from a grid and query points (query-baked, data-free).
The same adjoint can be applied to any `ȳ` vector regardless of value type.

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `C`: `CubicSplineCache` type (reused from forward interpolation)
- `BC`: `BCPair` type (normalized boundary condition)
- `Sym`: `Val{true}` or `Val{false}` — compile-time tridiagonal symmetry flag
- `PF`: `_AdjointPolyfitData` type (precomputed PolyFit stencil coefficients)

# Usage
```julia
adj = cubic_adjoint(x_grid, x_query; bc=CubicFit())

# Allocating
f_bar = adj(y_bar)

# In-place
adj(f_bar, y_bar)

# Dimensions
size(adj)    # (n_grid, n_query)
```

# Mathematical Background
Forward: `y = W·f` where `W = Eᵧ + E_z·A⁻¹·R`
Adjoint: `f̄ = Wᵀȳ = Eᵧᵀȳ + Rᵀ·A⁻ᵀ·E_zᵀȳ`

Here `A` is the tridiagonal moment matrix, `R` the finite-difference RHS operator,
`Eᵧ` and `E_z` the evaluation weight matrices for y-values and z-moments respectively.
"""
struct CubicAdjoint{Tg <: AbstractFloat, C <: CubicSplineCache{Tg}, BC <: BCPair, Sym, PF <: _AdjointPolyfitData}
    cache::C
    anchors::Vector{_CubicAnchoredQuery{Tg, Tg}}
    bc::BC
    polyfit_data::PF

    function CubicAdjoint(
            cache::C,
            anchors::Vector{_CubicAnchoredQuery{Tg, Tg}},
            bc::BC,
            polyfit_data::PF,
            ::Sym
        ) where {Tg <: AbstractFloat, C <: CubicSplineCache{Tg}, BC <: BCPair, Sym <: Val, PF <: _AdjointPolyfitData}
        return new{Tg, C, BC, Sym, PF}(cache, anchors, bc, polyfit_data)
    end
end

Base.size(adj::CubicAdjoint) = (length(adj.cache.x), length(adj.anchors))
Base.size(adj::CubicAdjoint, d::Integer) = size(adj)[d]

# ========================================
# Callable Methods
# ========================================

"""
    (adj::CubicAdjoint)(y_bar) -> f_bar

Apply the adjoint operator: `f̄ = Wᵀȳ`. Allocating version.

The output element type is `promote_type(eltype(y_bar), Tg)`.
"""
function (adj::CubicAdjoint{Tg})(y_bar::AbstractVector) where {Tg}
    @assert length(y_bar) == length(adj.anchors) "y_bar length ($(length(y_bar))) must match n_query ($(length(adj.anchors)))"
    Tv = promote_type(eltype(y_bar), Tg)
    f_bar = zeros(Tv, length(adj.cache.x))
    _cubic_adjoint_apply!(f_bar, adj, y_bar)
    return f_bar
end

"""
    (adj::CubicAdjoint)(f_bar, y_bar) -> f_bar

Apply the adjoint operator in-place: `f̄ = Wᵀȳ`.
Zeros `f_bar` before accumulating.
"""
function (adj::CubicAdjoint{Tg})(f_bar::AbstractVector, y_bar::AbstractVector) where {Tg}
    @assert length(f_bar) == length(adj.cache.x) "f_bar length ($(length(f_bar))) must match n_grid ($(length(adj.cache.x)))"
    @assert length(y_bar) == length(adj.anchors) "y_bar length ($(length(y_bar))) must match n_query ($(length(adj.anchors)))"
    fill!(f_bar, zero(eltype(f_bar)))
    _cubic_adjoint_apply!(f_bar, adj, y_bar)
    return f_bar
end

# ========================================
# Core Apply Function
# ========================================

function _cubic_adjoint_apply!(
        f_bar::AbstractVector{Tv},
        adj::CubicAdjoint{Tg, C, BC, Sym, PF},
        y_bar::AbstractVector
    ) where {Tv, Tg, C, BC, Sym, PF}
    n = length(adj.cache.x)

    # Step 1: Evaluation adjoint scatter — Eᵧᵀȳ → f̄, E_zᵀȳ → z̄
    z_bar = zeros(Tv, n)
    _scatter_eval_adjoint!(f_bar, z_bar, adj.anchors, y_bar)

    # Step 2: Transpose solve — A⁻ᵀz̄ → r̄ (result in z_bar)
    _adjoint_thomas_solve!(z_bar, adj.cache.thomas, Sym())

    # Step 3: RHS adjoint — f̄ += Rᵀr̄
    _compute_rhs_adjoint!(f_bar, z_bar, adj.cache.spacing, adj.bc, adj.polyfit_data)

    return f_bar
end

# ========================================
# Step 1: Evaluation Adjoint Scatter
# ========================================

"""
Scatter query-space sensitivities to grid-space using precomputed anchor weights.
Reuses the `w0 = (wyL, wyR, wzL, wzR)` field from `_CubicAnchoredQuery`.
"""
function _scatter_eval_adjoint!(
        f_bar::AbstractVector, z_bar::AbstractVector,
        anchors::Vector{<:_CubicAnchoredQuery}, y_bar::AbstractVector
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

# ========================================
# Step 2: Transpose Solve Dispatch
# ========================================

# Symmetric A (Deriv1, PolyFit): forward solve = transpose solve
@inline _adjoint_thomas_solve!(z_bar, thomas, ::Val{true}) =
    _ldiv_tridiagonal_nopiv!(z_bar, thomas)

# Asymmetric A (Deriv2, Deriv3): use transpose solve
@inline _adjoint_thomas_solve!(z_bar, thomas, ::Val{false}) =
    _ldiv_tridiagonal_transpose!(z_bar, thomas)

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
f_bar = adj(y_bar)       # allocating
adj(f_bar, y_bar)         # in-place

# Dot-product identity: ⟨W·f, ȳ⟩ = ⟨f, Wᵀȳ⟩
@assert dot(itp.(xq), y_bar) ≈ dot(f, adj(y_bar))
```

# Notes
- `PeriodicBC` is not supported in v1. Use `BCPair`-based BCs.
- The adjoint is the Jacobian transpose (∂y/∂f)ᵀ, NOT an inverse operator.
- For BCs with non-zero values (e.g., `Deriv1(0.5)`), the forward is affine:
  `y = W·f + c`. The adjoint computes `Wᵀȳ`, independent of the constant `c`.
"""
function cubic_adjoint(
        x::AbstractVector,
        x_query::AbstractVector;
        bc::AbstractBC = CubicFit(),
        autocache::Bool = true
    )
    # Reject PeriodicBC (deferred to v2)
    if _is_periodic_bc(bc)
        throw(ArgumentError(
            "PeriodicBC is not yet supported by cubic_adjoint. " *
                "Use BCPair-based boundary conditions (CubicFit, ZeroCurvBC, Deriv1, etc.)."
        ))
    end

    # Promote grid and query to AbstractFloat (handles Integer, Rational, etc.)
    Tg = _promote_grid_float(eltype(x), eltype(x_query))
    x_p = _to_float(x, Tg)
    xq_p = _to_float(x_query, Tg)

    # Normalize BC → BCPair (ZeroCurvBC → BCPair(Deriv2(0), Deriv2(0)), etc.)
    bc_pair = _normalize_bc(bc, Tg)

    # Get/build cache (reuses existing infrastructure + autocache)
    cache = _get_cubic_cache(x_p, bc_pair, autocache)

    # Build anchored queries (reuses existing anchor builder)
    anchors = _anchor_query(cache.x, xq_p, Val(:cubic))

    # Compile-time symmetry flag
    sym = Val(_is_symmetric_thomas(bc_pair))

    # Precompute PolyFit stencil coefficients (grid-only, computed once)
    pf = _build_polyfit_data(bc_pair, cache.x)

    return CubicAdjoint(cache, anchors, bc_pair, pf, sym)
end
