# ========================================
# ND Cubic Adjoint: Implementation
# ========================================
#
# Computes f̄ = Wᵀ · ȳ for N-dimensional cubic Hermite interpolation.
#
# Pipeline (adj(y_bar) → f_bar):
#   Step 0: Eval adjoint scatter  — y_bar → partials_bar (Hermite weight products)
#   Step 1: Build adjoint (d=N..1, reverse order):
#           a. moments_to_deriv_adjoint (dy_bar → z_bar, f_contrib)
#           b. transpose Thomas solve (A⁻ᵀ z_bar → r_bar)
#           c. RHS stencil adjoint (f_contrib += Rᵀ r_bar)
#           d. accumulate: partials_bar[p_src] += f_contrib
#   Step 2: Extract f_bar = partials_bar[1, ...]
#
# Dependencies (included before this file):
# - _NDAdjointAnchor, CubicAdjointND (cubic_nd_adjoint_types.jl)
# - _hermite_h00/10/01/11 (cubic_nd_math.jl)
# - _get_effective_bc (cubic_nd_build.jl)
# - _ldiv_tridiagonal_transpose! (thomas_lu_solver.jl)
# - _compute_rhs_adjoint!, _build_polyfit_data (cubic_adjoint.jl)
# - _get_cubic_cache (cubic_cache_pool.jl)
# - _normalize_bc, _resolve_bcs_nd (bc_types.jl, nd_utils.jl)
# - search_interval, DEFAULT_SEARCHER (search.jl)

# ========================================
# Moments-to-Derivatives Adjoint
# ========================================

"""
    _moments_to_deriv_adjoint_1d!(z_bar, f_contrib, dy_bar, spacing)

Adjoint of `_moments_to_derivatives_1d!` (cubic_nd_math.jl).

Given sensitivities w.r.t. derivatives (`dy_bar`), computes:
- `z_bar`: sensitivities w.r.t. moments (second derivatives)
- `f_contrib`: sensitivities w.r.t. function values

Both outputs are written (not accumulated), so they are overwritten.

The forward function computes:
- `dydx[1] = (y[2] - y[1]) * inv_h₁ - h₁/6 * (2m[1] + m[2])`     (right deriv)
- `dydx[i+1] = (y[i+1] - y[i]) * inv_hᵢ + hᵢ/6 * (m[i] + 2m[i+1])` (left deriv)
"""
function _moments_to_deriv_adjoint_1d!(
        z_bar::AbstractVector{Tv},
        f_contrib::AbstractVector{Tv},
        dy_bar::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg}
    ) where {Tv, Tg}
    n = length(dy_bar)

    fill!(z_bar, zero(Tv))
    fill!(f_contrib, zero(Tv))

    inv_6 = inv(Tg(6))

    # First point adjoint (right derivative: dydx[1])
    @inbounds begin
        h1 = _get_h(spacing, 1)
        inv_h1 = _get_inv_h(spacing, 1)
        h_over_6 = h1 * inv_6
        h_over_3 = h_over_6 * 2
        db = dy_bar[1]
        f_contrib[1] -= db * inv_h1
        f_contrib[2] += db * inv_h1
        z_bar[1] -= db * h_over_3    # -h/6 * 2 = -h/3
        z_bar[2] -= db * h_over_6    # -h/6 * 1
    end

    # Interior + last points adjoint (left derivative: dydx[i+1], i=1..n-1)
    @inbounds for i in 1:(n - 1)
        h = _get_h(spacing, i)
        inv_h = _get_inv_h(spacing, i)
        h_over_6 = h * inv_6
        h_over_3 = h_over_6 * 2
        db = dy_bar[i + 1]
        f_contrib[i] -= db * inv_h
        f_contrib[i + 1] += db * inv_h
        z_bar[i] += db * h_over_6     # +h/6 * 1
        z_bar[i + 1] += db * h_over_3 # +h/6 * 2 = +h/3
    end

    return nothing
end

# ========================================
# Anchor Baking
# ========================================

"""
    _compute_nd_anchor_weights(t, h, inv_h) -> (w0, w1, w2, w3)

Compute all 4 derivative-order weight tuples for one axis at normalized position `t`.
Each `wk` is `(w_fL, w_fR, w_dyL, w_dyR)` for DerivOp(k).

Matches the 1D `_CubicAnchoredQuery` pattern: bake all weights at construction so
the adjoint `adj(ȳ)` hot path does zero weight computation.
"""
@inline function _compute_nd_anchor_weights(t::Tg, h::Tg, inv_h::Tg) where {Tg}
    t_sq = t * t

    # k=0: EvalValue — P(t) = h00·fL + h01·fR + h10·h·dyL + h11·h·dyR
    w0 = (_hermite_h00(t), _hermite_h01(t), _hermite_h10(t) * h, _hermite_h11(t) * h)

    # k=1: EvalDeriv1 — dP/dx = (dP/dt) · inv_h
    dh00 = muladd(6, t_sq, -6 * t)                     # 6t² - 6t
    dh10 = muladd(3, t_sq, muladd(-4, t, one(Tg)))     # 3t² - 4t + 1
    dh01 = muladd(-6, t_sq, 6 * t)                     # -6t² + 6t
    dh11 = muladd(3, t_sq, -2 * t)                     # 3t² - 2t
    w1 = (dh00 * inv_h, dh01 * inv_h, dh10, dh11)

    # k=2: EvalDeriv2 — d²P/dx² = (d²P/dt²) · inv_h²
    inv_h2 = inv_h * inv_h
    d2h00 = muladd(12, t, -6)    # 12t - 6
    d2h10 = muladd(6, t, -4)     # 6t - 4
    d2h01 = muladd(-12, t, 6)    # -12t + 6
    d2h11 = muladd(6, t, -2)     # 6t - 2
    w2 = (d2h00 * inv_h2, d2h01 * inv_h2, d2h10 * inv_h, d2h11 * inv_h)

    # k=3: EvalDeriv3 — d³P/dx³ = constants · inv_h³
    inv_h3 = inv_h2 * inv_h
    w3 = (12 * inv_h3, -12 * inv_h3, 6 * inv_h2, 6 * inv_h2)

    return w0, w1, w2, w3
end

"""
    _bake_nd_anchors(grids, spacings, queries, extraps) -> Vector{_NDAdjointAnchor}

Precompute cell indices and all derivative-order Hermite weights for each query point.
Per-axis query preprocessing is handled by `_extrap_axis` (periodic wrapping, clamping, etc.).
For OOB queries, weights are zeroed per-axis following the same logic as 1D `_fixup_clampfill_anchors!`.
"""
function _bake_nd_anchors(
        grids::NTuple{N, AbstractVector{Tg}},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N, Tg <: AbstractFloat}
    nq = _query_length(queries)
    _query_validate(queries)

    anchors = Vector{_NDAdjointAnchor{Tg, N}}(undef, nq)
    @inbounds for q in 1:nq
        # Single pass: compute index + all 4 weight sets per axis together
        query_q = _extract_query_point(queries, q, Val(N))
        idx_and_weights = ntuple(Val(N)) do d
            xq_raw = Tg(query_q[d])
            xq_d = _extrap_axis(xq_raw, grids[d], extraps[d])
            idx, _, _ = search_interval(DEFAULT_SEARCHER, grids[d], spacings[d], xq_d)
            h = _get_h(spacings[d], idx)
            inv_h = _get_inv_h(spacings[d], idx)
            t = (xq_d - grids[d][idx]) * inv_h
            is_oob = xq_raw < first(grids[d]) || xq_raw > last(grids[d])
            return (idx, _compute_nd_anchor_weights(t, h, inv_h), is_oob)
        end
        indices = ntuple(d -> idx_and_weights[d][1], Val(N))
        w0 = ntuple(d -> idx_and_weights[d][2][1], Val(N))
        w1 = ntuple(d -> idx_and_weights[d][2][2], Val(N))
        w2 = ntuple(d -> idx_and_weights[d][2][3], Val(N))
        w3 = ntuple(d -> idx_and_weights[d][2][4], Val(N))

        # Per-axis OOB weight fixup (same logic as 1D _fixup_clampfill_anchors!)
        for d in 1:N
            idx_and_weights[d][3] || continue   # skip in-bounds axes
            ext_d = extraps[d]
            if ext_d isa FillExtrap
                zw = (zero(Tg), zero(Tg), zero(Tg), zero(Tg))
                w0 = Base.setindex(w0, zw, d)
                w1 = Base.setindex(w1, zw, d)
                w2 = Base.setindex(w2, zw, d)
                w3 = Base.setindex(w3, zw, d)
            elseif ext_d isa ClampExtrap
                zw = (zero(Tg), zero(Tg), zero(Tg), zero(Tg))
                w1 = Base.setindex(w1, zw, d)
                w2 = Base.setindex(w2, zw, d)
                w3 = Base.setindex(w3, zw, d)
            end
        end

        anchors[q] = _NDAdjointAnchor{Tg, N}(indices, w0, w1, w2, w3)
    end
    return anchors
end

# ========================================
# Eval Adjoint Scatter (@generated, any N)
# ========================================
#
# Adjoint of `_eval_nd_cell` (cubic_nd_eval.jl). For each query, scatters
# y_bar[q] into 2^N partials × 2^N corners = 4^N entries using tensor-product
# Hermite weight products.
#
# Weight indexing per axis d:
#   anchor.w0[d] = (w_fL, w_fR, w_dyL, w_dyR) for EvalValue
#   anchor.w1[d], w2[d], w3[d] for DerivOp(1), (2), (3) respectively
#   index = 1 + corner_d + 2*deriv_d
#   (corner=0,deriv=0) → 1=w_fL, (1,0) → 2=w_fR, (0,1) → 3=w_dyL, (1,1) → 4=w_dyR

# --- Codegen helper: emit 4^N scatter accumulations from weight symbols ---
function _scatter_nd_codegen(N, idx_syms, w_syms)
    stmts = Expr[]
    NP = 1 << N
    NC = 1 << N
    for p in 0:(NP - 1)
        for c in 0:(NC - 1)
            weight_factors = Symbol[]
            for d in 1:N
                corner_d = (c >> (d - 1)) & 1
                deriv_d = (p >> (d - 1)) & 1
                w_idx = 1 + corner_d + 2 * deriv_d
                push!(weight_factors, w_syms[d, w_idx])
            end
            wp_expr = weight_factors[1]
            for i in 2:length(weight_factors)
                wp_expr = :($wp_expr * $(weight_factors[i]))
            end
            offsets = _corner_offset_expr(c, N)
            idx_exprs = [:($(idx_syms[d]) + $(offsets[d])) for d in 1:N]
            p_idx = _partial_index(p)
            lhs = :(partials_bar[$p_idx, $(idx_exprs...)])
            push!(stmts, :($lhs += yb * $wp_expr))
        end
    end
    return stmts
end

"""
    _scatter_nd!(partials_bar, yb, anchor, ops)

@generated tensor-product scatter with per-axis DerivOp dispatch.
Selects `anchor.w0[d]`, `w1[d]`, `w2[d]`, or `w3[d]` per axis at compile time
based on the concrete DerivOp types in `ops`.

When all ops are `EvalValue()`, the generated code reads `anchor.w0[d]` for every
axis — identical to what a dedicated EvalValue-only method would produce.
"""
@inline @generated function _scatter_nd!(
        partials_bar::AbstractArray{Tv, NP1},
        yb::Tv,
        anchor::_NDAdjointAnchor{Tg, N},
        ops::OPS
    ) where {Tv, Tg, N, NP1, OPS <: NTuple{N, AbstractEvalOp}}
    NP1 == N + 1 || error("NP1 must equal N+1, got NP1=$NP1, N=$N")

    stmts = Expr[]

    idx_syms = ntuple(d -> Symbol("idx_", d), N)
    push!(stmts, :($(Expr(:tuple, idx_syms...)) = anchor.indices))

    # Per-axis: select weight field based on DerivOp order from type parameter
    w_field_names = [:w0, :w1, :w2, :w3]
    w_syms = Matrix{Symbol}(undef, N, 4)
    for d in 1:N
        k_d = deriv_order(fieldtype(OPS, d))
        wf = w_field_names[k_d + 1]
        w_syms[d, 1] = Symbol("w_", d, "_fL")
        w_syms[d, 2] = Symbol("w_", d, "_fR")
        w_syms[d, 3] = Symbol("w_", d, "_dyL")
        w_syms[d, 4] = Symbol("w_", d, "_dyR")
        lhs = Expr(:tuple, w_syms[d, 1], w_syms[d, 2], w_syms[d, 3], w_syms[d, 4])
        push!(stmts, :($lhs = anchor.$wf[$d]))
    end

    append!(stmts, _scatter_nd_codegen(N, idx_syms, w_syms))

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
        return nothing
    end
end

# ========================================
# Build Adjoint (Per-Axis, Reverse Order)
# ========================================

"""
    _adjoint_axis_pair!(src_3d, dst_3d, cache_d, spacing_d, bc_pair, grid_d,
                        shape_before, n_d, shape_after, z_bar, f_contrib, dy_bar_slice)

Function barrier for per-axis adjoint processing.

Accepts `cache_d` as a concrete-typed argument, forcing Julia to specialize
on its type. This eliminates the Union that would arise from
`p_src == 1 ? caches[d] : mixed_caches[d]` inside `_build_adjoint_nd!`.

PolyFit stencil coefficients are computed on the fly from `bc_pair` + `grid_d`,
matching the forward `compute_rhs!` pattern (O(D) per axis, negligible).
"""
@inline function _adjoint_axis_pair!(
        src_3d, dst_3d,
        cache_d::CubicSplineCache{Tg},
        spacing_d::AbstractGridSpacing{Tg},
        bc_pair::BCPair,
        grid_d::AbstractVector{Tg},
        shape_before::Int, n_d::Int, shape_after::Int,
        z_bar::AbstractVector{Tv},
        f_contrib::AbstractVector{Tv},
        dy_bar_slice::AbstractVector{Tv}
    ) where {Tv, Tg}
    # Compute polyfit stencil coefficients on the fly (O(D), grid-only)
    pf = _build_polyfit_data(bc_pair, grid_d)

    for j in 1:shape_after
        for i in 1:shape_before
            # Extract 1D dy_bar slice from partials_bar[p_dst]
            @inbounds for k in 1:n_d
                dy_bar_slice[k] = dst_3d[i, k, j]
            end

            # Step a: moments_to_deriv adjoint
            _moments_to_deriv_adjoint_1d!(z_bar, f_contrib, dy_bar_slice, spacing_d)

            # Step b: transpose Thomas solve (A⁻ᵀ z_bar)
            _ldiv_tridiagonal_transpose!(z_bar, cache_d.thomas)

            # Step c: RHS stencil adjoint (f_contrib += Rᵀ z_bar)
            _compute_rhs_adjoint!(f_contrib, z_bar, spacing_d, bc_pair, pf)

            # Step d: accumulate into partials_bar[p_src]
            @inbounds for k in 1:n_d
                src_3d[i, k, j] += f_contrib[k]
            end
        end
    end
    return nothing
end

"""
    _adjoint_axis_pair_periodic!(src_3d, dst_3d, cache_d, spacing_d,
                                  shape_before, n_d, shape_after,
                                  z_bar, f_contrib, dy_bar_slice, q_t)

Function barrier for per-axis **periodic** adjoint processing.

Differences from non-periodic `_adjoint_axis_pair!`:
- Step a: BC correction adjoint: avg(dy_bar[1], dy_bar[end]) (self-adjoint: Bᵀ=B)
- Step c: fold periodic closure adjoint: `z_bar[1] += z_bar[n_d]` (adjoint of `z[n+1]=z[1]`)
- Step d: Sherman-Morrison transpose solve instead of plain Thomas
- Step e: circulant RHS adjoint instead of standard boundary-conditioned RHS

`q_t = A'⁻ᵀu` is precomputed once per axis (not per slice) and passed in.
"""
@inline function _adjoint_axis_pair_periodic!(
        src_3d, dst_3d,
        cache_d::CubicSplineCache{Tg},
        spacing_d::AbstractGridSpacing{Tg},
        shape_before::Int, n_d::Int, shape_after::Int,
        z_bar::AbstractVector{Tv},
        f_contrib::AbstractVector{Tv},
        dy_bar_slice::AbstractVector{Tv},
        q_t::AbstractVector
    ) where {Tv, Tg}
    n = n_d - 1  # n intervals for periodic (n+1 grid points, z[n+1]=z[1])

    for j in 1:shape_after
        for i in 1:shape_before
            # Extract 1D dy_bar slice from partials_bar[p_dst]
            @inbounds for k in 1:n_d
                dy_bar_slice[k] = dst_3d[i, k, j]
            end

            # Step a: BC correction adjoint (self-adjoint: averaging is symmetric)
            # Forward: avg = 0.5*(dy[1]+dy[end]); dy[1]=avg; dy[end]=avg
            # Adjoint: same operation (Bᵀ = B since the matrix is symmetric)
            @inbounds begin
                avg = inv(Tg(2)) * (dy_bar_slice[1] + dy_bar_slice[n_d])
                dy_bar_slice[1] = avg
                dy_bar_slice[n_d] = avg
            end

            # Step b: moments_to_deriv adjoint (SAME as non-periodic)
            _moments_to_deriv_adjoint_1d!(z_bar, f_contrib, dy_bar_slice, spacing_d)

            # Step c: fold periodic closure (adjoint of z[n+1]=z[1])
            @inbounds z_bar[1] += z_bar[n_d]

            # Step d: Sherman-Morrison transpose solve on z_bar[1:n]
            _adjoint_periodic_solve!(z_bar, cache_d, q_t, n)

            # Step e: circulant RHS adjoint → f_contrib[1:n+1]
            _compute_rhs_adjoint_periodic!(f_contrib, z_bar, spacing_d, n)

            # Step f: accumulate into partials_bar[p_src]
            @inbounds for k in 1:n_d
                src_3d[i, k, j] += f_contrib[k]
            end
        end
    end
    return nothing
end

"""
    _build_adjoint_nd!(partials_bar, caches, mixed_caches, spacings,
                       bcs, mixed_bcs, grids, grid_size)

Apply the adjoint of the ND build pipeline for arbitrary N dimensions.
Processes axes in reverse order (d=N..1).

For each axis d, reverses the forward chain:
  partials[p_dst] = moments_to_deriv( A_d⁻¹ · R_d · partials[p_src] )

Cache/BC selection per (d, p_src) pair:
- **Periodic axis**: `caches[d]` for all `p_src` (periodic propagates through `_get_effective_bc`).
  Uses Sherman-Morrison transpose solve with `q_t = A'⁻ᵀu` precomputed once per axis.
- **Non-periodic, `p_src == 1`** (pure derivative): `caches[d]` + `bcs[d]` (user's BC)
- **Non-periodic, `p_src > 1`** (mixed partial): `mixed_caches[d]` + `mixed_bcs[d]` (CubicFit)

Uses function barriers (`_adjoint_axis_pair!` / `_adjoint_axis_pair_periodic!`)
so each branch dispatches on a concrete cache type — no Union boxing.
"""
@with_pool pool function _build_adjoint_nd!(
        partials_bar::AbstractArray{Tv},
        caches,
        mixed_caches,
        spacings,
        bcs,
        mixed_bcs,
        grids::NTuple{N, AbstractVector{Tg}},
        grid_size::NTuple{N, Int}
    ) where {Tv, Tg <: AbstractFloat, N}
    # Process axes in reverse order: d=N, N-1, ..., 1
    for d in N:-1:1
        bit_d = 1 << (d - 1)
        n_d = grid_size[d]
        spacing_d = spacings[d]
        is_periodic_d = _is_periodic_bc(bcs[d])

        # Compute reshape dimensions for axis d
        shape_before = 1
        for k in 1:(d - 1)
            shape_before *= grid_size[k]
        end
        shape_after = 1
        for k in (d + 1):N
            shape_after *= grid_size[k]
        end

        # Per-axis work buffers (reused across all slices)
        z_bar = acquire!(pool, Tv, n_d)
        f_contrib = acquire!(pool, Tv, n_d)
        dy_bar_slice = acquire!(pool, Tv, n_d)

        # Precompute q_t for periodic axis (once, reused across all p_src and slices)
        # q_t = A'^{-T} u where u = [1, 0, ..., 0, 1]
        if is_periodic_d
            n_intervals = n_d - 1
            q_t = acquire!(pool, Tg, n_intervals)
            fill!(q_t, zero(Tg))
            @inbounds q_t[1] = one(Tg)
            @inbounds q_t[n_intervals] = one(Tg)
            _ldiv_tridiagonal_transpose!(q_t, caches[d].thomas)
        end

        for p_src in 1:bit_d
            p_dst = p_src + bit_d

            # N-dim views via selectdim, then reshape to 3D: (shape_before, n_d, shape_after)
            src_3d = reshape(selectdim(partials_bar, 1, p_src), shape_before, n_d, shape_after)
            dst_3d = reshape(selectdim(partials_bar, 1, p_dst), shape_before, n_d, shape_after)

            if is_periodic_d
                # Periodic: same cache for all p_src (periodic propagates through _get_effective_bc)
                _adjoint_axis_pair_periodic!(
                    src_3d, dst_3d, caches[d], spacing_d,
                    shape_before, n_d, shape_after,
                    z_bar, f_contrib, dy_bar_slice, q_t
                )
            elseif p_src == 1
                _adjoint_axis_pair!(
                    src_3d, dst_3d, caches[d], spacing_d,
                    bcs[d], grids[d],
                    shape_before, n_d, shape_after,
                    z_bar, f_contrib, dy_bar_slice
                )
            else
                _adjoint_axis_pair!(
                    src_3d, dst_3d, mixed_caches[d], spacing_d,
                    mixed_bcs[d], grids[d],
                    shape_before, n_d, shape_after,
                    z_bar, f_contrib, dy_bar_slice
                )
            end
        end
    end

    return nothing
end

# NOTE: _has_exclusive_periodic, _adjoint_output_size(adj), _adjoint_nd_finalize,
# _adjoint_apply_exclusive_nd!, and all 6 callable methods (Vector/Real/Tuple ×
# alloc/in-place) are now shared on AbstractAdjointND in nd_adjoint_protocol.jl.
# CubicAdjointND inherits them via _n_queries, _grid_size, _adjoint_bcs,
# and _adjoint_nd_apply! defined in cubic_nd_adjoint_types.jl.

# ========================================
# Core Apply Pipeline
# ========================================

# ── Adjoint scatter dispatch ─────────────────────────────────────────────
# Scalar: direct _scatter_nd! call (no loop, no wrapping)
# Vector/Tuple: loop over elements

@inline function _adjoint_scatter_nd!(partials_bar, anchors, y_bar::Real, ops)
    return @inbounds _scatter_nd!(partials_bar, y_bar, anchors[1], ops)
end

@inline function _adjoint_scatter_nd!(partials_bar, anchors, y_bar, ops)
    return @inbounds for q in eachindex(y_bar)
        _scatter_nd!(partials_bar, y_bar[q], anchors[q], ops)
    end
end

@with_pool pool function _cubic_adjoint_nd_apply!(
        f_bar::AbstractArray{Tv, N},
        adj::CubicAdjointND{Tg, N},
        y_bar,  # Real, Tuple, or AbstractVector — dispatched via _adjoint_scatter_nd!
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N}
    NP = 1 << N
    total = NP * prod(adj.grid_size)

    # Pool-allocate partials_bar as 1D, reshape to (NP, n1, n2, ..., nN)
    pb_flat = acquire!(pool, Tv, total)
    fill!(pb_flat, zero(Tv))
    partials_bar = reshape(pb_flat, NP, adj.grid_size...)

    # Step 0: Eval adjoint scatter — dispatches on y_bar type
    _adjoint_scatter_nd!(partials_bar, adj.anchors, y_bar, ops)

    # Steps 1-3: Build adjoint (reverse axis order) — UNCHANGED by deriv
    _build_adjoint_nd!(
        partials_bar, adj.caches, adj.mixed_caches, adj.spacings,
        adj.bcs, adj.mixed_bcs, adj.grids, adj.grid_size
    )

    # Extract f_bar = partials_bar[1, ...]
    src = selectdim(partials_bar, 1, 1)
    f_bar .+= src

    return f_bar
end

# ========================================
# Constructor
# ========================================

"""
    cubic_adjoint(grids::NTuple{N}, queries::NTuple{N}; bc=CubicFit(), autocache=true)

Construct an N-dimensional cubic adjoint operator.

# Arguments
- `grids`: N-tuple of grid vectors, one per dimension
- `queries`: N-tuple of query coordinate vectors (SoA format)
- `bc`: Boundary condition (single or per-axis tuple)
- `autocache`: Whether to cache Thomas LU factorizations

# Returns
`CubicAdjointND` operator that can be called as `adj(y_bar)` or `adj(f_bar, y_bar)`.

# Example
```julia
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 15)
xq = rand(100)
yq = rand(100)

adj = cubic_adjoint((x, y), (xq, yq); bc=CubicFit())
f_bar = adj(y_bar)   # returns 20×15 matrix
```
"""
function cubic_adjoint(
        grids::NTuple{N, AbstractVector},
        queries::Tuple{AbstractVector, Vararg{AbstractVector}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        autocache::Bool = true,
        _extra...
    ) where {N}
    length(queries) == N || _throw_ndims_mismatch("query vectors", N, length(queries))
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, bcs, Val(N), Tg)
    return _build_nd_adjoint(grids_typed, queries, bcs, extraps, autocache)
end

# Single-tuple query: cubic_adjoint((x, y), (0.5, 0.5); ...)
# Wraps as 1-element tuple-vector for _bake_nd_anchors protocol.
function cubic_adjoint(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        autocache::Bool = true,
        _extra...
    ) where {N}
    return cubic_adjoint(grids, [query]; bc = bc, extrap = extrap, autocache = autocache)
end

# Generic query fallback: passes queries directly to _build_nd_adjoint.
# _bake_nd_anchors uses query protocol internally — no SoA conversion needed.
# Handles AoS (Vector{NTuple}), Vector{SVector}, SoA, or any protocol-implementing type.
function cubic_adjoint(
        grids::NTuple{N, AbstractVector},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        autocache::Bool = true,
        _extra...
    ) where {N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, bcs, Val(N), Tg)
    return _build_nd_adjoint(grids_typed, queries, bcs, extraps, autocache)
end

"""
    _build_nd_adjoint(grids, queries, bcs, autocache)

Internal builder for `CubicAdjointND`. Separated from the public API so that
`Tg` is bound via the argument type (`grids::NTuple{N, AbstractVector{Tg}}`),
making the return type fully inferrable.

Uses `map` instead of `ntuple` for per-axis tuple construction to avoid
Union return types from runtime tuple indexing in closures.

# Periodic Handling
- Exclusive grids are extended to inclusive form (grid-only, no data for adjoint)
- Periodic BCs are stored as-is (`PeriodicBC`), not normalized to `BCPair`
- Periodic caches route to the periodic cache pool (`_get_cubic_cache(x, PeriodicBC())`)
- `_get_effective_bc(PeriodicBC(), p_src, grid) = PeriodicBC()` for all `p_src`
"""
function _build_nd_adjoint(
        grids::NTuple{N, AbstractVector{Tg}},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        autocache::Bool
    ) where {N, Tg <: AbstractFloat}
    # Validate: PolyFit BCs have enough grid points (periodic axes skip this)
    _validate_polyfit_bcs(grids, bcs, Val(N))

    # Extend exclusive periodic grids → inclusive form (grid-only, no data needed for adjoint)
    grids_ext = map(grids, bcs) do grid_d, bc_d
        bc_d isa PeriodicBC{:exclusive} || return grid_d

        period = _resolve_exclusive_period(grid_d, bc_d)
        x_end = first(grid_d) + Tg(period)
        if grid_d isa AbstractRange
            range(first(grid_d); step = step(grid_d), length = length(grid_d) + 1)
        else
            vcat(grid_d, x_end)
        end
    end

    # Per-axis: normalize BC for cache + polyfit construction
    # Periodic axes store PeriodicBC as-is; non-periodic normalize to BCPair
    norm_bcs = map(bcs) do bc_d
        _is_periodic_bc(bc_d) ? bc_d : _normalize_bc(bc_d, Tg)
    end

    caches = map(grids_ext, norm_bcs) do grid_d, bp_d
        if _is_periodic_bc(bp_d)
            _get_cubic_cache(grid_d, PeriodicBC(), autocache)
        else
            _get_cubic_cache(grid_d, bp_d, autocache)
        end
    end

    # Mixed-partial BC pairs (p_src > 1): _get_effective_bc determines the BC.
    # For periodic: returns PeriodicBC (propagates). For non-periodic: typically CubicFit.
    mixed_bcs = map(grids_ext, bcs) do grid_d, bc_d
        mixed_bc = _get_effective_bc(bc_d, 2, grid_d)
        _is_periodic_bc(mixed_bc) ? mixed_bc : _normalize_bc(mixed_bc, Tg)
    end

    mixed_caches = map(grids_ext, mixed_bcs) do grid_d, mbp_d
        if _is_periodic_bc(mbp_d)
            _get_cubic_cache(grid_d, PeriodicBC(), autocache)
        else
            _get_cubic_cache(grid_d, mbp_d, autocache)
        end
    end

    spacings = _create_spacings_typed(grids_ext)

    # Bake per-query anchors (extrap handles periodic wrapping + OOB weight fixup)
    anchors = _bake_nd_anchors(grids_ext, spacings, queries, extraps)

    grid_size = ntuple(d -> length(grids_ext[d]), Val(N))

    return CubicAdjointND{
        Tg, N,
        typeof(grids_ext), typeof(spacings), typeof(caches), typeof(mixed_caches),
        typeof(norm_bcs), typeof(mixed_bcs),
    }(
        grids_ext, spacings, caches, mixed_caches, norm_bcs, mixed_bcs,
        anchors, grid_size
    )
end

# ========================================
# Matrix Materialization
# ========================================

"""
    Matrix(adj::CubicAdjointND{Tg, N}; deriv=EvalValue()) -> Matrix

Materialize the ND adjoint operator as a dense matrix `Wᵀ` of size
`(prod(output_grid_sizes), n_query)`.

Each column `q` of `Wᵀ` is computed by probing with a unit vector `eₑ`:
`Wᵀ[:, q] = vec(adj(eₑ))`, i.e., the grid-space sensitivity when only query
point `q` has unit sensitivity.

This is an O(n_grid × n_query) operation intended for debugging and verification,
not for production use.

# Example
```julia
adj = cubic_adjoint((x, y), (xq, yq); bc=CubicFit())
Wᵀ = Matrix(adj)                          # (n_grid × n_query)
W  = Matrix(adj)'                          # (n_query × n_grid)

@assert Wᵀ * y_bar ≈ vec(adj(y_bar))     # matrix-vector == operator
```
"""
function Base.Matrix(
        adj::CubicAdjointND{Tg, N};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {Tg, N}
    out_size = _adjoint_output_size(adj)
    n_out = prod(out_size)
    n_query = _n_queries(adj)
    W_T = zeros(Tg, n_out, n_query)
    e_q = zeros(Tg, n_query)
    f_bar = zeros(Tg, out_size...)
    @inbounds for q in 1:n_query
        e_q[q] = one(Tg)
        adj(f_bar, e_q; deriv = deriv)
        W_T[:, q] .= vec(f_bar)
        e_q[q] = zero(Tg)
    end
    return W_T
end
