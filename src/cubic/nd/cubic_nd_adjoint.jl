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
        h_over_3 = h_over_6 * Tg(2)
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
        h_over_3 = h_over_6 * Tg(2)
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
    _bake_nd_anchors(grids, spacings, queries, ::Type{Tg}) -> Vector{_NDAdjointAnchor}

Precompute cell indices and Hermite basis weights for each query point.
"""
function _bake_nd_anchors(
        grids::NTuple{N, AbstractVector{Tg}},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        queries::NTuple{N, AbstractVector}
    ) where {N, Tg <: AbstractFloat}
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(
            DimensionMismatch(
                "Query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
            )
        )
    end

    anchors = Vector{_NDAdjointAnchor{Tg, N}}(undef, n_queries)
    @inbounds for q in 1:n_queries
        indices = ntuple(Val(N)) do d
            xq_d = Tg(queries[d][q])
            x_d = grids[d]
            idx, _, _ = search_interval(DEFAULT_SEARCHER, x_d, spacings[d], xq_d)
            return idx
        end
        weights = ntuple(Val(N)) do d
            xq_d = Tg(queries[d][q])
            x_d = grids[d]
            idx = indices[d]
            xL = x_d[idx]
            h = _get_h(spacings[d], idx)
            inv_h = _get_inv_h(spacings[d], idx)
            dL = xq_d - xL
            t = dL * inv_h
            w_fL = _hermite_h00(t)
            w_fR = _hermite_h01(t)
            w_dyL = _hermite_h10(t) * h
            w_dyR = _hermite_h11(t) * h
            return (w_fL, w_fR, w_dyL, w_dyR)
        end
        anchors[q] = _NDAdjointAnchor{Tg, N}(indices, weights)
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
#   anchor.weights[d] = (w_fL, w_fR, w_dyL, w_dyR)
#   index = 1 + corner_d + 2*deriv_d
#   (corner=0,deriv=0) → 1=w_fL, (1,0) → 2=w_fR, (0,1) → 3=w_dyL, (1,1) → 4=w_dyR

"""
    _scatter_nd!(partials_bar, yb, anchor)

@generated tensor-product scatter for arbitrary N dimensions.
Accumulates `yb * prod(per-axis weights)` into partials_bar at all
(partial, corner) combinations. Produces straight-line code with no loops.

Reuses `_partial_index` and `_corner_offset_expr` from `nd_utils.jl`.
"""
@inline @generated function _scatter_nd!(
        partials_bar::AbstractArray{Tv, NP1},
        yb::Tv,
        anchor::_NDAdjointAnchor{Tg, N}
    ) where {Tv, Tg, N, NP1}
    NP1 == N + 1 || error("NP1 must equal N+1, got NP1=$NP1, N=$N")

    stmts = Expr[]
    NP = 1 << N  # 2^N partials
    NC = 1 << N  # 2^N corners

    # Destructure anchor.indices: (idx_1, idx_2, ..., idx_N)
    idx_syms = ntuple(d -> Symbol("idx_", d), N)
    push!(stmts, :($(Expr(:tuple, idx_syms...)) = anchor.indices))

    # Destructure anchor.weights per axis: (w_d_fL, w_d_fR, w_d_dyL, w_d_dyR)
    w_syms = Matrix{Symbol}(undef, N, 4)
    for d in 1:N
        w_syms[d, 1] = Symbol("w_", d, "_fL")
        w_syms[d, 2] = Symbol("w_", d, "_fR")
        w_syms[d, 3] = Symbol("w_", d, "_dyL")
        w_syms[d, 4] = Symbol("w_", d, "_dyR")
        lhs = Expr(:tuple, w_syms[d, 1], w_syms[d, 2], w_syms[d, 3], w_syms[d, 4])
        push!(stmts, :($lhs = anchor.weights[$d]))
    end

    # Generate one accumulation statement per (partial, corner) pair
    for p in 0:(NP - 1)
        for c in 0:(NC - 1)
            # Build weight product: prod_d w_d[1 + corner_d + 2*deriv_d]
            weight_factors = Symbol[]
            for d in 1:N
                corner_d = (c >> (d - 1)) & 1
                deriv_d = (p >> (d - 1)) & 1
                w_idx = 1 + corner_d + 2 * deriv_d
                push!(weight_factors, w_syms[d, w_idx])
            end

            # Chain multiply: w1 * w2 * ... * wN
            wp_expr = weight_factors[1]
            for i in 2:length(weight_factors)
                wp_expr = :($wp_expr * $(weight_factors[i]))
            end

            # Index expression: partials_bar[p+1, idx_1+off_1, ..., idx_N+off_N]
            offsets = _corner_offset_expr(c, N)
            idx_exprs = [:($(idx_syms[d]) + $(offsets[d])) for d in 1:N]
            p_idx = _partial_index(p)

            lhs = :(partials_bar[$p_idx, $(idx_exprs...)])
            push!(stmts, :($lhs += yb * $wp_expr))
        end
    end

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
    _adjoint_axis_pair!(src_3d, dst_3d, cache_d, spacing_d, bc_pair, pf,
                        shape_before, n_d, shape_after, z_bar, f_contrib, dy_bar_slice)

Function barrier for per-axis adjoint processing.

Accepts `cache_d` as a concrete-typed argument, forcing Julia to specialize
on its type. This eliminates the Union that would arise from
`p_src == 1 ? caches[d] : eff_caches[d]` when `C ≠ CE`.
"""
@inline function _adjoint_axis_pair!(
        src_3d, dst_3d,
        cache_d::CubicSplineCache{Tg},
        spacing_d::AbstractGridSpacing{Tg},
        bc_pair::BCPair,
        pf::_AdjointPolyfitData,
        shape_before::Int, n_d::Int, shape_after::Int,
        z_bar::AbstractVector{Tv},
        f_contrib::AbstractVector{Tv},
        dy_bar_slice::AbstractVector{Tv}
    ) where {Tv, Tg}
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
    _build_adjoint_nd!(partials_bar, caches, eff_caches, spacings,
                       bc_pairs, eff_bc_pairs, pf_user, pf_eff, grid_size)

Apply the adjoint of the ND build pipeline for arbitrary N dimensions.
Processes axes in reverse order (d=N..1).

For each axis d, reverses the forward chain:
  partials[p_dst] = moments_to_deriv( A_d⁻¹ · R_d · partials[p_src] )

All inputs are precomputed at construction time (no runtime BC resolution):
- `bc_pairs[d]` / `eff_bc_pairs[d]`: concrete `BCPair{L,R}` (from `_normalize_bc`)
- `pf_user[d]` / `pf_eff[d]`: polyfit stencil coefficients (from `_build_polyfit_data`)
- `caches[d]` / `eff_caches[d]`: Thomas LU factorizations

Uses `if p_src == 1` branching with a function barrier (`_adjoint_axis_pair!`)
to ensure cache dispatch is type-stable.
"""
@with_pool pool function _build_adjoint_nd!(
        partials_bar::AbstractArray{Tv},
        caches::NTuple{N, CubicSplineCache{Tg}},
        eff_caches::NTuple{N, CubicSplineCache{Tg}},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        bc_pairs::NTuple{N, BCPair},
        eff_bc_pairs::NTuple{N, BCPair},
        pf_user::NTuple{N, _AdjointPolyfitData},
        pf_eff::NTuple{N, _AdjointPolyfitData},
        grid_size::NTuple{N, Int}
    ) where {Tv, Tg <: AbstractFloat, N}
    # Process axes in reverse order: d=N, N-1, ..., 1
    for d in N:-1:1
        bit_d = 1 << (d - 1)
        n_d = grid_size[d]
        spacing_d = spacings[d]

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

        for p_src in 1:bit_d
            p_dst = p_src + bit_d

            # N-dim views via selectdim, then reshape to 3D: (shape_before, n_d, shape_after)
            src_3d = reshape(selectdim(partials_bar, 1, p_src), shape_before, n_d, shape_after)
            dst_3d = reshape(selectdim(partials_bar, 1, p_dst), shape_before, n_d, shape_after)

            # Branch on p_src to select concrete (cache, bc_pair, pf) triple
            if p_src == 1
                _adjoint_axis_pair!(
                    src_3d, dst_3d, caches[d], spacing_d,
                    bc_pairs[d], pf_user[d],
                    shape_before, n_d, shape_after,
                    z_bar, f_contrib, dy_bar_slice
                )
            else
                _adjoint_axis_pair!(
                    src_3d, dst_3d, eff_caches[d], spacing_d,
                    eff_bc_pairs[d], pf_eff[d],
                    shape_before, n_d, shape_after,
                    z_bar, f_contrib, dy_bar_slice
                )
            end
        end
    end

    return nothing
end

# ========================================
# Apply Methods
# ========================================

"""
    (adj::CubicAdjointND)(y_bar) -> f_bar::Array{Tv, N}

Allocating adjoint apply: compute `f̄ = Wᵀȳ`.
"""
function (adj::CubicAdjointND{Tg, N})(y_bar::AbstractVector) where {Tg, N}
    n_query = length(adj.anchors)
    length(y_bar) == n_query || throw(
        DimensionMismatch("y_bar length $(length(y_bar)) must match query count $n_query")
    )
    Tv = promote_type(eltype(y_bar), Tg)
    f_bar = zeros(Tv, adj.grid_size...)
    _cubic_adjoint_nd_apply!(f_bar, adj, y_bar)
    return f_bar
end

"""
    (adj::CubicAdjointND)(f_bar, y_bar) -> f_bar

In-place adjoint apply: compute `f̄ = Wᵀȳ` into pre-allocated `f_bar`.
Zeros `f_bar` before accumulation.
"""
function (adj::CubicAdjointND{Tg, N})(
        f_bar::AbstractArray{Tv, N}, y_bar::AbstractVector
    ) where {Tg, Tv, N}
    size(f_bar) == adj.grid_size || throw(
        DimensionMismatch("f_bar size $(size(f_bar)) must match grid size $(adj.grid_size)")
    )
    n_query = length(adj.anchors)
    length(y_bar) == n_query || throw(
        DimensionMismatch("y_bar length $(length(y_bar)) must match query count $n_query")
    )
    fill!(f_bar, zero(Tv))
    _cubic_adjoint_nd_apply!(f_bar, adj, y_bar)
    return f_bar
end

# ========================================
# Core Apply Pipeline
# ========================================

@with_pool pool function _cubic_adjoint_nd_apply!(
        f_bar::AbstractArray{Tv, N},
        adj::CubicAdjointND{Tg, N},
        y_bar::AbstractVector
    ) where {Tv, Tg, N}
    NP = 1 << N
    total = NP * prod(adj.grid_size)

    # Pool-allocate partials_bar as 1D, reshape to (NP, n1, n2, ..., nN)
    pb_flat = acquire!(pool, Tv, total)
    fill!(pb_flat, zero(Tv))
    partials_bar = reshape(pb_flat, NP, adj.grid_size...)

    # Step 0: Eval adjoint scatter (@generated tensor-product)
    @inbounds for q in eachindex(y_bar)
        _scatter_nd!(partials_bar, y_bar[q], adj.anchors[q])
    end

    # Steps 1-3: Build adjoint (reverse axis order)
    _build_adjoint_nd!(
        partials_bar, adj.caches, adj.eff_caches, adj.spacings,
        adj.bc_pairs, adj.eff_bc_pairs, adj.pf_user, adj.pf_eff, adj.grid_size
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
        queries::NTuple{N, AbstractVector};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        autocache::Bool = true
    ) where {N}
    # Type promotion
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)

    # Resolve per-axis BCs
    bcs = _resolve_bcs_nd(bc, Val(N))

    return _build_nd_adjoint(grids_typed, queries, bcs, autocache)
end

"""
    _build_nd_adjoint(grids, queries, bcs, autocache)

Internal builder for `CubicAdjointND`. Separated from the public API so that
`Tg` is bound via the argument type (`grids::NTuple{N, AbstractVector{Tg}}`),
making the return type fully inferrable.

Uses `map` instead of `ntuple` for per-axis tuple construction to avoid
Union return types from runtime tuple indexing in closures.
"""
function _build_nd_adjoint(
        grids::NTuple{N, AbstractVector{Tg}},
        queries::NTuple{N, AbstractVector},
        bcs::NTuple{N, AbstractBC},
        autocache::Bool
    ) where {N, Tg <: AbstractFloat}
    # Validate: PeriodicBC not yet supported (Phase 3)
    for d in 1:N
        _is_periodic_bc(bcs[d]) && throw(
            ArgumentError("PeriodicBC on axis $d is not yet supported by cubic_adjoint (planned for Phase 3)")
        )
    end

    # Validate: PolyFit BCs have enough grid points
    _validate_polyfit_bcs(grids, bcs, Val(N))

    # Per-axis: normalize BC for cache + polyfit construction
    # map dispatches per-element → each call gets concrete types (no Union)
    bc_pairs = map(bcs) do bc_d
        _normalize_bc(bc_d, Tg)
    end

    caches = map(grids, bc_pairs) do grid_d, bp_d
        _get_cubic_cache(grid_d, bp_d, autocache)
    end

    # Effective BC pairs for mixed partials (p_src > 1).
    # _get_effective_bc with p_src=2 returns the BC used for all mixed partials.
    eff_bc_pairs = map(grids, bcs) do grid_d, bc_d
        eff_bc = _get_effective_bc(bc_d, 2, grid_d)
        _normalize_bc(eff_bc, Tg)
    end

    eff_caches = map(grids, eff_bc_pairs) do grid_d, eff_bp_d
        _get_cubic_cache(grid_d, eff_bp_d, autocache)
    end

    # Precompute polyfit stencil coefficients (grid-only, computed once at construction)
    pf_user = map(bc_pairs, grids) do bp_d, grid_d
        _build_polyfit_data(bp_d, grid_d)
    end
    pf_eff = map(eff_bc_pairs, grids) do eff_bp_d, grid_d
        _build_polyfit_data(eff_bp_d, grid_d)
    end

    spacings = _create_spacings_typed(grids)

    # Bake per-query anchors
    anchors = _bake_nd_anchors(grids, spacings, queries)

    grid_size = ntuple(d -> length(grids[d]), Val(N))

    return CubicAdjointND{
        Tg, N,
        typeof(grids), typeof(spacings), typeof(caches), typeof(eff_caches),
        typeof(bc_pairs), typeof(eff_bc_pairs), typeof(pf_user), typeof(pf_eff),
    }(grids, spacings, caches, eff_caches, bc_pairs, eff_bc_pairs,
      pf_user, pf_eff, anchors, grid_size)
end
