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
# Eval Adjoint Scatter (N=2)
# ========================================

"""
    _scatter_nd_2d!(partials_bar, y_bar, anchors)

Scatter `y_bar` into `partials_bar` using tensor-product Hermite weights.
This is the adjoint of the forward tensor-product Hermite evaluation.

For each query, scatters to 4 partials × 4 corners = 16 entries.
Partial index encoding: p = 1 + dx + 2*dy where dx,dy ∈ {0,1}.
"""
function _scatter_nd_2d!(
        partials_bar::AbstractArray{Tv, 3},
        y_bar::AbstractVector,
        anchors::Vector{_NDAdjointAnchor{Tg, 2}}
    ) where {Tv, Tg}
    @inbounds for q in eachindex(y_bar)
        anchor = anchors[q]
        yb = y_bar[q]
        ix, iy = anchor.indices
        wx_fL, wx_fR, wx_dyL, wx_dyR = anchor.weights[1]
        wy_fL, wy_fR, wy_dyL, wy_dyR = anchor.weights[2]

        # p=1 (f): dx=0, dy=0
        partials_bar[1, ix, iy] += yb * wx_fL * wy_fL
        partials_bar[1, ix + 1, iy] += yb * wx_fR * wy_fL
        partials_bar[1, ix, iy + 1] += yb * wx_fL * wy_fR
        partials_bar[1, ix + 1, iy + 1] += yb * wx_fR * wy_fR

        # p=2 (df/dx): dx=1, dy=0
        partials_bar[2, ix, iy] += yb * wx_dyL * wy_fL
        partials_bar[2, ix + 1, iy] += yb * wx_dyR * wy_fL
        partials_bar[2, ix, iy + 1] += yb * wx_dyL * wy_fR
        partials_bar[2, ix + 1, iy + 1] += yb * wx_dyR * wy_fR

        # p=3 (df/dy): dx=0, dy=1
        partials_bar[3, ix, iy] += yb * wx_fL * wy_dyL
        partials_bar[3, ix + 1, iy] += yb * wx_fR * wy_dyL
        partials_bar[3, ix, iy + 1] += yb * wx_fL * wy_dyR
        partials_bar[3, ix + 1, iy + 1] += yb * wx_fR * wy_dyR

        # p=4 (d2f/dxdy): dx=1, dy=1
        partials_bar[4, ix, iy] += yb * wx_dyL * wy_dyL
        partials_bar[4, ix + 1, iy] += yb * wx_dyR * wy_dyL
        partials_bar[4, ix, iy + 1] += yb * wx_dyL * wy_dyR
        partials_bar[4, ix + 1, iy + 1] += yb * wx_dyR * wy_dyR
    end
    return nothing
end

# ========================================
# Build Adjoint (Per-Axis, Reverse Order)
# ========================================

"""
    _build_adjoint_nd_2d!(partials_bar, caches, spacings, bcs, grids, grid_size)

Apply the adjoint of the ND build pipeline (per-axis, reverse order d=2..1).

For each axis d, reverses the chain:
  partials[p_dst] = moments_to_deriv( A_d⁻¹ · R_d · partials[p_src] )

Adjoint:
  a. moments_to_deriv_adjoint(partials_bar[p_dst]) → z_bar, f_contrib
  b. A_d⁻ᵀ · z_bar (transpose Thomas solve)
  c. f_contrib += Rᵀ · z_bar (RHS stencil adjoint)
  d. partials_bar[p_src] += f_contrib
"""
@with_pool pool function _build_adjoint_nd_2d!(
        partials_bar::AbstractArray{Tv, 3},
        caches::NTuple{2, CubicSplineCache{Tg}},
        spacings::NTuple{2, AbstractGridSpacing{Tg}},
        bcs::NTuple{2, AbstractBC},
        grids::NTuple{2, AbstractVector{Tg}},
        grid_size::NTuple{2, Int}
    ) where {Tv, Tg}
    nx, ny = grid_size

    # Process axes in reverse order: d=2 (y-axis), then d=1 (x-axis)
    for d in 2:-1:1
        bit_d = 1 << (d - 1)
        n_d = grid_size[d]
        spacing_d = spacings[d]
        cache_d = caches[d]

        # Reshape dimensions for axis d
        shape_before = d == 1 ? 1 : nx
        shape_after = d == 2 ? 1 : ny

        # Per-axis work buffers (reused across all slices)
        z_bar = acquire!(pool, Tv, n_d)
        f_contrib = acquire!(pool, Tv, n_d)
        dy_bar_slice = acquire!(pool, Tv, n_d)

        for p_src in 1:bit_d
            p_dst = p_src + bit_d

            # Get effective BC for this (d, p_src) pair
            effective_bc = _get_effective_bc(bcs[d], p_src, grids[d])
            effective_bc_pair = _normalize_bc(effective_bc, Tg)
            effective_pf = _build_polyfit_data(effective_bc_pair, grids[d])

            # 3D views: (shape_before, n_d, shape_after)
            src_3d = reshape(view(partials_bar, p_src, :, :), shape_before, n_d, shape_after)
            dst_3d = reshape(view(partials_bar, p_dst, :, :), shape_before, n_d, shape_after)

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
                    _compute_rhs_adjoint!(f_contrib, z_bar, spacing_d, effective_bc_pair, effective_pf)

                    # Step d: accumulate into partials_bar[p_src]
                    @inbounds for k in 1:n_d
                        src_3d[i, k, j] += f_contrib[k]
                    end
                end
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
function (adj::CubicAdjointND{Tg, 2})(y_bar::AbstractVector) where {Tg}
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
function (adj::CubicAdjointND{Tg, 2})(
        f_bar::AbstractArray{Tv, 2}, y_bar::AbstractVector
    ) where {Tg, Tv}
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
        f_bar::AbstractArray{Tv, 2},
        adj::CubicAdjointND{Tg, 2},
        y_bar::AbstractVector
    ) where {Tv, Tg}
    nx, ny = adj.grid_size
    NP = 4  # 2^N for N=2

    # Pool-allocate partials_bar as 1D, reshape to 3D
    pb_flat = acquire!(pool, Tv, NP * nx * ny)
    fill!(pb_flat, zero(Tv))
    partials_bar = reshape(pb_flat, NP, nx, ny)

    # Step 0: Eval adjoint scatter
    _scatter_nd_2d!(partials_bar, y_bar, adj.anchors)

    # Steps 1-3: Build adjoint (reverse axis order)
    _build_adjoint_nd_2d!(
        partials_bar, adj.caches, adj.spacings,
        adj.bcs, adj.grids, adj.grid_size
    )

    # Extract f_bar = partials_bar[1, :, :]
    @inbounds for j in 1:ny, i in 1:nx
        f_bar[i, j] += partials_bar[1, i, j]
    end

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
    # Per-axis: normalize BC temporarily for cache construction
    # map dispatches per-element → each call gets concrete types (no Union)
    bc_pairs = map(bcs) do bc_d
        _normalize_bc(bc_d, Tg)
    end

    caches = map(grids, bc_pairs) do grid_d, bp_d
        _get_cubic_cache(grid_d, bp_d, autocache)
    end

    spacings = _create_spacings_typed(grids)

    # Bake per-query anchors
    anchors = _bake_nd_anchors(grids, spacings, queries)

    grid_size = ntuple(d -> length(grids[d]), Val(N))

    return CubicAdjointND{
        Tg, N,
        typeof(grids), typeof(spacings), typeof(caches), typeof(bcs),
    }(grids, spacings, caches, bcs, anchors, grid_size)
end
