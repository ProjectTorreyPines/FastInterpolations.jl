# ========================================
# QuadraticAdjointND: Scatter, Build Adjoint, Apply, Constructor
# ========================================
#
# Reuses from cubic ND adjoint:
#   - _NDAdjointAnchor{Tg, N}   (same 4-tuple weight anchor type)
#   - _scatter_nd!               (@generated scatter kernel)
#   - _adjoint_scatter_nd!       (scalar/vector dispatch)
#   - _scatter_nd_codegen        (codegen helper)
#
# New for quadratic:
#   - _bake_nd_quadratic_anchors (quadratic weight computation)
#   - _adjoint_axis_pair_quadratic! (recurrence + secant + BC adjoint)
#   - _build_adjoint_nd_quadratic!  (per-axis iteration, no dual caches)
#   - _quadratic_adjoint_nd_apply!  (apply pipeline)
#   - quadratic_adjoint(grids, queries; ...) constructors

# ========================================
# BC Normalization for ND Adjoint
# ========================================
#
# Converts raw AbstractBC (ZeroCurvBC, ZeroSlopeBC, bare PolyFit) to
# QuadraticBC types that _recurrence_adjoint! can dispatch on.
# The BC value is irrelevant for adjoint (only type matters).

@inline _to_quadratic_bc_adjoint(bc::Left, ::Type{Tg}) where {Tg} = bc
@inline _to_quadratic_bc_adjoint(bc::Right, ::Type{Tg}) where {Tg} = bc
@inline _to_quadratic_bc_adjoint(::MinCurvFit, ::Type{Tg}) where {Tg} = MinCurvFit()
@inline _to_quadratic_bc_adjoint(::ZeroCurvBC, ::Type{Tg}) where {Tg} = Right(Deriv2(zero(Tg)))
@inline _to_quadratic_bc_adjoint(::ZeroSlopeBC, ::Type{Tg}) where {Tg} = Left(Deriv1(zero(Tg)))
@inline _to_quadratic_bc_adjoint(bc::PolyFit, ::Type{Tg}) where {Tg} = Right(bc)

# ── Per-axis MinCurvFit constant (dispatch-based, used by map over bcs tuple) ──
# Using dispatch instead of runtime `isa` avoids Union-type issues in tuple closures.
@inline _mincurv_C_for_bc(::MinCurvFit, spacing::AbstractGridSpacing{Tg}, n::Int) where {Tg} = _compute_mincurv_C(spacing, n)
@inline _mincurv_C_for_bc(::AbstractBC, spacing::AbstractGridSpacing{Tg}, ::Int) where {Tg} = zero(Tg)

# ========================================
# Anchor Baking (ND)
# ========================================

"""
    _bake_nd_quadratic_anchors(grids, spacings, queries, extraps)

Precompute per-query cell indices and all 4 derivative-order weight sets per axis.
Uses 4-tuple weights (w_fL, w_fR, w_d, 0) to reuse cubic's `_NDAdjointAnchor` type.

OOB handling baked into weights at construction time.
"""
function _bake_nd_quadratic_anchors(
        grids::NTuple{N, AbstractVector{Tg}},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N, Tg}
    return _bake_nd_anchors_generic(
        grids, spacings, queries, extraps,
        (d, t, h, inv_h, dL) -> _compute_nd_quadratic_anchor_weights(t, h, inv_h)
    )
end

# ========================================
# Build Adjoint — Per-Axis Processing
# ========================================

"""
    _adjoint_axis_pair_quadratic!(src_3d, dst_3d, spacing_d, bc, grid_d,
                                   shape_before, n_d, shape_after,
                                   s_bar, d_bar_work, f_contrib)

Function barrier for per-axis quadratic adjoint processing.
Accepts concrete BC type to force specialization (avoids Union boxing).

Pipeline per 1D slice:
  1. Extract d_bar slice from dst_3d
  2. Recurrence adjoint: d̄ → s̄ (+ BC endpoint → f̄)
  3. Secant adjoint: s̄ → f̄
  4. Accumulate into src_3d
"""
@inline function _adjoint_axis_pair_quadratic!(
        src_3d, dst_3d,
        spacing_d::AbstractGridSpacing{Tg},
        bc::QuadraticBC,
        grid_d::AbstractVector{Tg},
        shape_before::Int, n_d::Int, shape_after::Int,
        s_bar::AbstractVector{Tv},
        d_bar_work::AbstractVector{Tv},
        f_contrib::AbstractVector{Tv},
        C_d::Tg
    ) where {Tv, Tg}
    for j in 1:shape_after
        for i in 1:shape_before
            # Extract 1D d_bar slice from partials_bar[p_dst]
            @inbounds for k in 1:n_d
                d_bar_work[k] = dst_3d[i, k, j]
            end

            # Zero f_contrib and s_bar
            fill!(f_contrib, zero(Tv))
            fill!(s_bar, zero(Tv))

            # Step 1: Recurrence adjoint: d̄ → s̄ (+ BC endpoint → f̄)
            _call_recurrence_adjoint!(s_bar, d_bar_work, f_contrib, bc, spacing_d, grid_d, C_d)

            # Step 2: Secant adjoint: s̄ → f̄
            _secant_adjoint!(f_contrib, s_bar, spacing_d)

            # Step 3: Accumulate into src_3d
            @inbounds for k in 1:n_d
                src_3d[i, k, j] += f_contrib[k]
            end
        end
    end
    return nothing
end

# ========================================
# Build Adjoint — ND Reverse-Axis Iteration
# ========================================

"""
    _build_adjoint_nd_quadratic!(partials_bar, spacings, bcs, grids, grid_size, mincurv_Cs)

Apply the adjoint of the ND quadratic build pipeline.
Processes axes in reverse order (d=N..1).

For each axis d, each partial pair (p_src, p_dst):
1. Get effective BC via `_get_effective_bc_quadratic`
2. Normalize to QuadraticBC for dispatch
3. Call function barrier `_adjoint_axis_pair_quadratic!`

`mincurv_Cs[d]` is the precomputed `inv(Σ inv_h)` per axis, passed through
to avoid O(n) recomputation per slice for MinCurvFit BCs.

Simpler than cubic: no dual caches, no periodic handling.
"""
@with_pool pool function _build_adjoint_nd_quadratic!(
        partials_bar::AbstractArray{Tv},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        bcs::NTuple{N, AbstractBC},
        grids::NTuple{N, AbstractVector{Tg}},
        grid_size::NTuple{N, Int},
        mincurv_Cs::NTuple{N, Tg}
    ) where {Tv, Tg, N}
    for d in N:-1:1
        bit_d = 1 << (d - 1)
        n_d = grid_size[d]
        spacing_d = spacings[d]

        shape_before = 1
        for k in 1:(d - 1)
            shape_before *= grid_size[k]
        end
        shape_after = 1
        for k in (d + 1):N
            shape_after *= grid_size[k]
        end

        # Per-axis work buffers (reused across all slices)
        s_bar = acquire!(pool, Tv, n_d - 1)
        d_bar_work = acquire!(pool, Tv, n_d)
        f_contrib = acquire!(pool, Tv, n_d)

        for p_src in 1:bit_d
            p_dst = p_src + bit_d

            src_3d = reshape(selectdim(partials_bar, 1, p_src), shape_before, n_d, shape_after)
            dst_3d = reshape(selectdim(partials_bar, 1, p_dst), shape_before, n_d, shape_after)

            # Get effective BC for this partial pair and normalize to QuadraticBC
            eff_bc = _get_effective_bc_quadratic(bcs[d], p_src, grids[d])
            bc_q = _to_quadratic_bc_adjoint(eff_bc, Tg)

            # Function barrier: concrete BC type forces specialization
            _adjoint_axis_pair_quadratic!(
                src_3d, dst_3d, spacing_d, bc_q, grids[d],
                shape_before, n_d, shape_after,
                s_bar, d_bar_work, f_contrib, mincurv_Cs[d]
            )
        end
    end

    return nothing
end

# ========================================
# Core Apply Pipeline
# ========================================

@with_pool pool function _quadratic_adjoint_nd_apply!(
        f_bar::AbstractArray{Tv, N},
        adj::QuadraticAdjointND{Tg, N},
        y_bar,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N}
    NP = 1 << N

    # Pool-allocate partials_bar (N+1 dimensional)
    partials_bar = zeros!(pool, Tv, NP, adj.grid_size...)

    # Step 0: Eval adjoint scatter (reuses cubic's _scatter_nd!)
    _adjoint_scatter_nd!(partials_bar, adj.anchors, y_bar, ops)

    # Step 1: Build adjoint (reverse axis order)
    _build_adjoint_nd_quadratic!(
        partials_bar, adj.spacings, adj.bcs, adj.grids, adj.grid_size, adj.mincurv_Cs
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
    quadratic_adjoint(grids::NTuple{N}, queries; bc=Left(QuadraticFit()), extrap=NoExtrap())

Construct an N-dimensional quadratic adjoint operator.

# Arguments
- `grids`: N-tuple of grid vectors, one per dimension
- `queries`: Query points (SoA tuple, AoS vector, or single point)
- `bc`: Boundary condition (single or per-axis tuple)
- `extrap`: Extrapolation mode (single or per-axis tuple)

# Returns
`QuadraticAdjointND` operator that can be called as `adj(y_bar)` or `adj(f_bar, y_bar)`.

# Example
```julia
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 15)
xq = rand(100)
yq = rand(100)

adj = quadratic_adjoint((x, y), (xq, yq); bc=Left(QuadraticFit()))
f_bar = adj(y_bar)   # returns 20×15 matrix
```
"""
function quadratic_adjoint(
        grids::NTuple{N, AbstractVector},
        queries::Tuple{AbstractVector, Vararg{AbstractVector}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = Left(QuadraticFit()),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(queries) == N || _throw_ndims_mismatch("query vectors", N, length(queries))
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    bcs = _resolve_bcs_nd(bc, Val(N))
    # 5-arg `_resolve_extrap` (with BCs): bc-aware per-axis materialize.
    extraps = _resolve_extrap(extrap, bcs, grids_typed, Val(N), Tg)
    return _build_nd_quadratic_adjoint(grids_typed, queries, bcs, extraps)
end

# Single-tuple query: quadratic_adjoint((x, y), (0.5, 0.5); ...)
function quadratic_adjoint(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = Left(QuadraticFit()),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    return quadratic_adjoint(grids, (query,); bc = bc, extrap = extrap)
end

# Single-vector query: quadratic_adjoint((x, y), SVector(0.5, 0.5); ...)
function quadratic_adjoint(
        grids::NTuple{N, AbstractVector},
        query::AbstractVector{<:Real};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = Left(QuadraticFit()),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(query) == N || _throw_ndims_mismatch("query elements", N, length(query))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return quadratic_adjoint(grids, (query_tuple,); bc = bc, extrap = extrap)
end

# Generic query fallback (AoS, Vector{NTuple}, etc.)
function quadratic_adjoint(
        grids::NTuple{N, AbstractVector},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = Left(QuadraticFit()),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    _query_check_ndims(queries, Val(N))
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    bcs = _resolve_bcs_nd(bc, Val(N))
    # 5-arg `_resolve_extrap` (with BCs): bc-aware per-axis materialize.
    extraps = _resolve_extrap(extrap, bcs, grids_typed, Val(N), Tg)
    return _build_nd_quadratic_adjoint(grids_typed, queries, bcs, extraps)
end

# ========================================
# Internal Builder
# ========================================

"""
    _build_nd_quadratic_adjoint(grids, queries, bcs, extraps)

Internal builder. Separated from public API so that `Tg` is bound via argument type,
making the return type fully inferrable.
"""
function _build_nd_quadratic_adjoint(
        grids::NTuple{N, AbstractVector{Tg}},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N, Tg}
    # Validate grid sizes
    for d in 1:N
        length(grids[d]) >= 2 || _throw_adjoint_grid_too_small(d, length(grids[d]))
    end

    # Validate PolyFit point requirements per axis
    for d in 1:N
        validate_polyfit_points(bcs[d], length(grids[d]))
    end

    spacings = _create_spacings_typed(grids)
    anchors = _bake_nd_quadratic_anchors(grids, spacings, queries, extraps)
    grid_size = ntuple(d -> length(grids[d]), Val(N))

    # Precompute per-axis MinCurvFit constants (dispatch-based, no runtime isa check)
    mincurv_Cs = map(_mincurv_C_for_bc, bcs, spacings, grid_size)

    return QuadraticAdjointND(grids, spacings, bcs, anchors, grid_size, mincurv_Cs)
end
