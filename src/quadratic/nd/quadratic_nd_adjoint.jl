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
    ) where {N, Tg <: AbstractFloat}
    nq = _query_length(queries)
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps)

    anchors = Vector{_NDAdjointAnchor{Tg, N}}(undef, nq)
    @inbounds for q in 1:nq
        query_q = _extract_query_point(queries, q, Val(N))
        idx_and_weights = ntuple(Val(N)) do d
            xq_raw = Tg(query_q[d])
            xq_d = _extrap_axis(xq_raw, grids[d], extraps[d])
            idx, _, _ = search_interval(DEFAULT_SEARCHER, grids[d], spacings[d], xq_d)
            h = _get_h(spacings[d], idx)
            inv_h = _get_inv_h(spacings[d], idx)
            t = (xq_d - grids[d][idx]) * inv_h
            is_oob = xq_raw < first(grids[d]) || xq_raw > last(grids[d])
            return (idx, _compute_nd_quadratic_anchor_weights(t, h, inv_h), is_oob)
        end
        indices = ntuple(d -> idx_and_weights[d][1], Val(N))
        w0 = ntuple(d -> idx_and_weights[d][2][1], Val(N))
        w1 = ntuple(d -> idx_and_weights[d][2][2], Val(N))
        w2 = ntuple(d -> idx_and_weights[d][2][3], Val(N))
        w3 = ntuple(d -> idx_and_weights[d][2][4], Val(N))

        # Per-axis OOB weight fixup
        for d in 1:N
            idx_and_weights[d][3] || continue
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
        f_contrib::AbstractVector{Tv}
    ) where {Tv, Tg}
    nm1 = n_d - 1

    for j in 1:shape_after
        for i in 1:shape_before
            # Extract 1D d_bar slice from partials_bar[p_dst]
            @inbounds for k in 1:n_d
                d_bar_work[k] = dst_3d[i, k, j]
            end

            # Zero f_contrib and s_bar
            @inbounds for k in 1:n_d
                f_contrib[k] = zero(Tv)
            end
            @inbounds for k in 1:nm1
                s_bar[k] = zero(Tv)
            end

            # Step 1: Recurrence adjoint: d̄ → s̄ (+ BC endpoint → f̄)
            _recurrence_adjoint!(s_bar, d_bar_work, f_contrib, bc, spacing_d, grid_d)

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
    _build_adjoint_nd_quadratic!(partials_bar, spacings, bcs, grids, grid_size)

Apply the adjoint of the ND quadratic build pipeline.
Processes axes in reverse order (d=N..1).

For each axis d, each partial pair (p_src, p_dst):
1. Get effective BC via `_get_effective_bc_quadratic`
2. Normalize to QuadraticBC for dispatch
3. Call function barrier `_adjoint_axis_pair_quadratic!`

Simpler than cubic: no dual caches, no periodic handling.
"""
@with_pool pool function _build_adjoint_nd_quadratic!(
        partials_bar::AbstractArray{Tv},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        bcs::NTuple{N, AbstractBC},
        grids::NTuple{N, AbstractVector{Tg}},
        grid_size::NTuple{N, Int}
    ) where {Tv, Tg <: AbstractFloat, N}
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
                s_bar, d_bar_work, f_contrib
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
        partials_bar, adj.spacings, adj.bcs, adj.grids, adj.grid_size
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
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, bcs, Val(N), Tg)
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
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, bcs, Val(N), Tg)
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
    ) where {N, Tg <: AbstractFloat}
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

    # copy() for mutation safety; typeof(grids_c) rebinds G after copy (view → Vector).
    grids_c = map(copy, grids)

    return QuadraticAdjointND{
        Tg, N,
        typeof(grids_c), typeof(spacings), typeof(bcs),
    }(grids_c, spacings, bcs, anchors, grid_size)
end
