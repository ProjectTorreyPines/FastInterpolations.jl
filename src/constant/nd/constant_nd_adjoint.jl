# ========================================
# ND Constant Adjoint: Implementation
# ========================================
#
# Computes f̄ = Wᵀ · ȳ for N-dimensional constant interpolation.
#
# Pipeline (adj(y_bar) → f_bar):
#   Step 0: Check for any derivative → early zero return
#   Step 1: Scatter — y_bar → f_bar (1 point per query)
#
# Simpler than linear (2^N corners) or cubic (4^N corners + solve).

# ========================================
# Anchor Baking
# ========================================

"""
    _bake_constant_nd_anchors(grids, queries, extraps)

Precompute per-axis `_ConstantAnchoredQuery` for each query point.

Reads `_get_h(grids[d], idx)` directly from the wrapped axes. Per-axis
processing:
1. Extrapolate position (wrap/clamp/extend/validate)
2. Search interval → get idx, h, dL
3. Detect OOB for FillExtrap skip

PeriodicBC seam handling is transparent: when `grids[d]` is
`_ExclusivePeriodicAxis`, search returns `idx_R = 1` for the seam cell and
`_get_h` returns the correct seam-cell width — no method-specific branching.

For constant interpolation with ExtendExtrap, the forward ND path passes
through OOB queries to the kernel (unlike 1D which clamps). The adjoint
follows the same convention for ND consistency.
"""
function _bake_constant_nd_anchors(
        grids::NTuple{N, AbstractVector{Tg}},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N, Tg}
    nq = _query_length(queries)
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps)

    # Tq widens to query precision so narrower grids never truncate.
    Tq = promote_type(Tg, _query_eltype(queries))
    anchors = Vector{NTuple{N, _ConstantAnchoredQuery{Tg, Tq}}}(undef, nq)
    @inbounds for q in 1:nq
        query_q = _extract_query_point(queries, q, Val(N))
        per_axis = ntuple(Val(N)) do d
            xq_raw = Tq(query_q[d])
            xq_d = _extrap_axis(xq_raw, grids[d], extraps[d])
            idx, idxR, xL, _ = search_interval(DEFAULT_SEARCHER, grids[d], xq_d)
            h = _get_h(grids[d], idx)
            dL = xq_d - xL

            # Determine OOB side flag
            is_oob = xq_raw < first(grids[d]) || xq_raw > last(grids[d])
            state_flag = if is_oob
                xq_raw < first(grids[d]) ? OOB_LEFT : OOB_RIGHT
            else
                IN_DOMAIN
            end

            return _ConstantAnchoredQuery{Tg, Tq}(_IdxPair(idx, idxR), xq_d, state_flag, h, dL)
        end
        anchors[q] = per_axis
    end
    return anchors
end

# ========================================
# Target Index Computation
# ========================================

"""
    _constant_nd_target(per_axis, sides)

Compute the N-dimensional target index for a single query's per-axis anchors.
Uses `_compute_single_offset` per axis — matches the ND forward kernel exactly
(no right-boundary special case, unlike the 1D path).
"""
@inline function _constant_nd_target(
        per_axis::NTuple{N, _ConstantAnchoredQuery{Tg, Tq}},
        sides::Tuple{Vararg{AbstractSide, N}}
    ) where {N, Tg, Tq}
    return ntuple(Val(N)) do d
        aq = per_axis[d]
        aq.idxL + _compute_single_offset(sides[d], aq.h, aq.dL)
    end
end

# ========================================
# FillExtrap OOB Detection
# ========================================

"""
    _any_fill_oob(per_axis, extraps)

Check if any axis has FillExtrap OOB. If so, the entire query should be
skipped (fill value is independent of f → zero gradient).
"""
@inline function _any_fill_oob(
        per_axis::NTuple{N, _ConstantAnchoredQuery},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    for d in 1:N
        @inbounds if extraps[d] isa FillExtrap && per_axis[d].state != IN_DOMAIN
            return true
        end
    end
    return false
end

# ========================================
# Core Apply Pipeline
# ========================================

# Scalar y_bar: single query, direct scatter
function _constant_adjoint_nd_apply!(
        f_bar::AbstractArray{Tv, N},
        adj::ConstantAdjointND,
        y_bar::Real,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, N}
    _has_any_derivative(ops, Val(N)) && return nothing
    @inbounds begin
        per_axis = adj.anchors[1]
        _any_fill_oob(per_axis, adj.extraps) && return nothing
        target = _constant_nd_target(per_axis, adj.sides)
        f_bar[target...] += y_bar
    end
    return nothing
end

# Vector/Tuple y_bar: loop over elements
function _constant_adjoint_nd_apply!(
        f_bar::AbstractArray{Tv, N},
        adj::ConstantAdjointND,
        y_bar,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, N}
    # Any derivative → f_bar stays zero (already zero-filled by caller)
    _has_any_derivative(ops, Val(N)) && return nothing

    @inbounds for q in eachindex(y_bar)
        per_axis = adj.anchors[q]
        _any_fill_oob(per_axis, adj.extraps) && continue
        target = _constant_nd_target(per_axis, adj.sides)
        f_bar[target...] += y_bar[q]
    end
    return nothing
end

# ========================================
# Constructor
# ========================================

"""
    constant_adjoint(grids::NTuple{N}, queries; bc=NoBC(), side=NearestSide(), extrap=NoExtrap())

Construct an N-dimensional constant adjoint operator.

# Arguments
- `grids`: N-tuple of grid vectors, one per dimension
- `queries`: Query points (SoA tuple of vectors, AoS, single tuple, SVector, etc.)
- `bc`: Boundary condition (single `AbstractBC` or per-axis tuple). Use
  `PeriodicBC()` for `:inclusive` periodicity or
  `PeriodicBC(endpoint=:exclusive, period=...)` for `:exclusive`. Default `NoBC()`.
- `side`: Side selection mode (single or per-axis tuple)
- `extrap`: Extrapolation mode (single or per-axis tuple). Auto-promoted to
  `WrapExtrap` on periodic axes.

# Returns
`ConstantAdjointND` operator that can be called as `adj(y_bar)` or `adj(f_bar, y_bar)`.

# Example
```julia
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 15)
xq = rand(100)
yq = rand(100)

adj = constant_adjoint((x, y), (xq, yq); side=NearestSide())
f_bar = adj(y_bar)   # returns 20×15 matrix
```
"""
function constant_adjoint(
        grids::NTuple{N, AbstractVector},
        queries::Tuple{AbstractVector, Vararg{AbstractVector}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide, N}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(queries) == N || _throw_ndims_mismatch("query vectors", N, length(queries))
    return _constant_adjoint_dispatch(grids, queries, bc, side, extrap)
end

# Single-tuple query: constant_adjoint((x, y), (0.5, 0.5); ...)
function constant_adjoint(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide, N}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    return constant_adjoint(grids, (query,); bc = bc, side = side, extrap = extrap)
end

# Single-vector query: constant_adjoint((x, y), SVector(0.5, 0.5); ...)
function constant_adjoint(
        grids::NTuple{N, AbstractVector},
        query::AbstractVector{<:Real};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide, N}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(query) == N || _throw_ndims_mismatch("query elements", N, length(query))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return constant_adjoint(grids, (query_tuple,); bc = bc, side = side, extrap = extrap)
end

# Generic query fallback
function constant_adjoint(
        grids::NTuple{N, AbstractVector},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide, N}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    _query_check_ndims(queries, Val(N))
    return _constant_adjoint_dispatch(grids, queries, bc, side, extrap)
end

# Shared resolution path (called by every public overload above).
@inline function _constant_adjoint_dispatch(
        grids::NTuple{N, AbstractVector}, queries, bc, side, extrap
    ) where {N}
    # Grid stays raw (no `float()` widening); adjoint buffer eltype comes
    # from the protocol's `_output_eltype`.
    Tg = _promote_grid_eltype(grids)
    grids_typed = _convert_grids_typed(grids, Tg)

    bcs = _resolve_bcs_nd(bc, Val(N))
    # 2-arg `_cache_axis` (no Tg closure) — Tg promotion already applied above;
    # mirrors forward ConstantInterpolantND wrap pattern.
    grids_typed = map(_cache_axis, grids_typed, bcs)

    # 5-arg `_resolve_extrap` with bcs: validates extrap/BC compat, auto-promotes
    # `WrapExtrap` on periodic axes.
    extraps = _resolve_extrap(extrap, bcs, grids_typed, Val(N), Tg)
    sides = _resolve_side_nd(side, Val(N))
    return _build_constant_nd_adjoint(grids_typed, queries, bcs, extraps, sides)
end

"""
    _build_constant_nd_adjoint(grids, queries, bcs, extraps, sides)

Internal builder for `ConstantAdjointND`. `grids` here are wrapped via
`_cache_axis`; `length(grids[d]) == n+1` (logical) for `:exclusive` periodic
axes — the ND adjoint protocol's `_adjoint_output_size` and
`_adjoint_apply_exclusive_nd!` trim back to `n` for the user-visible output.
"""
function _build_constant_nd_adjoint(
        grids::NTuple{N, AbstractVector{Tg}},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        sides::Tuple{Vararg{AbstractSide, N}}
    ) where {N, Tg}
    # Validate all axes have at least 2 points
    @inbounds for d in 1:N
        length(grids[d]) >= 2 || _throw_adjoint_grid_too_small(d, length(grids[d]))
    end

    anchors = _bake_constant_nd_anchors(grids, queries, extraps)
    grid_size = ntuple(d -> length(grids[d]), Val(N))

    return ConstantAdjointND(grids, bcs, extraps, sides, anchors, grid_size)
end

# Matrix materialization inherited from AbstractAdjointND (adjoint_protocol.jl)
