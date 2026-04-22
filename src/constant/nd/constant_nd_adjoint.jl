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
    _bake_constant_nd_anchors(grids, spacings, queries, extraps)

Precompute per-axis `_ConstantAnchoredQuery` for each query point.

Per-axis processing:
1. Extrapolate position (wrap/clamp/extend/validate)
2. Search interval → get idx, h, dL
3. Detect OOB for FillExtrap skip

For constant interpolation with ExtendExtrap, the forward ND path passes
through OOB queries to the kernel (unlike 1D which clamps). The adjoint
follows the same convention for ND consistency.
"""
function _bake_constant_nd_anchors(
        grids::NTuple{N, AbstractVector{Tg}},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N, Tg}
    nq = _query_length(queries)
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps)

    anchors = Vector{NTuple{N, _ConstantAnchoredQuery{Tg, Tg}}}(undef, nq)
    @inbounds for q in 1:nq
        query_q = _extract_query_point(queries, q, Val(N))
        per_axis = ntuple(Val(N)) do d
            xq_raw = Tg(query_q[d])
            xq_d = _extrap_axis(xq_raw, grids[d], extraps[d])
            idx, _, xL, _ = search_interval(DEFAULT_SEARCHER, grids[d], spacings[d], xq_d)
            h = _get_h(spacings[d], idx)
            dL = xq_d - xL

            # Determine OOB side flag
            is_oob = xq_raw < first(grids[d]) || xq_raw > last(grids[d])
            state_flag = if is_oob
                xq_raw < first(grids[d]) ? OOB_LEFT : OOB_RIGHT
            else
                IN_DOMAIN
            end

            return _ConstantAnchoredQuery{Tg, Tg}(idx, xq_d, state_flag, h, dL)
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
        per_axis::NTuple{N, _ConstantAnchoredQuery{Tg, Tg}},
        sides::Tuple{Vararg{AbstractSide, N}}
    ) where {N, Tg}
    return ntuple(Val(N)) do d
        aq = per_axis[d]
        aq.idx + _compute_single_offset(sides[d], aq.h, aq.dL)
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
        adj::ConstantAdjointND{Tg, N},
        y_bar::Real,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N}
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
        adj::ConstantAdjointND{Tg, N},
        y_bar,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N}
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
    constant_adjoint(grids::NTuple{N}, queries; side=NearestSide(), extrap=NoExtrap())

Construct an N-dimensional constant adjoint operator.

# Arguments
- `grids`: N-tuple of grid vectors, one per dimension
- `queries`: Query points (SoA tuple of vectors, AoS, single tuple, SVector, etc.)
- `side`: Side selection mode (single or per-axis tuple)
- `extrap`: Extrapolation mode (single or per-axis tuple)

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
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide, N}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(queries) == N || _throw_ndims_mismatch("query vectors", N, length(queries))
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    extraps = _resolve_extrap_nd(extrap, nothing, Val(N), Tg)
    # Materialize WrapExtrap{Nothing} so kernels never see the unmaterialized
    # singleton.
    extraps = map(_materialize_extrap, grids_typed, extraps)
    sides = _resolve_side_nd(side, Val(N))
    return _build_constant_nd_adjoint(grids_typed, queries, extraps, sides)
end

# Single-tuple query: constant_adjoint((x, y), (0.5, 0.5); ...)
function constant_adjoint(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Real, N}};
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide, N}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    return constant_adjoint(grids, (query,); side = side, extrap = extrap)
end

# Single-vector query: constant_adjoint((x, y), SVector(0.5, 0.5); ...)
function constant_adjoint(
        grids::NTuple{N, AbstractVector},
        query::AbstractVector{<:Real};
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide, N}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(query) == N || _throw_ndims_mismatch("query elements", N, length(query))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return constant_adjoint(grids, (query_tuple,); side = side, extrap = extrap)
end

# Generic query fallback
function constant_adjoint(
        grids::NTuple{N, AbstractVector},
        queries;
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide, N}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    _query_check_ndims(queries, Val(N))
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    extraps = _resolve_extrap_nd(extrap, nothing, Val(N), Tg)
    # Materialize WrapExtrap{Nothing} so kernels never see the unmaterialized
    # singleton.
    extraps = map(_materialize_extrap, grids_typed, extraps)
    sides = _resolve_side_nd(side, Val(N))
    return _build_constant_nd_adjoint(grids_typed, queries, extraps, sides)
end

"""
    _build_constant_nd_adjoint(grids, queries, extraps, sides)

Internal builder for `ConstantAdjointND`. Separated from the public API so that
`Tg` is bound via the argument type, making the return type fully inferrable.
"""
function _build_constant_nd_adjoint(
        grids::NTuple{N, AbstractVector{Tg}},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        sides::Tuple{Vararg{AbstractSide, N}}
    ) where {N, Tg}
    # Validate all axes have at least 2 points
    @inbounds for d in 1:N
        length(grids[d]) >= 2 || _throw_adjoint_grid_too_small(d, length(grids[d]))
    end

    spacings = _create_spacings_typed(grids)
    anchors = _bake_constant_nd_anchors(grids, spacings, queries, extraps)
    grid_size = ntuple(d -> length(grids[d]), Val(N))

    return ConstantAdjointND(grids, spacings, extraps, sides, anchors, grid_size)
end

# Matrix materialization inherited from AbstractAdjointND (adjoint_protocol.jl)
