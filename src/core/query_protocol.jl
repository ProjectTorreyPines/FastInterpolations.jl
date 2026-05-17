# ========================================
# ND Query Protocol
# ========================================
#
# Extensible protocol for ND query containers (3 core functions + optional validation).
#
# Generic defaults delegate to Base (length, getindex, eltype), so any type
# with correct Base semantics works out of the box — no protocol to implement.
# SoA (Tuple of Vectors) is the only built-in override (Base semantics differ).
#
# Override _query_* only when Base interfaces have wrong semantics for your type:
#   import FastInterpolations: _query_length, _query_extract, _query_eltype, _query_validate
#   _query_length(q::MyQueries) = ...
#   _query_extract(q::MyQueries, k) = ...       # return k-th point (any indexable)
#   _query_eltype(q::MyQueries) = ...
#   _query_validate(q::MyQueries) = ...         # optional: cross-container consistency checks

# ── Protocol function 1: query count ──

@inline _query_length(q::Tuple{Vararg{AbstractVector}}) = length(q[1])
@inline _query_length(q) = length(q)

# ── Protocol function 2: k-th query point extraction ──
# User-facing: simple getindex-like interface. No Val{N} needed.
# Override this for custom query types — just return the k-th point (any indexable).
# SoA returns NTuple directly (N known from tuple type parameter).

@inline _query_extract(q::Tuple{Vararg{AbstractVector, N}}, k) where {N} =
    ntuple(d -> @inbounds(q[d][k]), Val(N))
@inline _query_extract(q, k) = @inbounds q[k]

# ── Internal: NTuple-guaranteed extraction for kernel consumption ──
# Wraps _query_extract with _as_ntuple — ensures NTuple{N, <:Real} for @generated kernels.
# NOT part of the user protocol — users never override this.

@inline _extract_query_point(q, k, ::Val{N}) where {N} =
    _as_ntuple(_query_extract(q, k), Val(N))

@inline _as_ntuple(x::NTuple{N, <:Real}, ::Val{N}) where {N} = x
@inline _as_ntuple(x, ::Val{N}) where {N} = ntuple(d -> @inbounds(x[d]), Val(N))

# ── Protocol function 3: scalar element type for output allocation ──
# Extracts the scalar floating type from query elements (not the element type itself).
# e.g. Vector{NTuple{2,Float64}} → Float64, not NTuple{2,Float64}.

@inline _query_eltype(q::Tuple{Vararg{AbstractVector}}) = promote_type(map(eltype, q)...)
@inline _query_eltype(q) = _scalar_eltype(eltype(q))

# Scalar type extraction helpers — dispatch on element type
@inline _scalar_eltype(::Type{T}) where {T <: Real} = T
@inline _scalar_eltype(::Type{T}) where {T <: Tuple} = promote_type(fieldtypes(T)...)
@inline _scalar_eltype(::Type{Any}) = _throw_query_eltype_any()
@inline _scalar_eltype(::Type{T}) where {T} = eltype(T)

@noinline _throw_query_eltype_any() = throw(
    ArgumentError(
        "cannot determine scalar element type from query container with eltype Any; " *
            "override `FastInterpolations._query_eltype(q::YourType)` for your query type"
    )
)

# ── Protocol function 4: query consistency validation ──
# Default: no-op (AoS, custom types — single container, no mismatch possible).
# SoA: per-axis length consistency check (axes are independent containers).
# Override for custom multi-container query types that need validation.

@inline _query_validate(q) = nothing

@inline function _query_validate(q::Tuple{Vararg{AbstractVector}})
    n = length(q[1])
    for d in 2:length(q)
        length(q[d]) == n || _throw_query_axis_mismatch(n, d, length(q[d]))
    end
    return nothing
end

# ── Dimensionality validation ──
# SoA: ndims = tuple length (checked directly, no point extraction needed).
# Generic (AoS, SVector, etc.): check the first point's length
# to produce a clear error instead of an opaque BoundsError from _as_ntuple.

# SoA: ndims is the tuple length (number of axes), not a point property.
# Avoids BoundsError from _query_extract when axes have inconsistent lengths.
function _query_check_ndims(queries::Tuple{Vararg{AbstractVector}}, ::Val{N}) where {N}
    length(queries) == N || _throw_query_ndims_mismatch(N, length(queries))
    return nothing
end

function _query_check_ndims(queries, ::Val{N}) where {N}
    nq = _query_length(queries)
    nq == 0 && return nothing
    pt = _query_extract(queries, 1)
    nd = length(pt)
    nd == N || _throw_query_ndims_mismatch(N, nd)
    return nothing
end

# ── Error helpers (@noinline cold path) ──

@noinline _throw_query_output_mismatch(nq, no) =
    throw(DimensionMismatch("output length $no != query length $nq"))
@noinline _throw_query_axis_mismatch(n1, d, nd) =
    throw(DimensionMismatch("query axis lengths differ: dim 1 has $n1, dim $d has $nd"))
@noinline _throw_query_ndims_mismatch(expected, got) =
    throw(DimensionMismatch("expected $expected-dimensional query points, got $got-dimensional"))

# ── Pre-loop NoExtrap domain validation ──
#
# Validates all queries against grid bounds BEFORE @inbounds eval loops.
# This is the SOLE safety gate: _extrap_axis wraps _handle_axis_extrap in
# @inbounds, so @boundscheck _check_domain(::NoExtrap) is ALWAYS elided
# in production mode (--check-bounds=auto).
#
# Three dispatch methods:
# 1. Scalar point: per-axis check via map (for oneshot single-point paths)
# 2. SoA (tuple-of-vectors): efficient min/max per axis via vector _check_domain
# 3. Generic (AoS): per-query per-axis scalar _check_domain fallback
#
# Non-NoExtrap axes are no-ops via _check_domain dispatch.

# Scalar point: per-axis scalar check (for oneshot single-point paths)
# Uses map (not for-loop) to dispatch per-element with concrete types,
# avoiding Union boxing on heterogeneous extrap tuples.
@inline function _validate_nd_domain(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Real, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    map(_check_domain, grids, query, extraps)
    return nothing
end

# SoA queries: O(nq) per axis via minimum/maximum reduction
# Uses map for same reason as scalar (heterogeneous extrap safety).
@inline function _validate_nd_domain(
        grids::NTuple{N, AbstractVector},
        queries::Tuple{AbstractVector, Vararg{AbstractVector}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    isempty(first(queries)) && return nothing
    map(_check_domain, grids, queries, extraps)
    return nothing
end

# Generic queries (AoS, etc.): per-query per-axis scalar check.
# Uses map over (grids, extraps, axes) to avoid runtime-Int indexing of
# heterogeneous grid tuples — prevents Union boxing on mixed-grid combos.
function _validate_nd_domain(
        grids::NTuple{N, AbstractVector},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    any(e -> e isa NoExtrap, extraps) || return nothing
    nq = _query_length(queries)
    axes = ntuple(identity, Val(N))
    for q in 1:nq
        query_q = _extract_query_point(queries, q, Val(N))
        map(grids, extraps, axes) do grid, extrap, d
            extrap isa NoExtrap && _check_domain(grid, query_q[d], NoExtrap())
            nothing
        end
    end
    return nothing
end

# ── Batch-level domain check returning per-axis InBounds-or-original ──
#
# ND analog of 1D `_check_domain(x, xi::AbstractVector, extrap) -> InBounds | extrap`.
# Per-axis: if all queries on axis d are in-bounds, return `InBounds()` for that
# axis; otherwise keep the original extrap. Result is `NTuple{N, AbstractExtrap}`
# where each slot is `Union{InBounds, original}` — Julia specializes the map per
# axis (heterogeneous map, no boxing) into a concrete tuple type.
#
# Downstream `_handle_axis_extrap(q, axis, ::InBounds) = q` (nd_utils.jl) is a
# no-op for InBounds axes, eliding wrap/clamp/fill per-query branches when all
# queries are in-bounds. Fixes Julia 1.10+ deep-`@inbounds`-non-propagation
# overhead (ND NoExtrap batch N=2 Nq=1000: 2000 calls → 0 calls expected).
#
# SoA only — AoS/generic fallback returns `extraps` unchanged (per-axis min/max
# would require a linear pass via `_query_extract`; deferred follow-up).

# SoA batch: reuse 1D vector `_check_domain` per axis. NoExtrap throws via the
# 1D `@boundscheck _is_all_inbounds || _throw_batch_oob`; Wrap/Clamp/Fill return
# `InBounds()` when all in-bounds, original extrap otherwise; ExtendExtrap (and
# any fallback) returns itself unchanged.
@inline function _check_domain_nd(
        grids::NTuple{N, AbstractVector},
        queries::Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    isempty(first(queries)) && return extraps
    return map(_check_domain, grids, queries, extraps)
end

# Scalar point: no batch-level promotion to do — defer to per-axis `_check_domain`
# inside the eval path. Validate via existing `_validate_nd_domain` then return
# `extraps` unchanged.
@inline function _check_domain_nd(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Real, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    _validate_nd_domain(grids, query, extraps)
    return extraps
end

# Generic / AoS: deferred — single linear pass with per-axis min/max accumulators
# could enable AoS promotion later. For now, validate (NoExtrap throw) and
# return `extraps` unchanged so the inner per-query path handles each axis.
@inline function _check_domain_nd(
        grids::NTuple{N, AbstractVector},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    _validate_nd_domain(grids, queries, extraps)
    return extraps
end
