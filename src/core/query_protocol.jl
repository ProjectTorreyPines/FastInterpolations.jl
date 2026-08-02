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
# Shaped SoA (tuple of same-size coordinate arrays): count from the first axis.
@inline _query_length(q::Tuple{AbstractArray, Vararg{AbstractArray}}) = length(first(q))
@inline _query_length(q) = length(q)

# ── Optional: query output shape ──
# The batch cores fill a length-`_query_length` buffer by LINEAR index, so a
# query container that carries a shape reports it here: the allocating path
# builds an output of that shape and in-place callers pass a matching array
# (linear indexing lands each result at the right N-D position, column-major).
# `_query_length == prod(_query_size)` mirrors Base's `length`/`size`. Default
# is a flat 1-D output (a plain vector), so ordinary batch queries are unchanged.

@inline _query_size(q) = (_query_length(q),)
# An ordinary array query reports its own shape, so the allocating path returns an
# `Array` of that shape and in-place callers pass a matching array (column-major
# linear indexing lands each result at its N-D slot). SoA reports the shared axis size.
@inline _query_size(q::AbstractArray) = size(q)
@inline _query_size(q::Tuple{AbstractArray, Vararg{AbstractArray}}) = size(first(q))

# ── Optional: shaped output allocation + exact-size validation ──
# Single source of truth for batch output shape. Allocating builds a dense `Array`
# of the query shape (never `similar(q)`: a query container may be read-only, lazy,
# or point-valued). In-place requires an EXACT size match — equal length with a
# different shape is rejected (a length-4 vector is not a valid sink for a 2×2 query).
@inline _alloc_query_output(::Type{T}, q) where {T} = Array{T}(undef, _query_size(q))

@inline function _check_query_output_size(output, q)
    expected = _query_size(q)
    size(output) == expected || _throw_query_output_size_mismatch(expected, size(output))
    return nothing
end

# ── Protocol function 2: k-th query point extraction ──
# User-facing: simple getindex-like interface. No Val{N} needed.
# Override this for custom query types — just return the k-th point (any indexable).
# SoA returns NTuple directly (N known from tuple type parameter).

@inline _query_extract(q::Tuple{Vararg{AbstractVector, N}}, k) where {N} =
    ntuple(d -> @inbounds(q[d][k]), Val(N))
# Shaped SoA: linear index k into each coordinate array (column-major pairwise).
@inline _query_extract(q::Tuple{Vararg{AbstractArray, N}}, k) where {N} =
    ntuple(d -> @inbounds(q[d][k]), Val(N))
@inline _query_extract(q, k) = @inbounds q[k]

# ── Internal: NTuple-guaranteed extraction for kernel consumption ──
# Wraps _query_extract with _as_ntuple — normalizes to an NTuple{N} for @generated
# kernels (a Real tuple short-circuits; duck/unit queries take the generic arm).
# NOT part of the user protocol — users never override this.

@inline _extract_query_point(q, k, ::Val{N}) where {N} =
    _as_ntuple(_query_extract(q, k), Val(N))

# Grid-aware form for the batch loops: a bare `GridIdx(k)` carries no coordinate
# (`val = NaN` poison) until it meets its axis, and the scalar entries resolve at
# the door. Batch points must resolve too — an unresolved index on an
# INTERPOLATING axis silently evaluates the kernel at NaN. Identity for every
# non-GridIdx element, so Real batches keep their codegen.
@inline _extract_query_point(q, k, ::Val{N}, grids::Tuple{Vararg{AbstractVector, N}}) where {N} =
    map(_resolve_grididx, _extract_query_point(q, k, Val(N)), grids)

@inline _as_ntuple(x::NTuple{N, <:Real}, ::Val{N}) where {N} = x
@inline _as_ntuple(x, ::Val{N}) where {N} = ntuple(d -> @inbounds(x[d]), Val(N))

# ── Protocol function 3: scalar element type for output allocation ──
# Extracts the scalar floating type from query elements (not the element type itself).
# e.g. Vector{NTuple{2,Float64}} → Float64, not NTuple{2,Float64}.

@inline _query_eltype(q::Tuple{Vararg{AbstractVector}}) = promote_type(map(eltype, q)...)
@inline _query_eltype(q::Tuple{AbstractArray, Vararg{AbstractArray}}) = promote_type(map(eltype, q)...)
@inline _query_eltype(q) = _scalar_eltype(eltype(q))

# Scalar type extraction helpers — dispatch on element type
@inline _scalar_eltype(::Type{T}) where {T <: Real} = T
@inline _scalar_eltype(::Type{T}) where {T <: Tuple} = promote_type(fieldtypes(T)...)
@inline _scalar_eltype(::Type{Any}) = _throw_query_eltype_any()
@inline _scalar_eltype(::Type{T}) where {T} = eltype(T)
@inline _carrier_oneunit(::Type{T}) where {T} = oneunit(_scalar_eltype(T))

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

# Shaped SoA: every coordinate array must share the SAME size, not merely length
# (a 2×2 and a 4×1 have equal length but different shape and cannot align pairwise).
@inline function _query_validate(q::Tuple{AbstractArray, Vararg{AbstractArray}})
    sz = size(first(q))
    for d in 2:length(q)
        size(q[d]) == sz || _throw_query_axis_size_mismatch(sz, d, size(q[d]))
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

# Shaped SoA: ndims = number of axes (tuple length). Mandatory peer — without it a
# tuple-of-arrays falls to the generic method below, which reads the first WHOLE
# array as "point 1" and throws a spurious ndims mismatch.
function _query_check_ndims(queries::Tuple{AbstractArray, Vararg{AbstractArray}}, ::Val{N}) where {N}
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
@noinline _throw_query_output_size_mismatch(expected, got) =
    throw(DimensionMismatch("output size $got != query size $expected"))
@noinline _throw_query_axis_mismatch(n1, d, nd) =
    throw(DimensionMismatch("query axis lengths differ: dim 1 has $n1, dim $d has $nd"))
@noinline _throw_query_axis_size_mismatch(sz1, d, szd) =
    throw(DimensionMismatch("query axis sizes differ: dim 1 has size $sz1, dim $d has size $szd"))
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

# 1D NoExtrap domain validation: throw `DomainError` for any out-of-domain query.
# Routes per-query through `_check_domain`, so the `_CachedRange` widened-bracket
# acceptance is applied and the error reports the physical endpoints — callers
# (the 1D adjoints) never touch `_domain_bounds` directly.
@inline function _validate_domain(axis::AbstractVector, queries::AbstractVector)
    @inbounds for i in eachindex(queries)
        _check_domain(axis, queries[i], NoExtrap())
    end
    return nothing
end

# ── ND domain validation, returning the per-axis effective (search) extrap ──
# Validates (a NoExtrap axis throws `DomainError` when OOB) AND returns the effective per-axis
# extrap: an in-domain NoExtrap axis promotes to `InBounds()` (→ lean search), every other mode
# passes through. Callers needing only validation ignore the return. Mirrors the scalar 1D
# `_check_domain`; the heterogeneous `map` specializes per axis → a concrete tuple (no boxing).
#
# Scalar point (oneshot single-point paths). `map` dispatches per-axis on concrete types.
@inline function _validate_nd_domain(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Number, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    # thread the axis index so a NoExtrap OOB names the offending axis (`_check_domain_axis`
    # forwards `dim` only to the NoExtrap thrower; all other modes ignore it).
    return map(_check_domain_axis, grids, query, extraps, ntuple(identity, Val(N)))
end

# SoA batch (Real vectors): per-axis vector `_check_domain` (O(nq) min/max). All-in-bounds →
# `InBounds()` for that axis, else the original extrap; empty batch → `extraps` unchanged. The
# `<:Real` bound is load-bearing: batch `_check_domain` is only defined for `AbstractVector{<:Real}`,
# so a non-Real / duck-typed SoA batch falls to the generic method below (validate, no promotion).
#
# Exclusive-last promotions are DEMOTED to closed here (`_check_axis_batch_closed`):
# unions nested in a tuple type don't union-split, so per-axis exclusive would send an
# abstract extrap tuple into the batch locate loop (measured ~10× on 2D SoA). The
# strict upgrade happens tuple-level on the all-NoExtrap lane below.
@inline function _validate_nd_domain(
        grids::NTuple{N, AbstractVector},
        queries::Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    isempty(first(queries)) && return extraps
    return map(_check_axis_batch_closed, grids, queries, extraps, ntuple(identity, Val(N)))
end

# Per-axis SoA check with concrete return shapes: validation/throw from the batch
# `_check_domain` verbatim; only the exclusive upgrade collapses back to closed.
# `dim` reaches only the NoExtrap thrower (axis-named message); other modes ignore it.
@inline _check_axis_batch_closed(g, q, e::AbstractExtrap, dim::Int) = _check_domain(g, q, e)
@inline function _check_axis_batch_closed(g, q, e::NoExtrap, dim::Int)
    _check_domain(g, q, e, dim)      # throws on OOB; promotion result intentionally discarded
    return InBounds()
end
@inline function _check_axis_batch_closed(g, q, e::Union{ClampExtrap, FillExtrap, WrapExtrap}, dim::Int)
    r = _check_domain(g, q, e)
    return r isa InBounds ? InBounds() : e
end

# All-unit-step, all-NoExtrap SoA lane: per-axis 3-outcome checks (validate/throw),
# then ALL-OR-NOTHING — every axis exclusive → all-exclusive tuple, else all-closed.
# The transient per-axis unions are consumed by one tuple type test and never reach
# the locate loop; mixed-grid / mixed-extrap tuples take the demoting generic lane.
@inline function _validate_nd_domain(
        grids::NTuple{N, _CachedRange{<:Any, <:Any, <:_AbstractUnitStep}},
        queries::Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}},
        extraps::NTuple{N, NoExtrap}
    ) where {N}
    isempty(first(queries)) && return extraps
    effs = map(_check_domain_axis, grids, queries, extraps, ntuple(identity, Val(N)))
    return effs isa NTuple{N, InBounds{:inclusive, :exclusive}} ?
        ntuple(_ -> InBounds(last = :exclusive), Val(N)) :
        ntuple(_ -> InBounds(), Val(N))
end

# Generic (AoS, etc.): per-query per-axis NoExtrap check (min/max promotion deferred), so return
# `extraps` unchanged. `map` over (grids, extraps, axes) dodges runtime-Int indexing of the
# heterogeneous grid tuple (no Union boxing on mixed-grid combos).
function _validate_nd_domain(
        grids::NTuple{N, AbstractVector},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    if any(e -> e isa NoExtrap, extraps)
        nq = _query_length(queries)
        axes = ntuple(identity, Val(N))
        for q in 1:nq
            query_q = _extract_query_point(queries, q, Val(N))
            map(grids, extraps, axes) do grid, extrap, d
                extrap isa NoExtrap && _check_domain(grid, query_q[d], NoExtrap(), d)
                nothing
            end
        end
    end
    return extraps
end
