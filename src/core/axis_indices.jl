# ========================================
# Fixed-size per-axis index representations
# ========================================
#
# `_AbstractIndices{K}` is the common protocol for an ordered collection of K
# grid indices along one axis. The concrete type records how those indices are
# represented:
#
#   * `_ContiguousIndices{K}` stores only the first index and derives the rest.
#   * `_ExplicitIndices{K}` stores every index, including wrapped/repeated ones.
#
# The types deliberately do not subtype `AbstractVector`. Hot kernels need only
# fixed-size indexing and length; avoiding the array interface keeps the
# protocol small and prevents generic array fallbacks from entering core paths.
#
# `K` is the number of ordered indices along ONE axis, not the spatial dimension.
# Two distinct roles use this family:
#   * `interval` (K=2): the physical cell endpoints `(idxL, idxR)` returned by
#     `search_interval` — what every anchor carries today.
#   * `support` (K=4/6, future): the broader data neighborhood a Local-Hermite
#     method needs to estimate both endpoint slopes. Not stored until a kernel
#     consumes it (dead payload otherwise), and periodic support additionally
#     needs geometry (`x[1] + period` after `x[n]`) kept separate from the index
#     collection — see `_periodic_secant` / `_periodic_cell_width`.

abstract type _AbstractIndices{K} end

"""
    _ContiguousIndices{K}(first)

Ordered indices `first:(first + K - 1)`, represented by the first index only.
"""
struct _ContiguousIndices{K} <: _AbstractIndices{K}
    first::Int
end

"""
    _ExplicitIndices(indices::NTuple{K,Int})
    _ExplicitIndices(i1, i2, ..., iK)

Ordered indices stored explicitly. This representation supports wrapped,
repeated, and otherwise non-contiguous index sequences.
"""
struct _ExplicitIndices{K} <: _AbstractIndices{K}
    indices::NTuple{K, Int}
end

@inline _ExplicitIndices(indices::Vararg{Int, K}) where {K} =
    _ExplicitIndices{K}(indices)

# ────────────────────────────────────────
# Fixed-size access protocol
# ────────────────────────────────────────

@inline function Base.getindex(indices::_ContiguousIndices{K}, k::Int) where {K}
    @boundscheck 1 <= k <= K || throw(BoundsError(indices, k))
    return getfield(indices, :first) + (k - 1)
end

@inline function Base.getindex(indices::_ExplicitIndices{K}, k::Int) where {K}
    @boundscheck 1 <= k <= K || throw(BoundsError(indices, k))
    return @inbounds getfield(indices, :indices)[k]
end

@inline function Base.getindex(indices::_ContiguousIndices{K}, ::Val{J}) where {K, J}
    @boundscheck 1 <= J <= K || throw(BoundsError(indices, J))
    return getfield(indices, :first) + (J - 1)
end

@inline function Base.getindex(indices::_ExplicitIndices{K}, ::Val{J}) where {K, J}
    @boundscheck 1 <= J <= K || throw(BoundsError(indices, J))
    return @inbounds getfield(indices, :indices)[J]
end

@inline Base.length(::_AbstractIndices{K}) where {K} = K
@inline Base.firstindex(::_AbstractIndices) = 1
@inline Base.lastindex(::_AbstractIndices{K}) where {K} = K
@inline Base.first(indices::_AbstractIndices) = indices[Val(1)]
@inline Base.last(indices::_AbstractIndices{K}) where {K} = indices[Val(K)]

Base.IteratorSize(::Type{<:_AbstractIndices}) = Base.HasLength()
Base.IteratorEltype(::Type{<:_AbstractIndices}) = Base.HasEltype()
Base.eltype(::Type{<:_AbstractIndices}) = Int

@inline function Base.iterate(indices::_AbstractIndices{K}, state::Int = 1) where {K}
    state > K && return nothing
    return (indices[state], state + 1)
end

# ────────────────────────────────────────
# Representation-independent value semantics
# ────────────────────────────────────────

@inline _indices_tuple(indices::_AbstractIndices{K}) where {K} =
    ntuple(k -> indices[k], Val(K))

@inline Base.:(==)(a::_AbstractIndices{K}, b::_AbstractIndices{K}) where {K} =
    _indices_tuple(a) == _indices_tuple(b)

@inline Base.hash(indices::_AbstractIndices, h::UInt) =
    hash(_indices_tuple(indices), hash(:_AbstractIndices, h))
