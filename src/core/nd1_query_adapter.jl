# ========================================
# N=1 scalar-query adapter
# ========================================
# The 1D batch engines read one scalar per index; an ND caller at N=1 may hand a
# 1-tuple grid a POINT container (AoS, `Vector{Vector}`, shaped AoS, `GriddedQuery`).
# `_scalar_query` bridges the two: numeric arrays / single-axis SoA pass through,
# anything else becomes `_ScalarQuery` — a lazy view that reads point k through the
# query protocol and projects it to its one coordinate. No copy, shape kept.

struct _ScalarQuery{T, M, Q} <: AbstractArray{T, M}
    q::Q
end
@inline _ScalarQuery(q::AbstractArray{<:Any, M}) where {M} =
    _ScalarQuery{_query_eltype(q), M, typeof(q)}(q)

@inline Base.size(v::_ScalarQuery) = _query_size(v.q)
@inline Base.IndexStyle(::Type{<:_ScalarQuery}) = IndexLinear()

# Outer bounds in `@boundscheck` (elided under the engines' `@inbounds`); the arity
# check is not — a ragged / over-dimensioned point throws at its own access.
@inline Base.@propagate_inbounds function Base.getindex(v::_ScalarQuery, i::Int)
    @boundscheck checkbounds(v, i)
    return _scalar_coordinate(_query_extract(v.q, i))
end

@inline _scalar_coordinate(pt::Tuple{Any}) = pt[1]
@inline function _scalar_coordinate(pt)
    length(pt) == 1 || _throw_query_ndims_mismatch(1, length(pt))
    return @inbounds pt[1]
end

@inline _scalar_query(q::AbstractArray{<:Number}) = q
@inline _scalar_query(q::Tuple{AbstractArray}) = only(q)
@inline _scalar_query(q::AbstractArray) = _ScalarQuery(q)
