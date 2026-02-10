@inline function _normalize_bounds_nd(lo::Tuple{Vararg{Real,N}}, hi::Tuple{Vararg{Real,N}}) where {N}
    nflips = 0
    @inbounds for d in 1:N
        nflips += (lo[d] > hi[d])
    end
    sign = iseven(nflips) ? 1 : -1
    lo2 = ntuple(d -> @inbounds(min(lo[d], hi[d])), Val(N))
    hi2 = ntuple(d -> @inbounds(max(lo[d], hi[d])), Val(N))
    return sign, lo2, hi2
end

# Locate cell index ranges for ND integration bounds
@inline function _nd_cell_ranges(
    grids::NTuple{N,AbstractVector},
    spacings,
    lo::Tuple{Vararg{Real,N}},
    hi::Tuple{Vararg{Real,N}},
    search_tuple,
    hint
) where {N}
    idx_lo = ntuple(Val(N)) do d
        s = _to_searcher(search_tuple[d], isnothing(hint) ? nothing : hint[d])
        i, _, _ = search_interval(s, grids[d], spacings[d], lo[d])
        i
    end
    idx_hi = ntuple(Val(N)) do d
        s = _to_searcher(search_tuple[d], isnothing(hint) ? nothing : hint[d])
        i, _, _ = search_interval(s, grids[d], spacings[d], hi[d])
        i
    end
    return idx_lo, idx_hi
end

# Shared ND preamble: normalize bounds, domain checks, cell range computation.
@inline function _integrate_nd_preamble(
    grids, spacings, lo::Tuple{Vararg{Real,N}}, hi::Tuple{Vararg{Real,N}},
    search, hint
) where {N}
    sign, lo2, hi2 = _normalize_bounds_nd(lo, hi)
    @inbounds for d in 1:N
        _check_domain(grids[d], lo2[d], Val(:none))
        _check_domain(grids[d], hi2[d], Val(:none))
    end
    search_tuple = _resolve_search_nd(search, Val(N))
    idx_lo, idx_hi = _nd_cell_ranges(grids, spacings, lo2, hi2, search_tuple, hint)
    return (sign, lo2, hi2, idx_lo, idx_hi)
end

# Per-cell geometry: indices, cell widths, and local integration bounds.
@inline function _nd_cell_geom(
    grids, spacings, lo2, hi2, I::CartesianIndex{N}, ::Val{N}
) where {N}
    idx  = ntuple(d -> I[d], Val(N))
    hs   = ntuple(d -> @inbounds(_get_h(spacings[d], idx[d])), Val(N))
    Ls   = ntuple(d -> @inbounds(grids[d][idx[d]]), Val(N))
    Rs   = ntuple(d -> @inbounds(grids[d][idx[d] + 1]), Val(N))
    ulos = ntuple(d -> max(lo2[d], Ls[d]) - Ls[d], Val(N))
    uhis = ntuple(d -> min(hi2[d], Rs[d]) - Ls[d], Val(N))
    return (idx, hs, ulos, uhis)
end
