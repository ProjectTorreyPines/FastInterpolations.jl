@inline function _normalize_bounds_nd(lo::NTuple{N,<:Real}, hi::NTuple{N,<:Real}) where {N}
    sign = 1
    lo2 = ntuple(Val(N)) do d
        if lo[d] <= hi[d]
            lo[d]
        else
            sign = -sign
            hi[d]
        end
    end
    hi2 = ntuple(Val(N)) do d
        if lo[d] <= hi[d]
            hi[d]
        else
            lo[d]
        end
    end
    return sign, lo2, hi2
end

# Locate cell index ranges for ND integration bounds
@inline function _nd_cell_ranges(
    grids::NTuple{N,AbstractVector},
    spacings,
    lo::NTuple{N,<:Real},
    hi::NTuple{N,<:Real},
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
