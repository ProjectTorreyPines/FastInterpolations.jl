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
