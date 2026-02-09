@inline function _normalize_bounds_1d(a::Real, b::Real)
    if a < b
        return (1, a, b)
    elseif a > b
        return (-1, b, a)
    else
        return (0, a, b)
    end
end

# Generic 1D split-accumulate: split [a,b] into cells, call partial/full kernels.
# `partial_fn(i, xL, a, b)` — integrate cell i from local a to b
# `full_fn(i)` — integrate full cell i
@inline function _integrate_1d_cellwise(
    x, spacing, a::Real, b::Real;
    search, hint, partial_fn, full_fn, Tout
)
    sign, lo, hi = _normalize_bounds_1d(a, b)
    sign == 0 && return zero(Tout)

    searcher = _to_searcher(search, hint)
    i0, xL0, _ = search_interval(searcher, x, spacing, lo)
    i1, xL1, _ = search_interval(searcher, x, spacing, hi)

    if i0 == i1
        return sign * partial_fn(i0, xL0, lo, hi)
    end

    total = partial_fn(i0, xL0, lo, xL0 + _get_h(spacing, i0))
    @inbounds for i in (i0 + 1):(i1 - 1)
        total += full_fn(i)
    end
    total += partial_fn(i1, xL1, xL1, hi)

    return sign * total
end
