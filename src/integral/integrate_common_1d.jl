@inline function _normalize_bounds_1d(a::Real, b::Real)
    if a < b
        return (1, a, b)
    elseif a > b
        return (-1, b, a)
    else
        return (0, a, b)
    end
end

# ═══════════════════════════════════════════════════════════════
# Extrapolation-aware dispatch for 1D integrate
# ═══════════════════════════════════════════════════════════════

# :none — strict domain check
@inline function _dispatch_extrap_integrate_1d(
    ::Val{:none}, in_domain_fn, x, y_left, y_right, x0::Real, x1::Real, ::Type{Tout}
) where Tout
    _check_domain(x, min(x0, x1), Val(:none))
    _check_domain(x, max(x0, x1), Val(:none))
    return in_domain_fn(x0, x1)
end

# :constant — flat tails outside domain
@inline function _dispatch_extrap_integrate_1d(
    ::Val{:constant}, in_domain_fn, x, y_left, y_right, x0::Real, x1::Real, ::Type{Tout}
) where Tout
    sign, lo, hi = _normalize_bounds_1d(x0, x1)
    sign == 0 && return zero(Tout)
    xmin, xmax = first(x), last(x)
    total = zero(Tout)
    if lo < xmin
        total += y_left * (min(hi, xmin) - lo)
    end
    lo_in = max(lo, xmin)
    hi_in = min(hi, xmax)
    if hi_in > lo_in
        total += in_domain_fn(lo_in, hi_in)
    end
    if hi > xmax
        total += y_right * (hi - max(lo, xmax))
    end
    return sign * total
end

# :wrap — periodic decomposition
@inline function _dispatch_extrap_integrate_1d(
    ::Val{:wrap}, in_domain_fn, x, y_left, y_right, x0::Real, x1::Real, ::Type{Tout}
) where Tout
    sign, lo, hi = _normalize_bounds_1d(x0, x1)
    sign == 0 && return zero(Tout)
    xmin, xmax = first(x), last(x)
    period = xmax - xmin
    len = hi - lo

    n_full = floor(Int, len / period)
    rem = len - n_full * period

    I_period = in_domain_fn(xmin, xmax)
    total = Tout(n_full) * I_period

    if rem > zero(rem)
        start = _wrap_to_domain(lo, xmin, xmax)
        stop = start + rem
        if stop <= xmax
            total += in_domain_fn(start, stop)
        else
            total += in_domain_fn(start, xmax)
            total += in_domain_fn(xmin, xmin + (stop - xmax))
        end
    end
    return sign * total
end

# :extension — not yet implemented
@inline function _dispatch_extrap_integrate_1d(
    ::Val{:extension}, in_domain_fn, x, y_left, y_right, x0::Real, x1::Real, ::Type{Tout}
) where Tout
    throw(ArgumentError("integrate with extrap=:extension is not yet implemented"))
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
