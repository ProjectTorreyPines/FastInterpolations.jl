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

# :extension — extend boundary cell polynomial beyond domain
@inline function _dispatch_extrap_integrate_1d(
    ::Val{:extension}, in_domain_fn, x, y_left, y_right, x0::Real, x1::Real, ::Type{Tout}
) where Tout
    return in_domain_fn(x0, x1)
end

# Generic 1D split-accumulate: split [a,b] into cells, call partial/full kernels.
# `partial_fn(i, xL, h, a2, b2)` — integrate cell i from a2 to b2, with cell width h
# `full_fn(i, h)` — integrate full cell i with cell width h
# Uses no-spacing search_interval variant (avoids VectorSpacing allocation).
@inline function _integrate_1d_cellwise(
    x::AbstractVector, a::Real, b::Real,
    searcher::S, partial_fn::PF, full_fn::FF, ::Type{Tout}
) where {S<:Searcher, PF, FF, Tout}
    sign, lo, hi = _normalize_bounds_1d(a, b)
    sign == 0 && return zero(Tout)

    i0, xL0, _ = search_interval(searcher, x, lo)
    i1, xL1, _ = search_interval(searcher, x, hi)

    if i0 == i1
        @inbounds h = x[i0 + 1] - x[i0]
        return sign * partial_fn(i0, xL0, h, lo, hi)
    end

    @inbounds h0 = x[i0 + 1] - x[i0]
    total = partial_fn(i0, xL0, h0, lo, xL0 + h0)
    @inbounds for i in (i0 + 1):(i1 - 1)
        total += full_fn(i, x[i + 1] - x[i])
    end
    @inbounds h1 = x[i1 + 1] - x[i1]
    total += partial_fn(i1, xL1, h1, xL1, hi)

    return sign * total
end

# ═══════════════════════════════════════════════════════════════
# Full-domain integration helpers (no search, all cells full)
# ═══════════════════════════════════════════════════════════════

# Scalar accumulator: ∫ over entire grid using full-cell kernel only.
# `full_fn(i, h)` — same closure signature as _integrate_1d_cellwise.
@inline function _integrate_1d_fulldomain(
    x::AbstractVector, full_fn::F, ::Type{Tout}
) where {F, Tout}
    n = length(x)
    total = zero(Tout)
    @inbounds for i in 1:(n - 1)
        total += full_fn(i, x[i + 1] - x[i])
    end
    return total
end

# Prefix-sum (in-place): write cumulative integral into pre-allocated buffer.
# `out` must have length ≥ length(x). Works with Vector or @view(matrix[:, k]).
function _cumulative_integrate_1d!(
    out::AbstractVector, x::AbstractVector, full_fn::F
) where {F}
    n = length(x)
    out[1] = zero(eltype(out))
    @inbounds for i in 1:(n - 1)
        out[i + 1] = out[i] + full_fn(i, x[i + 1] - x[i])
    end
    return out
end

# Allocating wrapper: creates a fresh Vector and fills it.
function _cumulative_integrate_1d(
    x::AbstractVector, full_fn::F, ::Type{Tout}
) where {F, Tout}
    result = Vector{Tout}(undef, length(x))
    return _cumulative_integrate_1d!(result, x, full_fn)
end
