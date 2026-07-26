@inline function _normalize_bounds_1d(a, b)
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

@inline function _dispatch_extrap_integrate_1d(
        ::NoExtrap, in_domain_fn, x, y_left, y_right, x0, x1, ::Type{Tout}
    ) where {Tout}
    _check_domain(x, min(x0, x1), NoExtrap())
    _check_domain(x, max(x0, x1), NoExtrap())
    return in_domain_fn(x0, x1)
end

@inline function _dispatch_extrap_integrate_1d(
        ::ClampExtrap, in_domain_fn, x, y_left, y_right, x0, x1, ::Type{Tout}
    ) where {Tout}
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

@inline function _dispatch_extrap_integrate_1d(
        e::FillExtrap, in_domain_fn, x, _y_left, _y_right, x0, x1, ::Type{Tout}
    ) where {Tout}
    sign, lo, hi = _normalize_bounds_1d(x0, x1)
    sign == 0 && return zero(Tout)
    xmin, xmax = first(x), last(x)
    total = zero(Tout)
    v = e.fill_value
    if lo < xmin
        total += v * (min(hi, xmin) - lo)
    end
    lo_in = max(lo, xmin)
    hi_in = min(hi, xmax)
    if hi_in > lo_in
        total += in_domain_fn(lo_in, hi_in)
    end
    if hi > xmax
        total += v * (hi - max(lo, xmax))
    end
    return sign * total
end

@inline function _dispatch_extrap_integrate_1d(
        ::WrapExtrap, in_domain_fn, x, y_left, y_right, x0, x1, ::Type{Tout}
    ) where {Tout}
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

@inline function _dispatch_extrap_integrate_1d(
        ::ExtendExtrap, in_domain_fn, x, y_left, y_right, x0, x1, ::Type{Tout}
    ) where {Tout}
    return in_domain_fn(x0, x1)
end

# Generic 1D split-accumulate: split [a,b] into cells, call partial/full kernels.
# `partial_fn(i, xL, h, a2, b2)` — integrate cell i from a2 to b2, with cell width h
# `full_fn(i, h)` — integrate full cell i with cell width h
#
# Cell widths are fetched via `_get_h(x, i)` directly on the grid. For wrapped
# grids (`_CachedRange` / `_CachedVector` / `_ExclusivePeriodicAxis`) this is a
# cached lookup; for raw `AbstractRange` / `AbstractVector` (Quadratic legacy
# Vector grids) it falls back to `step(x)` / `float(x[i+1] - x[i])` — single
# subtraction per cell, no allocation.
@inline function _integrate_1d_cellwise(
        x::AbstractVector, a, b,
        searcher::S, partial_fn::PF, full_fn::FF, ::Type{Tout}
    ) where {S <: Searcher, PF, FF, Tout}
    sign, lo, hi = _normalize_bounds_1d(a, b)
    sign == 0 && return zero(Tout)

    i0, _, xL0, _ = search_interval(searcher, x, lo)
    i1, _, xL1, _ = search_interval(searcher, x, hi)

    if i0 == i1
        h = _get_h(x, i0)
        return convert(Tout, sign * partial_fn(i0, xL0, h, lo, hi))
    end

    # The two end cells pair a caller BOUND with a grid NODE, and `partial_fn`
    # subtracts both from the same node — so they must share a type. They already
    # do whenever the bound is spelled in the grid's own unit (and always on Real
    # grids, where `promote` is the identity), but a `cm` grid integrated between
    # `m` bounds would otherwise hand the kernel one `m` offset and one `cm` one.
    # `promote` also widens a Float64 bound against a Float32 grid, as before.
    h0 = _get_h(x, i0)
    lo0, hi0 = promote(lo, xL0 + h0)
    total = convert(Tout, partial_fn(i0, xL0, h0, lo0, hi0))
    @inbounds @simd for i in (i0 + 1):(i1 - 1)
        total += convert(Tout, full_fn(i, _get_h(x, i)))
    end
    h1 = _get_h(x, i1)
    lo1, hi1 = promote(xL1, hi)
    total += convert(Tout, partial_fn(i1, xL1, h1, lo1, hi1))

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
    # @simd: pure reduction (only cross-iteration dep is `total`). LLVM won't
    # vectorize an FP reduction without it (non-associative `+` → scalar chain,
    # ~6× slower); no @fastmath — only the accumulation order is relaxed.
    @inbounds @simd for i in 1:(n - 1)
        total += full_fn(i, _get_h(x, i))
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
        out[i + 1] = out[i] + full_fn(i, _get_h(x, i))
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
