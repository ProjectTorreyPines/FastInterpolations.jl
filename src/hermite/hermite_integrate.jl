# ========================================
# Hermite Family 1D — Bounded Integration (shared)
# ========================================
# Single top-level dispatch for all AbstractHermiteInterpolant1D subtypes.
# Branches on `itp.dy` concrete type (AbstractVector = PreCompute,
# AbstractSlopeMethod = OnTheFly) via trait-dispatch helper. Both strategies
# use the same `_hermite_integral_kernel_1d`; only the slope source differs.

@inline function integrate(
        itp::AbstractHermiteInterpolant1D{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg, Tv}
    return _integrate_hermite_1d(itp.dy, itp, x0, x1, search, hint)
end

# ── PreCompute path: dy is a precomputed slope vector (unchanged behavior) ──
@inline function _integrate_hermite_1d(
        dy::AbstractVector,
        itp::AbstractHermiteInterpolant1D{Tg, Tv},
        x0::Real, x1::Real,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}},
    ) where {Tg, Tv}
    x = itp.x
    y = itp.y
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(x, x0, search, hint)

    # Axis-as-truth: `x` is the wrapped axis (`_CachedRange`/`_CachedVector`),
    # so `_get_inv_h(x, i)` reads a cached scalar (Range) or indexed load
    # (Vector) without a hardware division.
    partial = @inline (i, xL, h, a2, b2) -> begin
        inv_h = _get_inv_h(x, i)
        @inbounds _hermite_integral_kernel_1d(
            y[i], y[i + 1], dy[i], dy[i + 1],
            h, inv_h, a2 - xL, b2 - xL,
        )
    end
    full = @inline (i, h) -> begin
        inv_h = _get_inv_h(x, i)
        @inbounds _hermite_integral_kernel_1d(
            y[i], y[i + 1], dy[i], dy[i + 1],
            h, inv_h, zero(Tg), h,
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── OnTheFly path: dy is a slope method sentinel, slopes computed per cell ──
# Uses a dedicated sliding-window helper (`_integrate_hermite_onthefly_inner`)
# as the `in_domain` callback. Each call from the extrap dispatcher gets a
# fresh sliding-window pass, so WrapExtrap's multiple sub-interval calls work
# automatically without any state leakage.
# Memory overhead: 2 scalars (dy_L, dy_R) on the stack — zero heap, zero pool,
# regardless of grid size. This preserves the OnTheFly "memory O(1)" contract.
@inline function _integrate_hermite_1d(
        sm::AbstractSlopeMethod,
        itp::AbstractHermiteInterpolant1D{Tg, Tv},
        x0::Real, x1::Real,
        search::AbstractSearchPolicy,
        hint::Union{Nothing, Base.RefValue{Int}},
    ) where {Tg, Tv}
    x = itp.x
    y = itp.y
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(x, x0, search, hint)

    in_domain = @inline (a, b) -> _integrate_hermite_onthefly_inner(
        sm, x, y, a, b, searcher, Tout
    )

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── Sliding-window inner helper for OnTheFly bounded integrate ──
# Walks cells [i0..i1] in order, threading two slope scalars (dy_L, dy_R)
# through the loop. Each cell reuses the previous cell's right slope as its
# own left slope, so `_local_slope` is called exactly `(i1 - i0 + 2)` times
# total — one for i0, one for each subsequent cell's right edge.
#
# Axis-as-truth: `x` is the wrapped axis carrying cached h/inv_h via
# `_get_h(x, idx)` / `_get_inv_h(x, idx)` — no separate spacing arg.
@inline function _integrate_hermite_onthefly_inner(
        sm::AbstractSlopeMethod,
        x::AbstractVector, y::AbstractVector,
        a::Real, b::Real, searcher::Searcher, ::Type{Tout},
    ) where {Tout}
    sign, lo, hi = _normalize_bounds_1d(a, b)
    sign == 0 && return zero(Tout)
    n = length(x)

    i0, i0_R, xL0, _ = search_interval(searcher, x, lo)
    i1, i1_R, xL1, _ = search_interval(searcher, x, hi)

    dy_L = _local_slope(sm, x, y, i0, n)
    dy_R = _local_slope(sm, x, y, i0_R, n)

    # Single-cell fast path — both bounds fall in the same cell.
    if i0 == i1
        h = _get_h(x, i0)
        inv_h = _get_inv_h(x, i0)
        return sign * _hermite_integral_kernel_1d(
            @inbounds(y[i0]), @inbounds(y[i0_R]), dy_L, dy_R,
            h, inv_h, lo - xL0, hi - xL0,
        )
    end

    # First partial cell [lo, xL0 + h0]
    h0 = _get_h(x, i0)
    inv_h0 = _get_inv_h(x, i0)
    total = _hermite_integral_kernel_1d(
        @inbounds(y[i0]), @inbounds(y[i0_R]), dy_L, dy_R,
        h0, inv_h0, lo - xL0, h0,
    )

    # Interior full cells — sliding window
    @inbounds for i in (i0 + 1):(i1 - 1)
        dy_L = dy_R                                    # roll: old right becomes new left
        dy_R = _local_slope(sm, x, y, i + 1, n)        # only right side is fresh
        h = _get_h(x, i)
        inv_h = _get_inv_h(x, i)
        total += _hermite_integral_kernel_1d(
            y[i], y[i + 1], dy_L, dy_R,
            h, inv_h, zero(eltype(x)), h,
        )
    end

    # Final partial cell [xL1, hi]
    dy_L = dy_R
    dy_R = _local_slope(sm, x, y, i1_R, n)
    h1 = _get_h(x, i1)
    inv_h1 = _get_inv_h(x, i1)
    total += _hermite_integral_kernel_1d(
        @inbounds(y[i1]), @inbounds(y[i1_R]), dy_L, dy_R,
        h1, inv_h1, zero(eltype(x)), hi - xL1,
    )

    return sign * total
end
