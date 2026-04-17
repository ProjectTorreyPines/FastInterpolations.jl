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
    spacing = _spacing(itp)
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _resolve_search(x, x0, search, hint)

    # `_get_inv_h(spacing, i)` reads a cached field (ScalarSpacing) or an
    # indexed load (VectorSpacing) — always cheaper than `inv(h)` which emits
    # a hardware division. The spacing is already in scope for the cellwise
    # walker, so capturing it in the closure is free.
    partial = @inline (i, xL, h, a2, b2) -> begin
        inv_h = _get_inv_h(spacing, i)
        @inbounds _hermite_integral_kernel_1d(
            y[i], y[i + 1], dy[i], dy[i + 1],
            h, inv_h, a2 - xL, b2 - xL,
        )
    end
    full = @inline (i, h) -> begin
        inv_h = _get_inv_h(spacing, i)
        @inbounds _hermite_integral_kernel_1d(
            y[i], y[i + 1], dy[i], dy[i + 1],
            h, inv_h, zero(Tg), h,
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, spacing, a, b, searcher, partial, full, Tout)

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
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _resolve_search(x, x0, search, hint)
    spacing = _spacing(itp)

    in_domain = @inline (a, b) -> _integrate_hermite_onthefly_inner(
        sm, x, y, spacing, a, b, searcher, Tout
    )

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── Sliding-window inner helper for OnTheFly bounded integrate ──
# Walks cells [i0..i1] in order, threading two slope scalars (dy_L, dy_R)
# through the loop. Each cell reuses the previous cell's right slope as its
# own left slope, so `_local_slope` is called exactly `(i1 - i0 + 2)` times
# total — one for i0, one for each subsequent cell's right edge.
#
# This is the OnTheFly analogue of `_integrate_1d_cellwise` with the inline
# closure pattern, but with state threaded through local variables instead
# of captured state. The duplication is limited to the cell-location logic
# (~10 lines); the numerical kernel is `_hermite_integral_kernel_1d`, shared
# with the PreCompute path above.
@inline function _integrate_hermite_onthefly_inner(
        sm::AbstractSlopeMethod,
        x::AbstractVector, y::AbstractVector, spacing::AbstractGridSpacing,
        a::Real, b::Real, searcher::Searcher, ::Type{Tout},
    ) where {Tout}
    sign, lo, hi = _normalize_bounds_1d(a, b)
    sign == 0 && return zero(Tout)
    n = length(x)

    i0, i0_R, xL0, _ = search_interval(searcher, x, spacing, lo)
    i1, i1_R, xL1, _ = search_interval(searcher, x, spacing, hi)

    dy_L = _local_slope(sm, x, y, i0, n)
    dy_R = _local_slope(sm, x, y, i0_R, n)

    # Single-cell fast path — both bounds fall in the same cell.
    # `_get_inv_h(spacing, i)` reads a cached field (ScalarSpacing) or an
    # indexed load (VectorSpacing), avoiding a hardware division per cell.
    if i0 == i1
        h = _get_h(spacing, i0)
        inv_h = _get_inv_h(spacing, i0)
        return sign * _hermite_integral_kernel_1d(
            @inbounds(y[i0]), @inbounds(y[i0_R]), dy_L, dy_R,
            h, inv_h, lo - xL0, hi - xL0,
        )
    end

    # First partial cell [lo, xL0 + h0]
    h0 = _get_h(spacing, i0)
    inv_h0 = _get_inv_h(spacing, i0)
    total = _hermite_integral_kernel_1d(
        @inbounds(y[i0]), @inbounds(y[i0_R]), dy_L, dy_R,
        h0, inv_h0, lo - xL0, h0,
    )

    # Interior full cells — sliding window
    @inbounds for i in (i0 + 1):(i1 - 1)
        dy_L = dy_R                                    # roll: old right becomes new left
        dy_R = _local_slope(sm, x, y, i + 1, n)        # only right side is fresh
        h = _get_h(spacing, i)
        inv_h = _get_inv_h(spacing, i)
        total += _hermite_integral_kernel_1d(
            y[i], y[i + 1], dy_L, dy_R,
            h, inv_h, zero(eltype(x)), h,
        )
    end

    # Final partial cell [xL1, hi]
    dy_L = dy_R
    dy_R = _local_slope(sm, x, y, i1_R, n)
    h1 = _get_h(spacing, i1)
    inv_h1 = _get_inv_h(spacing, i1)
    total += _hermite_integral_kernel_1d(
        @inbounds(y[i1]), @inbounds(y[i1_R]), dy_L, dy_R,
        h1, inv_h1, zero(eltype(x)), hi - xL1,
    )

    return sign * total
end
