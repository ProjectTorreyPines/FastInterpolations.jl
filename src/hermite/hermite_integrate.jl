# ========================================
# Hermite Family 1D — Bounded Integration (shared)
# ========================================
# Single dispatch for all AbstractLocalCubicInterpolant1D subtypes.
# All store (x, y, dy, spacing) and use _hermite_integral_kernel_1d.

@inline function integrate(
        itp::AbstractLocalCubicInterpolant1D{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing,
    ) where {Tg <: AbstractFloat, Tv}
    x = itp.x
    y = itp.y
    dy = itp.dy
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _resolve_search(x, x0, search, hint)

    partial = @inline (i, xL, h, a2, b2) -> begin
        inv_h = inv(h)
        @inbounds _hermite_integral_kernel_1d(
            y[i], y[i + 1], dy[i], dy[i + 1],
            h, inv_h, a2 - xL, b2 - xL,
        )
    end
    full = @inline (i, h) -> begin
        inv_h = inv(h)
        @inbounds _hermite_integral_kernel_1d(
            y[i], y[i + 1], dy[i], dy[i + 1],
            h, inv_h, zero(Tg), h,
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, _spacing(itp), a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end
