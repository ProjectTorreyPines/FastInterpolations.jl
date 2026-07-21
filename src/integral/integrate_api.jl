# ── 1D grid accessor (CubicInterpolant/CubicSeriesInterpolant store x in cache) ──
@inline _grid_1d(itp::CubicInterpolant) = itp.cache.x
@inline _grid_1d(itp::CubicSeriesInterpolant) = itp.cache.x
@inline _grid_1d(itp::AbstractInterpolant) = itp.x

# ── One-shot quadrature: integrate(x, y[, a, b]; method) ──
# Build the `method` interpolant of `(x, y)` with raw reference storage (nothing
# copied) and integrate it — full-domain, or over `[a, b]` when bounds are given.
@inline function _oneshot_build_1d(method::AbstractInterpMethod, x, y)
    fn, _, opts = _interp1d_route(method)
    return fn(x, y; opts..., store = StorePolicy(copy = false, cache_axis = false))
end
@inline integrate(x::AbstractVector, y::AbstractVector; method::AbstractInterpMethod) =
    integrate(_oneshot_build_1d(method, x, y))
@inline integrate(x::AbstractVector, y::AbstractVector, a::Real, b::Real; method::AbstractInterpMethod) =
    integrate(_oneshot_build_1d(method, x, y), a, b)

# ── One-shot cumulative: cumulative_integrate(x, y; method) ──
# Running-integral sibling of one-shot `integrate` (same raw-storage build);
# `out[end]` == `integrate(x, y; method)`. 1-D only — ND is not defined here.
@inline cumulative_integrate(x::AbstractVector, y::AbstractVector; method::AbstractInterpMethod) =
    cumulative_integrate(_oneshot_build_1d(method, x, y))

# ── One-shot quadrature (ND): integrate(grids, data[, lo, hi]; method) ──
# ND mirror — only tensor-product types integrate (Linear/Cubic/Quadratic/
# Constant); the local-Hermite family (Pchip/Akima/Cardinal) has no homogeneous
# ND factory reachable from a single `method`, so it is rejected up front.
# Trivial methods use raw storage; PreCompute types keep the ctor default.
@inline function _oneshot_build_nd(method::AbstractInterpMethod, grids, data)
    _nd_integrable_method(method) || _throw_nd_oneshot_unsupported(method)
    fn, _, opts = _interp1d_route(method)
    return _is_trivial_method(method) ?
        fn(grids, data; opts..., store = StorePolicy(copy = false, cache_axis = false)) :
        fn(grids, data; opts...)
end
# A per-axis method tuple can't be built by the single-method one-shot path;
# point at the persistent build instead of a bare kwarg MethodError.
@noinline _oneshot_build_nd(method::Tuple, grids, data) = _throw_nd_oneshot_tuple(method)

@inline integrate(grids::NTuple{N, AbstractVector}, data::AbstractArray{<:Any, N}; method) where {N} =
    integrate(_oneshot_build_nd(method, grids, data))
@inline integrate(
    grids::NTuple{N, AbstractVector}, data::AbstractArray{<:Any, N},
    lo::NTuple{N, Real}, hi::NTuple{N, Real}; method
) where {N} = integrate(_oneshot_build_nd(method, grids, data), lo, hi)

# Single source of truth for "has an ND separable integral": the type-keyed
# predicate the engine uses (`_is_separable_method_type`, integrate_fulldomain.jl),
# so the supported-family list lives in exactly one place.
@inline _nd_integrable_method(m::AbstractInterpMethod) = _is_separable_method_type(typeof(m))

@noinline function _throw_nd_oneshot_unsupported(method)
    throw(
        ArgumentError(
            "integrate(grids, data[, lo, hi]; method=$(nameof(typeof(method)))(…)) — ND " *
                "integration is implemented only for LinearInterp, CubicInterp, " *
                "QuadraticInterp, and ConstantInterp. Hermite-family methods " *
                "(Pchip/Akima/Cardinal) have no ND integral yet; integrate axis-by-axis " *
                "on per-fiber 1-D interpolants instead."
        )
    )
end

@noinline function _throw_nd_oneshot_tuple(method)
    throw(
        ArgumentError(
            "one-shot ND `integrate(grids, data; method=…)` takes a single tensor-product " *
                "method, not a per-axis tuple ($(method)). Build the interpolant first and " *
                "integrate it: `integrate(interp(grids, data; method=$(method)))` " *
                "(add `coeffs=PreCompute()` for Cubic/Quadratic axes)."
        )
    )
end


# ── Fallback stub (bounded 1D) ──
function integrate(itp::AbstractInterpolant, x0::Real, x1::Real; search = nothing, hint = nothing)
    throw(ArgumentError("integrate(itp, x0, x1) is not implemented for $(typeof(itp)) yet"))
end

# ── CubicInterpolant 1D ──

@inline function integrate(
        itp::CubicInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tv}
    x = itp.cache.x
    y = itp.y
    z = itp.z
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(x, x0, search, hint)

    partial = @inline (i, xL, h, a2, b2) -> begin
        @inbounds _cubic_integral_kernel(
            _EvalIntegralPartial(), z[i], z[i + 1], y[i], y[i + 1], h, a2 - xL, b2 - xL
        )
    end
    full = @inline (i, h) -> begin
        @inbounds _cubic_integral_kernel(
            _EvalIntegralCell(), z[i], z[i + 1], y[i], y[i + 1], h
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── LinearInterpolant 1D ──

@inline function integrate(
        itp::LinearInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tv}
    x = itp.x
    y = itp.y
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(x, x0, search, hint)

    partial = @inline (i, xL, h, a2, b2) -> begin
        @inbounds _linear_integral_kernel(
            _EvalIntegralPartial(), y[i], y[i + 1], h, a2 - xL, b2 - xL
        )
    end
    full = @inline (i, h) -> begin
        @inbounds _linear_integral_kernel(
            _EvalIntegralCell(), y[i], y[i + 1], h
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── QuadraticInterpolant 1D ──

@inline function integrate(
        itp::QuadraticInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tv}
    x = itp.x
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(x, x0, search, hint)

    partial = @inline (i, xL, h, a2, b2) -> begin
        @inbounds _quadratic_integral_kernel(
            _EvalIntegralPartial(), itp.a[i], itp.d[i], itp.y[i], a2 - xL, b2 - xL
        )
    end
    full = @inline (i, h) -> begin
        @inbounds _quadratic_integral_kernel(
            _EvalIntegralCell(), itp.a[i], itp.d[i], itp.y[i], h
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, itp.y[1], itp.y[end], x0, x1, Tout)
end

# ── ConstantInterpolant 1D ──

@inline function integrate(
        itp::ConstantInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tv}
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(itp.x, x0, search, hint)
    return _integrate_constant_1d_impl(itp.x, itp.y, itp.side, itp.extrap, x0, x1, searcher, Tg, Tout)
end

# Receives concrete side (parametric from struct) + extrap mode.
# Uses the generic _integrate_1d_cellwise path — side is already concrete here.
@inline function _integrate_constant_1d_impl(
        x::AbstractVector, y::AbstractVector, side::AbstractSide, extrap::AbstractExtrap,
        x0::Real, x1::Real, searcher::SR, ::Type{Tg}, ::Type{Tout}
    ) where {SR <: Searcher, Tg, Tout}
    partial = @inline (i, xL, h, a2, b2) -> begin
        @inbounds _constant_integral_kernel(
            _EvalIntegralPartial(), y[i], y[i + 1], h, a2 - xL, b2 - xL, side
        )
    end
    full = @inline (i, h) -> begin
        @inbounds _constant_integral_kernel(
            _EvalIntegralPartial(), y[i], y[i + 1], h, zero(Tg), h, side
        )
    end
    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)
    y_left = @inbounds y[1]
    y_right = @inbounds y[end]
    return _dispatch_extrap_integrate_1d(extrap, in_domain, x, y_left, y_right, x0, x1, Tout)
end

# ═══════════════════════════════════════════════════════════════
# 1D Series Integration
# ═══════════════════════════════════════════════════════════════

# ── CubicSeriesInterpolant 1D ──

@inline function integrate(
        sitp::CubicSeriesInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tv}
    x = sitp.cache.x
    y = sitp.y
    z = sitp.z
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(x, x0, search, hint)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        partial = @inline (i, xL, h, a2, b2) -> _cubic_integral_kernel(
            _EvalIntegralPartial(), z[i, k], z[i + 1, k], y[i, k], y[i + 1, k], h, a2 - xL, b2 - xL
        )
        full = @inline (i, h) -> _cubic_integral_kernel(
            _EvalIntegralCell(), z[i, k], z[i + 1, k], y[i, k], y[i + 1, k], h
        )
        in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)
        results[k] = _dispatch_extrap_integrate_1d(sitp.extrap, in_domain, x, y[1, k], y[end, k], x0, x1, Tout)
    end
    return results
end

# ── LinearSeriesInterpolant 1D ──

@inline function integrate(
        sitp::LinearSeriesInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tv}
    x = sitp.x
    y = sitp.y
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(x, x0, search, hint)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        partial = @inline (i, xL, h, a2, b2) -> _linear_integral_kernel(
            _EvalIntegralPartial(), y[i, k], y[i + 1, k], h, a2 - xL, b2 - xL
        )
        full = @inline (i, h) -> _linear_integral_kernel(
            _EvalIntegralCell(), y[i, k], y[i + 1, k], h
        )
        in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)
        results[k] = _dispatch_extrap_integrate_1d(sitp.extrap, in_domain, x, y[1, k], y[end, k], x0, x1, Tout)
    end
    return results
end

# ── QuadraticSeriesInterpolant 1D ──

@inline function integrate(
        sitp::QuadraticSeriesInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tv}
    x = sitp.x
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(x, x0, search, hint)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        partial = @inline (i, xL, h, a2, b2) -> _quadratic_integral_kernel(
            _EvalIntegralPartial(), sitp.a[i, k], sitp.d[i, k], sitp.y[i, k], a2 - xL, b2 - xL
        )
        full = @inline (i, h) -> _quadratic_integral_kernel(
            _EvalIntegralCell(), sitp.a[i, k], sitp.d[i, k], sitp.y[i, k], h
        )
        in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)
        results[k] = _dispatch_extrap_integrate_1d(sitp.extrap, in_domain, x, sitp.y[1, k], sitp.y[end, k], x0, x1, Tout)
    end
    return results
end

# ── ConstantSeriesInterpolant 1D ──

@inline function integrate(
        sitp::ConstantSeriesInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tv}
    Tspan = promote_type(typeof(x0), typeof(x1))
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tspan)
    searcher = _resolve_search(sitp.x, x0, search, hint)
    return _integrate_constant_series_1d(sitp.x, sitp.y, sitp.side, sitp.extrap, x0, x1, searcher, Tg, Tout)
end

@inline function _integrate_constant_series_1d(
        x::AbstractVector, y::AbstractMatrix, side::AbstractSide, extrap::AbstractExtrap,
        x0::Real, x1::Real, searcher::SR, ::Type{Tg}, ::Type{Tout}
    ) where {SR <: Searcher, Tg, Tout}
    n = size(y, 2)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        partial = @inline (i, xL, h, a2, b2) -> _constant_integral_kernel(
            _EvalIntegralPartial(), y[i, k], y[i + 1, k], h, a2 - xL, b2 - xL, side
        )
        full = @inline (i, h) -> _constant_integral_kernel(
            _EvalIntegralPartial(), y[i, k], y[i + 1, k], h, zero(Tg), h, side
        )
        in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)
        results[k] = _dispatch_extrap_integrate_1d(extrap, in_domain, x, y[1, k], y[end, k], x0, x1, Tout)
    end
    return results
end

# ═══════════════════════════════════════════════════════════════
# ND Integration
# ═══════════════════════════════════════════════════════════════
#
# The ND `integrate(itp)` / `integrate(itp, lo, hi)` entry points live in
# integrate_fulldomain.jl — one separable engine for every tensor-product family
# (homogeneous and heterogeneous). Only the shared output-type helper is here.

@inline function _integrate_nd_output_type(
        ::Type{Tv}, ::Type{Tg},
        lo2::Tuple{Vararg{Any, N}},
        hi2::Tuple{Vararg{Any, N}}
    ) where {Tv, Tg, N}
    Tspan = promote_type(map(typeof, lo2)..., map(typeof, hi2)...)
    return _promote_eltype(_integrate_op, Tg, Tv, Tspan)
end
