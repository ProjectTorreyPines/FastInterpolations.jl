# ── 1D grid accessor (CubicInterpolant/CubicSeriesInterpolant store x in cache) ──
@inline _grid_1d(itp::CubicInterpolant) = itp.cache.x
@inline _grid_1d(itp::CubicSeriesInterpolant) = itp.cache.x
@inline _grid_1d(itp::AbstractInterpolant) = itp.x

# ── One-shot quadrature: integrate(x, y; method) ──
# Full-domain integral of the `method` interpolant of `(x, y)`, routed like
# `interp(x, y; method=…)` (same method tags; bc/side/tension forwarded). Built
# with StorePolicy(copy=false, cache_axis=false) — nothing copied, no axis caches.
@inline function integrate(
        x::AbstractVector,
        y::AbstractVector;
        method::AbstractInterpMethod,
    )
    fn, _, opts = _interp1d_route(method)
    itp = fn(x, y; opts..., store = StorePolicy(copy = false, cache_axis = false))
    return integrate(itp)
end

# ── One-shot quadrature (ND): integrate(grids, data; method) ──
# ND mirror of the 1-D form, but with a smaller supported set: ND full-domain
# integration exists only for the specialized tensor-product types, whereas 1-D
# integrates every method. A single `method` applied to every axis builds a
# homogeneous specialized ND interpolant, then integrates it. Trivial methods
# (linear/constant) use raw reference storage (0-copy, no axis caches); PreCompute
# methods (cubic/quadratic) own their coefficient arrays, so `store` stays at the
# ctor default. Hermite-family methods are rejected up front (see below).
@inline function integrate(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        method::AbstractInterpMethod,
    ) where {N}
    _nd_integrable_method(method) || _throw_nd_oneshot_unsupported(method)
    fn, _, opts = _interp1d_route(method)
    itp = _is_trivial_method(method) ?
        fn(grids, data; opts..., store = StorePolicy(copy = false, cache_axis = false)) :
        fn(grids, data; opts...)
    return integrate(itp)
end

# Which methods have a specialized ND type with a full-domain integral. Hermite
# axes collapse to a HeteroInterpolantND (no ND integral), so they are excluded —
# the one behavioral split from the 1-D one-shot, which integrates all methods.
@inline _nd_integrable_method(::Union{LinearInterp, CubicInterp, QuadraticInterp, ConstantInterp}) = true
@inline _nd_integrable_method(::AbstractInterpMethod) = false

@noinline function _throw_nd_oneshot_unsupported(method)
    throw(
        ArgumentError(
            "integrate(grids, data; method=$(nameof(typeof(method)))(…)) — ND full-domain " *
                "integration is implemented only for LinearInterp, CubicInterp, " *
                "QuadraticInterp, and ConstantInterp. Hermite-family methods " *
                "(Pchip/Akima/Cardinal) have no ND integral yet; integrate axis-by-axis " *
                "on per-fiber 1-D interpolants instead."
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

# ── Fallback stub (bounded ND) ──
function integrate(
        itp::AbstractInterpolantND{Tg, Tv, N},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}};
        search = nothing,
        hint = nothing
    ) where {Tg, Tv, N}
    throw(ArgumentError("integrate(itp_nd, lo, hi) is not implemented for $(typeof(itp)) yet"))
end

@inline function _integrate_nd_output_type(
        ::Type{Tv}, ::Type{Tg},
        lo2::Tuple{Vararg{Any, N}},
        hi2::Tuple{Vararg{Any, N}}
    ) where {Tv, Tg, N}
    Tspan = promote_type(map(typeof, lo2)..., map(typeof, hi2)...)
    return _promote_eltype(_integrate_op, Tg, Tv, Tspan)
end

# ── CubicInterpolantND bounded ──

@inline function integrate(
        itp::CubicInterpolantND{Tg, Tv, N},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}};
        search = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    _zero = Tout <: Number ? zero(Tout) : 0 * itp.nodal_derivs.partials[1]
    sign == 0 && return _zero

    total = _zero
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.grids[d], idx[d])), Val(N))
            total += convert(Tout, _integrate_nd_cubic_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, uhis))
        end
    end
    return sign * total
end

# ═══════════════════════════════════════════════════════════════
# ND Linear Integration
# ═══════════════════════════════════════════════════════════════

@inline function integrate(
        itp::LinearInterpolantND{Tg, Tv, N},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}};
        search = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    _zero = Tout <: Number ? zero(Tout) : 0 * itp.data[1]
    sign == 0 && return _zero

    total = _zero
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            total += convert(Tout, _integrate_linear_nd_cell(itp.data, idx, hs, ulos, uhis))
        end
    end
    return sign * total
end

# ═══════════════════════════════════════════════════════════════
# ND Quadratic Integration
# ═══════════════════════════════════════════════════════════════

@inline function integrate(
        itp::QuadraticInterpolantND{Tg, Tv, N},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}};
        search = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    _zero = Tout <: Number ? zero(Tout) : 0 * itp.nodal_derivs.partials[1]
    sign == 0 && return _zero

    total = _zero
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.grids[d], idx[d])), Val(N))
            total += convert(Tout, _integrate_nd_quad_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, uhis))
        end
    end
    return sign * total
end

# ═══════════════════════════════════════════════════════════════
# ND Constant Integration
# ═══════════════════════════════════════════════════════════════

@inline function integrate(
        itp::ConstantInterpolantND{Tg, Tv, N},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}};
        search = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    _zero = Tout <: Number ? zero(Tout) : 0 * itp.data[1]
    sign == 0 && return _zero

    total = _zero
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            total += convert(Tout, _integrate_constant_nd_cell(itp.data, idx, hs, ulos, uhis, itp.sides))
        end
    end
    return sign * total
end
