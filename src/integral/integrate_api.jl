# ── 1D grid accessor (CubicInterpolant/CubicSeriesInterpolant store x in cache) ──
@inline _grid_1d(itp::CubicInterpolant) = itp.cache.x
@inline _grid_1d(itp::CubicSeriesInterpolant) = itp.cache.x
@inline _grid_1d(itp::AbstractInterpolant) = itp.x

# ── 1D spacing accessor ──
# Returns an object that responds to `_get_h(., i)` / `_get_inv_h(., i)` 2-arg
# form. Migrated families (Linear, Constant, Cubic) use the wrapped axis itself
# (`_CachedRange`/`_CachedVector`/`_ExclusivePeriodicAxis`) as the source of
# truth for h/inv_h. Pre-migration families (Quadratic, Hermite-family) still
# carry a separate `spacing::S` field.
@inline _spacing(itp::LinearInterpolant) = itp.x
@inline _spacing(itp::ConstantInterpolant) = itp.x
@inline _spacing(itp::CubicInterpolant) = itp.cache.x
@inline _spacing(sitp::CubicSeriesInterpolant) = sitp.cache.x
# Pre-migration default: families that still carry a `spacing::S` field.
@inline _spacing(itp::AbstractInterpolant1D) = itp.spacing
@inline _spacing(sitp::AbstractSeriesInterpolant) = sitp.spacing

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
    ) where {Tg <: AbstractFloat, Tv}
    x = itp.cache.x
    y = itp.y
    z = itp.z
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
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

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, _spacing(itp), a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── LinearInterpolant 1D ──

@inline function integrate(
        itp::LinearInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    x = itp.x
    y = itp.y
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
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

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, _spacing(itp), a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── QuadraticInterpolant 1D ──

@inline function integrate(
        itp::QuadraticInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    x = itp.x
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
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

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, _spacing(itp), a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, itp.y[1], itp.y[end], x0, x1, Tout)
end

# ── ConstantInterpolant 1D ──

@inline function integrate(
        itp::ConstantInterpolant{Tg, Tv},
        x0::Real, x1::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _resolve_search(itp.x, x0, search, hint)
    return _integrate_constant_1d_impl(itp.x, _spacing(itp), itp.y, itp.side, itp.extrap, x0, x1, searcher, Tg, Tout)
end

# Receives concrete side (parametric from struct) + extrap mode.
# Uses the generic _integrate_1d_cellwise path — side is already concrete here.
@inline function _integrate_constant_1d_impl(
        x::AbstractVector, spacing::Union{AbstractGridSpacing, AbstractVector}, y::AbstractVector, side::AbstractSide, extrap::AbstractExtrap,
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
    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, spacing, a, b, searcher, partial, full, Tout)
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
    ) where {Tg <: AbstractFloat, Tv}
    x = sitp.cache.x
    y = sitp.y
    z = sitp.z
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
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
        in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, _spacing(sitp), a, b, searcher, partial, full, Tout)
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
    ) where {Tg <: AbstractFloat, Tv}
    x = sitp.x
    y = sitp.y
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
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
        in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, _spacing(sitp), a, b, searcher, partial, full, Tout)
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
    ) where {Tg <: AbstractFloat, Tv}
    x = sitp.x
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
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
        in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, _spacing(sitp), a, b, searcher, partial, full, Tout)
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
    ) where {Tg <: AbstractFloat, Tv}
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _resolve_search(sitp.x, x0, search, hint)
    return _integrate_constant_series_1d(sitp.x, _spacing(sitp), sitp.y, sitp.side, sitp.extrap, x0, x1, searcher, Tg, Tout)
end

@inline function _integrate_constant_series_1d(
        x::AbstractVector, spacing::Union{AbstractGridSpacing, AbstractVector}, y::AbstractMatrix, side::AbstractSide, extrap::AbstractExtrap,
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
        in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, spacing, a, b, searcher, partial, full, Tout)
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
    Tout = promote_type(Tv, Tg, map(typeof, lo2)..., map(typeof, hi2)...)
    return isconcretetype(Tout) ? Tout : Tv
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
        itp.grids, itp.spacings, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    _zero = Tout <: Number ? zero(Tout) : 0 * itp.nodal_derivs.partials[1]
    sign == 0 && return _zero

    total = _zero
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, itp.spacings, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.spacings[d], idx[d])), Val(N))
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
        itp.grids, itp.spacings, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    _zero = Tout <: Number ? zero(Tout) : 0 * itp.data[1]
    sign == 0 && return _zero

    total = _zero
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, itp.spacings, lo2, hi2, I, Val(N))
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
        itp.grids, itp.spacings, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    _zero = Tout <: Number ? zero(Tout) : 0 * itp.nodal_derivs.partials[1]
    sign == 0 && return _zero

    total = _zero
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, itp.spacings, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.spacings[d], idx[d])), Val(N))
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
        itp.grids, itp.spacings, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    _zero = Tout <: Number ? zero(Tout) : 0 * itp.data[1]
    sign == 0 && return _zero

    total = _zero
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, itp.spacings, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            total += convert(Tout, _integrate_constant_nd_cell(itp.data, idx, hs, ulos, uhis, itp.sides))
        end
    end
    return sign * total
end
