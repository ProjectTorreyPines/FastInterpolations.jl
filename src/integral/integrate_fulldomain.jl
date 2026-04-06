# ═══════════════════════════════════════════════════════════════
# Full-domain fast path + cumulative integration
#
# When integrating over the entire domain, ALL cells are full cells:
# no search, no partial-cell handling, no bound normalization.
# This file defines:
#   - _full_cell_fn(itp, ...) trait: per-type closure for one cell's integral
#   - integrate(itp)          fast path: scalar accumulator (1D, Series, ND)
#   - cumulative_integrate    prefix-sum of per-cell integrals (1D, Series)
# ═══════════════════════════════════════════════════════════════

# ── 1D scalar trait: _full_cell_fn(itp) → (i, h) -> value ──

@inline function _full_cell_fn(itp::CubicInterpolant)
    y, z = itp.y, itp.z
    return @inline (i, h) -> @inbounds _cubic_integral_kernel(_EvalIntegralCell(), z[i], z[i + 1], y[i], y[i + 1], h)
end

@inline function _full_cell_fn(itp::LinearInterpolant)
    y = itp.y
    return @inline (i, h) -> @inbounds _linear_integral_kernel(_EvalIntegralCell(), y[i], y[i + 1], h)
end

@inline function _full_cell_fn(itp::QuadraticInterpolant)
    a, d, y = itp.a, itp.d, itp.y
    return @inline (i, h) -> @inbounds _quadratic_integral_kernel(_EvalIntegralCell(), a[i], d[i], y[i], h)
end

@inline function _full_cell_fn(itp::AbstractHermiteInterpolant1D)
    y, dy = itp.y, itp.dy
    return @inline (i, h) -> begin
        inv_h = inv(h)
        @inbounds _hermite_integral_kernel_1d(y[i], y[i + 1], dy[i], dy[i + 1], h, inv_h, zero(h), h)
    end
end

@inline function _full_cell_fn(itp::ConstantInterpolant{Tg}, side::AbstractSide) where {Tg}
    y = itp.y
    return @inline (i, h) -> @inbounds _constant_integral_kernel(_EvalIntegralPartial(), y[i], y[i + 1], h, zero(Tg), h, side)
end

# ── 1D series trait: _full_cell_fn(sitp, k) → (i, h) -> value ──

@inline function _full_cell_fn(sitp::CubicSeriesInterpolant, k::Int)
    y, z = sitp.y, sitp.z
    return @inline (i, h) -> @inbounds _cubic_integral_kernel(_EvalIntegralCell(), z[i, k], z[i + 1, k], y[i, k], y[i + 1, k], h)
end

@inline function _full_cell_fn(sitp::LinearSeriesInterpolant, k::Int)
    y = sitp.y
    return @inline (i, h) -> @inbounds _linear_integral_kernel(_EvalIntegralCell(), y[i, k], y[i + 1, k], h)
end

@inline function _full_cell_fn(sitp::QuadraticSeriesInterpolant, k::Int)
    a, d, y = sitp.a, sitp.d, sitp.y
    return @inline (i, h) -> @inbounds _quadratic_integral_kernel(_EvalIntegralCell(), a[i, k], d[i, k], y[i, k], h)
end

@inline function _full_cell_fn(sitp::ConstantSeriesInterpolant{Tg}, k::Int, side::AbstractSide) where {Tg}
    y = sitp.y
    return @inline (i, h) -> @inbounds _constant_integral_kernel(_EvalIntegralPartial(), y[i, k], y[i + 1, k], h, zero(Tg), h, side)
end

# ═══════════════════════════════════════════════════════════════
# integrate(itp) — 1D full-domain fast path
# ═══════════════════════════════════════════════════════════════

# Generic 1D: catches Cubic, Linear, Quadratic (not Constant — see override below)
@inline function integrate(
        itp::AbstractInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: AbstractFloat, Tv}
    # Guard: OnTheFly Hermite interpolants don't support integrate yet
    if itp isa AbstractHermiteInterpolant1D && itp.dy isa AbstractSlopeMethod
        _throw_onthefly_unsupported("integrate")
    end
    x = _grid_1d(itp)
    Tout = promote_type(Tv, Tg)
    return _integrate_1d_fulldomain(x, _spacing(itp), _full_cell_fn(itp), Tout)
end

# Constant override: side is parametric → compiler knows concrete type
@inline function integrate(
        itp::ConstantInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: AbstractFloat, Tv}
    x = itp.x
    Tout = promote_type(Tv, Tg)
    return _integrate_1d_fulldomain(x, _spacing(itp), _full_cell_fn(itp, itp.side), Tout)
end

# ═══════════════════════════════════════════════════════════════
# integrate(sitp) — 1D Series full-domain fast path
# ═══════════════════════════════════════════════════════════════

# Generic Series: catches Cubic, Linear, Quadratic series
@inline function integrate(
        sitp::AbstractSeriesInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(sitp)
    Tout = promote_type(Tv, Tg)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        results[k] = _integrate_1d_fulldomain(x, _spacing(sitp), _full_cell_fn(sitp, k), Tout)
    end
    return results
end

# Constant Series override: side is parametric → compiler knows concrete type
@inline function integrate(
        sitp::ConstantSeriesInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: AbstractFloat, Tv}
    x = sitp.x
    Tout = promote_type(Tv, Tg)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        results[k] = _integrate_1d_fulldomain(x, _spacing(sitp), _full_cell_fn(sitp, k, sitp.side), Tout)
    end
    return results
end

# ═══════════════════════════════════════════════════════════════
# integrate(itp) — ND full-domain fast path
# ═══════════════════════════════════════════════════════════════

# ND per-type trait: _full_cell_integral_nd(itp, idx, hs, inv_hs)
# Each calls the existing ND kernel with ulos = zeros, uhis = hs.

@inline function _full_cell_integral_nd(
        itp::CubicInterpolantND{Tg, Tv, N}, idx, hs, inv_hs
    ) where {Tg, Tv, N}
    ulos = ntuple(d -> zero(Tg), Val(N))
    return _integrate_nd_cubic_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, hs)
end

@inline function _full_cell_integral_nd(
        itp::LinearInterpolantND{Tg, Tv, N}, idx, hs, inv_hs
    ) where {Tg, Tv, N}
    ulos = ntuple(d -> zero(Tg), Val(N))
    return _integrate_linear_nd_cell(itp.data, idx, hs, ulos, hs)
end

@inline function _full_cell_integral_nd(
        itp::QuadraticInterpolantND{Tg, Tv, N}, idx, hs, inv_hs
    ) where {Tg, Tv, N}
    ulos = ntuple(d -> zero(Tg), Val(N))
    return _integrate_nd_quad_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, hs)
end

@inline function _full_cell_integral_nd(
        itp::ConstantInterpolantND{Tg, Tv, N}, idx, hs, inv_hs
    ) where {Tg, Tv, N}
    ulos = ntuple(d -> zero(Tg), Val(N))
    return _integrate_constant_nd_cell(itp.data, idx, hs, ulos, hs, itp.sides)
end

# Sample Tv value for duck-typing safe zero initialization
@inline _nd_sample_value(itp::LinearInterpolantND) = @inbounds itp.data[1]
@inline _nd_sample_value(itp::ConstantInterpolantND) = @inbounds itp.data[1]
@inline _nd_sample_value(itp::CubicInterpolantND) = @inbounds itp.nodal_derivs.partials[1]
@inline _nd_sample_value(itp::QuadraticInterpolantND) = @inbounds itp.nodal_derivs.partials[1]

# Generic ND full-domain: catches all ND types
@inline function integrate(
        itp::AbstractInterpolantND{Tg, Tv, N};
        search = nothing,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    Tout = promote_type(Tv, Tg)
    total = Tout <: Number ? zero(Tout) : 0 * _nd_sample_value(itp)
    cell_ranges = ntuple(d -> 1:(length(itp.grids[d]) - 1), Val(N))
    for I in CartesianIndices(cell_ranges)
        idx = ntuple(d -> I[d], Val(N))
        hs = ntuple(d -> @inbounds(_get_h(itp.spacings[d], idx[d])), Val(N))
        inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.spacings[d], idx[d])), Val(N))
        total += _full_cell_integral_nd(itp, idx, hs, inv_hs)
    end
    return total
end

# ═══════════════════════════════════════════════════════════════
# cumulative_integrate — 1D prefix-sum of per-cell integrals
# ═══════════════════════════════════════════════════════════════

# Generic 1D: catches Cubic, Linear, Quadratic
function cumulative_integrate(
        itp::AbstractInterpolant{Tg, Tv}
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(itp)
    Tout = promote_type(Tv, Tg)
    return _cumulative_integrate_1d(x, _spacing(itp), _full_cell_fn(itp), Tout)
end

# Constant override: side is parametric → compiler knows concrete type
function cumulative_integrate(
        itp::ConstantInterpolant{Tg, Tv}
    ) where {Tg <: AbstractFloat, Tv}
    x = itp.x
    Tout = promote_type(Tv, Tg)
    return _cumulative_integrate_1d(x, _spacing(itp), _full_cell_fn(itp, itp.side), Tout)
end

# Generic Series: catches Cubic, Linear, Quadratic series
function cumulative_integrate(
        sitp::AbstractSeriesInterpolant{Tg, Tv}
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(sitp)
    Tout = promote_type(Tv, Tg)
    n_pts = length(x)
    n_ser = n_series(sitp)
    result = Matrix{Tout}(undef, n_pts, n_ser)
    @inbounds for k in 1:n_ser
        _cumulative_integrate_1d!(@view(result[:, k]), x, _spacing(sitp), _full_cell_fn(sitp, k))
    end
    return result
end

# Constant Series override: side is parametric → compiler knows concrete type
function cumulative_integrate(
        sitp::ConstantSeriesInterpolant{Tg, Tv}
    ) where {Tg <: AbstractFloat, Tv}
    x = sitp.x
    Tout = promote_type(Tv, Tg)
    n_pts = length(x)
    n_ser = n_series(sitp)
    result = Matrix{Tout}(undef, n_pts, n_ser)
    @inbounds for k in 1:n_ser
        _cumulative_integrate_1d!(@view(result[:, k]), x, _spacing(sitp), _full_cell_fn(sitp, k, sitp.side))
    end
    return result
end

# ND override: more specific than AbstractInterpolant, throws clear error
function cumulative_integrate(itp::AbstractInterpolantND)
    throw(
        ArgumentError(
            "cumulative_integrate is not supported for $(typeof(itp)). " *
                "Only 1D interpolants and 1D series interpolants are supported."
        )
    )
end
