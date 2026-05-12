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

@inline _full_cell_fn(itp::AbstractHermiteInterpolant1D) =
    _hermite_full_cell_fn(itp, itp.dy)

# PreCompute branch: slopes live in `itp.dy::AbstractVector`, indexed directly.
# `inv_h` is pulled from the wrapped grid's precomputed cache via `_get_inv_h`
# — single field load for `_CachedRange`, indexed load for `_CachedVector` —
# always cheaper than computing `inv(h)` per cell.
@inline function _hermite_full_cell_fn(itp::AbstractHermiteInterpolant1D, dy::AbstractVector)
    x = _grid_1d(itp)
    y = itp.y
    return @inline (i, h) -> begin
        inv_h = _get_inv_h(x, i)
        @inbounds _hermite_integral_kernel_1d(y[i], y[i + 1], dy[i], dy[i + 1], h, inv_h, zero(h), h)
    end
end

# OnTheFly branch: slopes computed locally per call via `_local_slope`.
# Each cell independently computes its two slopes — no sliding-window state,
# so the closure stays immutable and heap-free. The PCHIP/Cardinal/Akima stencils
# are O(1), so the extra recomputation (2(n-1) calls vs n for a sliding window)
# adds only a constant factor to full-domain integration; the code simplification
# removes three dedicated kernels.
@inline function _hermite_full_cell_fn(itp::AbstractHermiteInterpolant1D, sm::AbstractSlopeMethod)
    x, y = _grid_1d(itp), itp.y
    n = length(x)
    return @inline (i, h) -> begin
        dy_L = _local_slope(sm, x, y, i, n)
        dy_R = _local_slope(sm, x, y, i + 1, n)
        inv_h = _get_inv_h(x, i)
        @inbounds _hermite_integral_kernel_1d(y[i], y[i + 1], dy_L, dy_R, h, inv_h, zero(h), h)
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

# Generic 1D: catches any interpolant whose `_full_cell_fn(itp)` trait is the
# single-arg form — Cubic, Linear, Quadratic, and the entire Hermite family
# (PreCompute + OnTheFly via `_full_cell_fn`'s internal dispatch on `itp.dy`).
# Constant has its own override below because its full-cell trait depends on `side`.
@inline function integrate(
        itp::AbstractInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(itp)
    Tout = promote_type(Tv, Tg)
    return _integrate_1d_fulldomain(x, _full_cell_fn(itp), Tout)
end

# Constant override: side is parametric → compiler knows concrete type,
# so `_full_cell_fn(itp, side)` is a distinct method from the generic single-arg form.
@inline function integrate(
        itp::ConstantInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(itp)
    Tout = promote_type(Tv, Tg)
    return _integrate_1d_fulldomain(x, _full_cell_fn(itp, itp.side), Tout)
end

# Hermite family is handled by the generic path above — `_full_cell_fn`
# internally dispatches on `itp.dy` (PreCompute vs OnTheFly).

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
        results[k] = _integrate_1d_fulldomain(x, _full_cell_fn(sitp, k), Tout)
    end
    return results
end

# Constant Series override: side is parametric → compiler knows concrete type
@inline function integrate(
        sitp::ConstantSeriesInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(sitp)
    Tout = promote_type(Tv, Tg)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        results[k] = _integrate_1d_fulldomain(x, _full_cell_fn(sitp, k, sitp.side), Tout)
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

# HeteroInterpolantND: explicit not-implemented error. Beats the MethodError
# the generic dispatch below would hit inside `_full_cell_integral_nd`, and
# tells the user about the actual workaround (per-axis 1D integration, or
# switching to a homogeneous specialized ND type).
function integrate(
        itp::HeteroInterpolantND;
        search = nothing,
        hint = nothing,
    )
    _throw_hetero_nd_integrate_unsupported(itp.methods)
end

# Bounded ND integrate never existed; add a HeteroInterpolantND-specific
# overload too so `integrate(itp, a, b)` doesn't surface as a MethodError
# asking the user about Real vs Tuple bound types. Must use `::Real, ::Real`
# to win dispatch against the `(::AbstractInterpolant, ::Real, ::Real)`
# fallback in integrate_api.jl (more specific first arg + equal specificity
# on bounds).
function integrate(
        itp::HeteroInterpolantND, ::Real, ::Real;
        search = nothing, hint = nothing,
    )
    _throw_hetero_nd_integrate_unsupported(itp.methods)
end

@noinline function _throw_hetero_nd_integrate_unsupported(methods)
    throw(
        ArgumentError(
            "integrate is not yet implemented for HeteroInterpolantND " *
                "(method tuple: $(methods)). ND tensor-product integration over " *
                "mixed/Hermite axes is not yet supported. Workarounds: " *
                "(1) use a homogeneous specialized ND type " *
                "(`cubic_interp`, `quadratic_interp`, `linear_interp`, `constant_interp`) " *
                "which do support `integrate`; " *
                "(2) integrate axis-by-axis via 1D `integrate` calls on per-fiber " *
                "1D interpolants."
        )
    )
end

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
        hs = ntuple(d -> @inbounds(_get_h(itp.grids[d], idx[d])), Val(N))
        inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.grids[d], idx[d])), Val(N))
        total += _full_cell_integral_nd(itp, idx, hs, inv_hs)
    end
    return total
end

# ═══════════════════════════════════════════════════════════════
# cumulative_integrate / cumulative_integrate! — 1D prefix-sum
# ═══════════════════════════════════════════════════════════════
#
# Architecture: the in-place `cumulative_integrate!(out, itp)` is the real
# implementation; the allocating `cumulative_integrate(itp)` is a thin
# wrapper that allocates `out` and forwards. Every interpolant type gets
# *one* method on `cumulative_integrate!` — Hermite OnTheFly joins the
# generic path via `_full_cell_fn(itp)`'s internal dispatch on `itp.dy`.

@noinline function _throw_cumulative_length_mismatch(got::Int, expected::Int)
    throw(
        DimensionMismatch(
            "cumulative_integrate! output buffer length $got does not match grid length $expected"
        )
    )
end

@inline function _check_cumulative_out(out::AbstractVector, n::Int)
    length(out) == n || _throw_cumulative_length_mismatch(length(out), n)
    return nothing
end

# Generic 1D: catches Cubic, Linear, Quadratic, and the entire Hermite family
# (PreCompute + OnTheFly via `_full_cell_fn`'s trait dispatch on `itp.dy`).
function cumulative_integrate!(
        out::AbstractVector, itp::AbstractInterpolant{Tg, Tv}
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(itp)
    _check_cumulative_out(out, length(x))
    return _cumulative_integrate_1d!(out, x, _full_cell_fn(itp))
end

# Constant override: side is parametric → compiler knows concrete type,
# so `_full_cell_fn(itp, side)` dispatches to a distinct specialized method.
function cumulative_integrate!(
        out::AbstractVector, itp::ConstantInterpolant{Tg, Tv}
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(itp)
    _check_cumulative_out(out, length(x))
    return _cumulative_integrate_1d!(out, x, _full_cell_fn(itp, itp.side))
end

# Allocating wrappers: allocate output vector then forward to the in-place path.
function cumulative_integrate(itp::AbstractInterpolant{Tg, Tv}) where {Tg <: AbstractFloat, Tv}
    Tout = promote_type(Tv, Tg)
    out = Vector{Tout}(undef, length(_grid_1d(itp)))
    return cumulative_integrate!(out, itp)
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
        _cumulative_integrate_1d!(@view(result[:, k]), x, _full_cell_fn(sitp, k))
    end
    return result
end

# Constant Series override: side is parametric → compiler knows concrete type
function cumulative_integrate(
        sitp::ConstantSeriesInterpolant{Tg, Tv}
    ) where {Tg <: AbstractFloat, Tv}
    x = _grid_1d(sitp)
    Tout = promote_type(Tv, Tg)
    n_pts = length(x)
    n_ser = n_series(sitp)
    result = Matrix{Tout}(undef, n_pts, n_ser)
    @inbounds for k in 1:n_ser
        _cumulative_integrate_1d!(@view(result[:, k]), x, _full_cell_fn(sitp, k, sitp.side))
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
