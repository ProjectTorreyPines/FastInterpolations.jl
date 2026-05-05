# ========================================
# Cubic Spline Evaluation Functions
# ========================================
# Internal functions for evaluating cubic splines.
# Include order: cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

# ========================================
# Core Evaluation Functions
# ========================================

# ========================================
# Extrapolation-aware Evaluation
# ========================================
#
# `cache.x` is the cached/wrapped axis — `_get_h(x, idx)` / `_get_inv_h(x, idx)`
# handle interior cells (`_CachedRange`/`_CachedVector`) and seam cells
# (`_ExclusivePeriodicAxis` virtual width) uniformly. Periodic and non-periodic
# share the same eval kernel; the only differentiator is the extrap branch
# (`WrapExtrap` vs others), which `_resolve_extrap` materializes from `cache.bc`
# at construction time.

# NoExtrap / ExtendExtrap: direct search + kernel.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    # Resolve bare `GridIdx(k)` (carries `val=NaN` until bound) to its grid
    # coordinate so subsequent arithmetic sees a real value.
    xq = _resolve_grididx(xq, x)
    @boundscheck _check_domain(x, xq, extrap)
    idx, idx_R, xL, xR = search_interval(searcher, x, xq)
    dL = xq - xL
    dR = xR - xq
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds begin
        zL = z[idx]; zR = z[idx_R]
        yL = y[idx]; yR = y[idx_R]
    end
    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    xq_primal = _extract_primal(xq)
    xq_primal < first(x) && return _eval_extrapolation(op, first(y), extrap, xq)
    xq_primal > last(x) && return _eval_extrapolation(op, last(y), extrap, xq)
    idx, idx_R, xL, xR = search_interval(searcher, x, xq)
    dL = xq - xL
    dR = xR - xq
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds begin
        zL = z[idx]; zR = z[idx_R]
        yL = y[idx]; yR = y[idx_R]
    end
    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# WrapExtrap: wrap query to domain → search + kernel.
# Wrap domain `[first(x), last(x))` is read directly from the (possibly-wrapped) axis.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        extrap::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    xq_wrapped = _wrap_to_domain(xq, x)
    idx, idx_R, xL, xR = search_interval(searcher, x, xq_wrapped)
    dL = xq_wrapped - xL
    dR = xR - xq_wrapped
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds begin
        zL = z[idx]; zR = z[idx_R]
        yL = y[idx]; yR = y[idx_R]
    end
    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

"""
    _eval_cubic_deriv1_and_deriv2(itp::CubicInterpolant, xq) -> (d1, d2)

Evaluate the first **and** second derivatives of a non-periodic `CubicInterpolant`
at `xq` with a **single** interval search.  Avoids the binary-search overhead of two
independent `itp(xq; deriv=DerivOp(1))` / `itp(xq; deriv=DerivOp(2))` calls when
both derivatives are needed at the same point (e.g. in the atom loops of
`PromolecularRef`).

Only valid for non-periodic boundary conditions (`BCPair`).
Handles `FillExtrap` / `ClampExtrap` out-of-bounds by returning zero for both
derivatives (consistent with a constant fill).
"""
@inline function _eval_cubic_deriv1_and_deriv2(
        itp::CubicInterpolant{Tg, Tv, <:CubicSplineCache{Tg, <:Any, <:Any, <:BCPair}},
        xq::Real,
    ) where {Tg, Tv}
    cache  = itp.cache
    extrap = itp.extrap
    x      = cache.x
    sp     = cache.spacing
    xq_primal = _extract_primal(xq)
    if xq_primal < first(x)
        d1 = _eval_extrapolation(EvalDeriv1(), first(itp.y), extrap, xq)
        d2 = _eval_extrapolation(EvalDeriv2(), first(itp.y), extrap, xq)
        return d1, d2
    end
    if xq_primal > last(x)
        d1 = _eval_extrapolation(EvalDeriv1(), last(itp.y), extrap, xq)
        d2 = _eval_extrapolation(EvalDeriv2(), last(itp.y), extrap, xq)
        return d1, d2
    end
    searcher  = _resolve_search(x, xq, itp.search_policy, nothing)
    idx, idx_R, xL, xR = search_interval(searcher, x, sp, xq)
    dL    = xq - xL
    dR    = xR - xq
    h     = _get_h(sp, idx)
    inv_h = _get_inv_h(sp, idx)
    @inbounds begin
        zL = itp.z[idx];   zR = itp.z[idx_R]
        yL = itp.y[idx];   yR = itp.y[idx_R]
    end
    d1 = _cubic_kernel(EvalDeriv1(), zL, zR, yL, yR, h, inv_h, dL, dR)
    d2 = _cubic_kernel(EvalDeriv2(), zL, zR, yL, yR, h, inv_h, dL, dR)
    return d1, d2
end


# ========================================
# Vector Loop Function
# ========================================
#
# Unified across periodic and non-periodic: the wrapped axis carries seam
# semantics (`_get_h(x, idx)` for any idx; `_wrap_to_domain` against
# `(first, last)`); the extrap branch (Wrap vs Clamp/Fill vs No/Extend) is
# the only differentiator and is handled inside `_eval_cubic_at_point`.

"Vector loop for cubic spline. Accepts any Real query type (AD-compatible)."
@inline function _cubic_vector_loop!(
        output::AbstractVector,
        cache::CubicSplineCache{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        x_query::AbstractVector{<:Real},
        ev::E,
        op::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    ev = _check_domain(cache.x, x_query, ev)
    return @inbounds for k in eachindex(x_query, output)
        output[k] = _eval_cubic_at_point(cache.x, y, z, x_query[k], ev, op, searcher)
    end
end

# ========================================
# Scalar Evaluation Entry Point
# ========================================

"""
Scalar cubic spline evaluation (solves system once, evaluates once).

# Thread-Safety
Uses task-local pool for workspace allocation.
"""
@inline @with_pool pool function cubic_interp_scalar(
        cache::CubicSplineCache{Tg},
        y::AbstractVector{Tv},
        x_query::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    Tz = _output_eltype(Tv, eltype(cache.x))
    z = acquire!(pool, Tz, length(y))
    _solve_system!(z, cache, y, cache.bc)

    searcher = _resolve_search(cache.x, x_query, search, hint)
    extrap_eff = _resolve_extrap(extrap, cache.bc, cache.x)
    @boundscheck _check_domain(cache.x, x_query, extrap_eff)
    _eval_cubic_at_point(cache.x, y, z, x_query, extrap_eff, deriv, searcher)
end
