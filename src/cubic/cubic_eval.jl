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

# Core in-bounds path: search + kernel only. All other extrap overloads do
# their preprocessing (boundary check / wrap / OOB return) and then delegate
# here once the query is known to be inside the domain. `_resolve_grididx`
# is called here as well so direct InBounds-dispatch callers (e.g., the
# `_cubic_vector_loop!` Union-splitting fast path) get GridIdx resolution
# without going through a preprocess overload.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        e::InBounds,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    idx, idx_R, xL, xR = search_interval(searcher, x, xq, e)
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

# NoExtrap / ExtendExtrap / others matching AbstractExtrap: domain check (NoExtrap throws
# on OOB; ExtendExtrap is a no-op and may arrive OOB). Runs the standard two-sided-clamp
# search + kernel HERE — must NOT delegate to the lean `::InBounds` core, whose one-sided
# clamp would return idx ≤ 0 on an OOB-left ExtendExtrap query. The boundary cell (idx ∈
# [1, n-1]) lets `dL`/`dR` extrapolate past the ends.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    # NoExtrap → InBounds for the search once the domain check passes (lean search);
    # ExtendExtrap passes through and keeps the two-sided-clamp search (it may arrive OOB).
    extrap_eff = _check_domain(x, xq, extrap)
    idx, idx_R, xL, xR = search_interval(searcher, x, xq, extrap_eff)
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

# ClampExtrap / FillExtrap: boundary check → extrap value or delegate.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    # Promote to Tc so the OOB extrap value carries the grid carrier (Dual grid →
    # Dual), matching the in-domain kernel. Identity on Float64; Int grids stay Int.
    xq = _promote_coord(_resolve_grididx(xq, x), eltype(x))
    xq_primal = _extract_primal(xq)
    st = _oob_state(x, xq_primal)
    scale = _deriv_unit_scale(oneunit(eltype(x)), op)
    st == OOB_LEFT && return _eval_extrapolation(op, first(y), extrap, xq, scale)
    st == OOB_RIGHT && return _eval_extrapolation(op, last(y), extrap, xq, scale)
    return _eval_cubic_at_point(x, y, z, xq, InBounds(), op, searcher)
end

# WrapExtrap: wrap query to domain → delegate with wrapped value.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(_resolve_grididx(xq, x), x)
    return _eval_cubic_at_point(x, y, z, xq_wrapped, InBounds(), op, searcher)
end

# ========================================
# Vector Loop Function
# ========================================
#
# Unified across periodic and non-periodic: the wrapped axis carries seam
# semantics (`_get_h(x, idx)` for any idx; `_wrap_to_domain` against
# `(first, last)`); the extrap branch (Wrap vs Clamp/Fill vs No/Extend) is
# the only differentiator and is handled inside `_eval_cubic_at_point`.

"Vector loop for cubic spline. Accepts any query type (duck: AD Dual, Unitful)."
@inline function _cubic_vector_loop!(
        output::AbstractArray,
        cache::CubicSplineCache{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        x_query::AbstractArray{Tq},
        ev::E,
        op::O,
        searcher::P
    ) where {Tg, Tv, Tq, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    # Resolve domain here so the inner kernel sees a concrete `ev` type.
    # `_check_domain` for Clamp/Fill/Wrap returns `Union{InBounds, E}` —
    # passing through a function-barrier call lets Julia's union-splitting
    # specialize the inner loop per concrete `ev` (no per-iteration union
    # dispatch). For `NoExtrap` paths return type is already `InBounds`, so
    # this is a no-op.
    ev_eff = _check_domain(cache.x, x_query, ev)
    return _cubic_vector_loop_inner!(output, cache, y, z, x_query, ev_eff, op, searcher)
end

@inline function _cubic_vector_loop_inner!(
        output::AbstractArray,
        cache::CubicSplineCache{Tg},
        y::AbstractVector{Tv},
        z::AbstractVector,
        x_query::AbstractArray{Tq},
        ev::E,
        op::O,
        searcher::P
    ) where {Tg, Tv, Tq, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    @inbounds for k in eachindex(x_query, output)
        output[k] = _eval_cubic_at_point(cache.x, y, z, x_query[k], ev, op, searcher)
    end
    return output
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
    ) where {Tg, Tv, Tq}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    Tz = _promote_eltype(_coeff_op2, eltype(cache.x), Tv)
    z = acquire!(pool, Tz, length(y))
    _solve_system!(z, cache, y, cache.bc)

    searcher = _resolve_search(cache.x, x_query, search, hint)
    extrap_eff = _resolve_extrap(extrap, cache.bc, cache.x)
    # Domain check delegated to `_eval_cubic_at_point(::AbstractExtrap)`
    # below — its inner `@boundscheck _check_domain` handles NoExtrap throw,
    # while Clamp/Fill/Wrap specialized methods handle OOB without checking.
    _eval_cubic_at_point(cache.x, y, z, x_query, extrap_eff, deriv, searcher)
end
