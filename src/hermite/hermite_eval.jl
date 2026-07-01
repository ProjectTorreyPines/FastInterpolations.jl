# ========================================
# Cubic Hermite 1D Evaluation Core
# ========================================
# Single axis-as-truth path: `_get_h(x, idx)` reads h/inv_h from the wrapped
# axis (`_CachedRange`/`_CachedVector`/`_ExclusivePeriodicAxis`) when present,
# or falls through to `x[idx+1] - x[idx]` for raw vectors. Three extrap
# overloads (AbstractExtrap, _ClampOrFill, WrapExtrap) × two slope shapes
# (`dy::AbstractVector` precomputed / `sm::AbstractSlopeMethod` on-the-fly).

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  PRECOMPUTED SLOPES (dy::AbstractVector)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Core in-bounds path: search + kernel. All non-InBounds extrap overloads
# delegate here after preprocessing — see cubic_eval.jl for the same
# `InBounds = core fast path` pattern.
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq,
        ::InBounds,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq, InBounds())
    dL = xq - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# NoExtrap / ExtendExtrap / others matching AbstractExtrap: domain check
# → delegate.
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    # NoExtrap → InBounds for the search once the domain check passes (lean search).
    # ExtendExtrap passes through: it may arrive OOB → standard two-sided-clamp search (not the
    # lean InBounds one, whose one-sided clamp would give idx ≤ 0 OOB-left); boundary cell extrapolates.
    extrap_eff = _check_domain(x, xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq, extrap_eff)
    dL = xq - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or delegate.
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
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
    st == OOB_LEFT && return _eval_extrapolation(op, first(y), extrap, xq)
    st == OOB_RIGHT && return _eval_extrapolation(op, last(y), extrap, xq)
    return _hermite_eval_at_point(x, y, dy, xq, InBounds(), op, searcher)
end

# WrapExtrap: wrap query to domain → delegate with wrapped value.
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(_resolve_grididx(xq, x), x)
    return _hermite_eval_at_point(x, y, dy, xq_wrapped, InBounds(), op, searcher)
end

# Vector loop — generic. WrapExtrap fast/slow path is routed by
# `_check_domain` for Clamp/Fill/Wrap returns `Union{InBounds, E}`. Passing
# through a function-barrier call lets Julia's union-splitting specialize
# the inner loop per concrete `extrap` (no per-iteration union dispatch);
# more robust than relying on LLVM loop unswitching, which can give up on
# union splitting when the inner kernel exceeds heuristic size thresholds.
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    extrap_eff = _check_domain(x, xq, extrap)
    return _hermite_vector_loop_inner!(output, x, y, dy, xq, extrap_eff, deriv, searcher)
end

@inline function _hermite_vector_loop_inner!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    @inbounds for i in eachindex(xq, output)
        output[i] = _hermite_eval_at_point(x, y, dy, xq[i], extrap, deriv, searcher)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              ON-THE-FLY SLOPES (sm::AbstractSlopeMethod)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Core in-bounds path (slope method): search + local-slope + kernel. All
# non-InBounds overloads delegate here after preprocessing — same pattern
# as the pre-baked-slopes variant above.
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::Tq,
        ::InBounds,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq, InBounds())
    n = _data_length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    yr = _raw(y)
    @inbounds return _hermite_kernel_1d(op, yr[idx], yr[idx_R], dyL, dyR, h, inv_h, dL)
end

@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq = _resolve_grididx(xq, x)
    # NoExtrap → InBounds for the search once the domain check passes (lean search).
    # ExtendExtrap passes through: it may arrive OOB → standard two-sided-clamp search (not the
    # lean InBounds one, whose one-sided clamp would give idx ≤ 0 OOB-left); boundary cell extrapolates.
    extrap_eff = _check_domain(x, xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq, extrap_eff)
    n = _data_length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    yr = _raw(y)
    @inbounds return _hermite_kernel_1d(op, yr[idx], yr[idx_R], dyL, dyR, h, inv_h, dL)
end

@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
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
    st == OOB_LEFT && return _eval_extrapolation(op, first(y), extrap, xq)
    st == OOB_RIGHT && return _eval_extrapolation(op, last(y), extrap, xq)
    return _hermite_eval_at_point(x, y, sm, xq, InBounds(), op, searcher)
end

@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::Tq,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(_resolve_grididx(xq, x), x)
    return _hermite_eval_at_point(x, y, sm, xq_wrapped, InBounds(), op, searcher)
end

# Vector loop — generic (slope method). Function-barrier pattern: outer
# resolves domain, inner sees concrete `extrap` (see pre-baked-slopes
# variant above for rationale).
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    extrap_eff = _check_domain(x, xq, extrap)
    return _hermite_vector_loop_inner!(output, x, y, sm, xq, extrap_eff, deriv, searcher)
end

@inline function _hermite_vector_loop_inner!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    @inbounds for i in eachindex(xq, output)
        output[i] = _hermite_eval_at_point(x, y, sm, xq[i], extrap, deriv, searcher)
    end
    return output
end
