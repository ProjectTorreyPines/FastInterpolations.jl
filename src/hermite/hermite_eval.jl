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

# NoExtrap / ExtendExtrap / generic: direct search + kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq)
    dL = xq - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_primal = _extract_primal(xq)
    if xq_primal < _extract_primal(first(x))
        return _eval_extrapolation(op, first(y), extrap, xq)
    elseif xq_primal > _extract_primal(last(x))
        return _eval_extrapolation(op, last(y), extrap, xq)
    end
    idx, idx_R, xL, _ = search_interval(searcher, x, xq)
    dL = xq - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# WrapExtrap: wrap query to domain → search + kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq,
        extrap::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, x)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq_wrapped)
    dL = xq_wrapped - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# Vector loop — generic
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
    extrap = _check_domain(x, xq, extrap)
    @inbounds for i in eachindex(xq, output)
        output[i] = _hermite_eval_at_point(x, y, dy, xq[i], extrap, deriv, searcher)
    end
    return output
end

# Vector loop — WrapExtrap specialization (2-stage: bulk-wrap + ExtendExtrap kernel)
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::AbstractVector{<:Real},
        extrap::WrapExtrap,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, O <: AbstractEvalOp, P <: Searcher}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(xq), maximum(xq)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(xq, output)
            output[i] = _hermite_eval_at_point(x, y, dy, xq[i], ExtendExtrap(), deriv, searcher)
        end
    else
        @inbounds for i in eachindex(xq, output)
            xi_wrapped = _wrap_to_domain(xq[i], x)
            output[i] = _hermite_eval_at_point(x, y, dy, xi_wrapped, ExtendExtrap(), deriv, searcher)
        end
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              ON-THE-FLY SLOPES (sm::AbstractSlopeMethod)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq)
    n = _data_length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dyL, dyR, h, inv_h, dL)
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
    xq_primal = _extract_primal(xq)
    if xq_primal < _extract_primal(first(x))
        return _eval_extrapolation(op, first(y), extrap, xq)
    elseif xq_primal > _extract_primal(last(x))
        return _eval_extrapolation(op, last(y), extrap, xq)
    end
    idx, idx_R, xL, _ = search_interval(searcher, x, xq)
    n = _data_length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dyL, dyR, h, inv_h, dL)
end

@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::Tq,
        extrap::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, x)
    idx, idx_R, xL, _ = search_interval(searcher, x, xq_wrapped)
    n = _data_length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq_wrapped - xL
    h = _get_h(x, idx)
    inv_h = _get_inv_h(x, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dyL, dyR, h, inv_h, dL)
end

# Vector loop — generic (slope method)
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
    extrap = _check_domain(x, xq, extrap)
    @inbounds for i in eachindex(xq, output)
        output[i] = _hermite_eval_at_point(x, y, sm, xq[i], extrap, deriv, searcher)
    end
    return output
end

# Vector loop — WrapExtrap specialization (slope method)
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::AbstractVector{<:Real},
        extrap::WrapExtrap,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, O <: AbstractEvalOp, P <: Searcher}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(xq), maximum(xq)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(xq, output)
            output[i] = _hermite_eval_at_point(x, y, sm, xq[i], ExtendExtrap(), deriv, searcher)
        end
    else
        @inbounds for i in eachindex(xq, output)
            xi_wrapped = _wrap_to_domain(xq[i], x)
            output[i] = _hermite_eval_at_point(x, y, sm, xi_wrapped, ExtendExtrap(), deriv, searcher)
        end
    end
    return output
end
