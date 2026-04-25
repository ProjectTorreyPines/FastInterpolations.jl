# ========================================
# Cubic Hermite 1D Evaluation Core
# ========================================
# Two call patterns:
#   - Oneshot: no spacing arg (computes h, inv_h from xR-xL inline)
#   - Interpolant: with spacing arg (precomputed h, inv_h for perf)
#
# Each has 3 extrap overloads: AbstractExtrap, _ClampOrFill, WrapExtrap
# Mirrors _linear_eval_at_point pattern from linear_oneshot.jl.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     ONESHOT PATH (no spacing)                             ║
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
    idx, idx_R, xL, xR = search_interval(searcher, x, xq)
    dL = xq - xL
    h = _get_h(x, xL, xR)
    inv_h = _get_inv_h(x, xL, xR)
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
    idx, idx_R, xL, xR = search_interval(searcher, x, xq)
    dL = xq - xL
    h = _get_h(x, xL, xR)
    inv_h = _get_inv_h(x, xL, xR)
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
    xq_wrapped = _wrap_to_domain(xq, extrap)
    idx, idx_R, xL, xR = search_interval(searcher, x, xq_wrapped)
    dL = xq_wrapped - xL
    h = _get_h(x, xL, xR)
    inv_h = _get_inv_h(x, xL, xR)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  INTERPOLANT PATH (with spacing)                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# NoExtrap / ExtendExtrap / generic: direct search + kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, spacing, xq)
    dL = xq - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
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
    idx, idx_R, xL, _ = search_interval(searcher, x, spacing, xq)
    dL = xq - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# WrapExtrap: wrap query to domain → search + kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::Tq,
        extrap::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, spacing, xq_wrapped)
    dL = xq_wrapped - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dy[idx], dy[idx_R], h, inv_h, dL)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR LOOPS                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Oneshot vector loop (no spacing)
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

# Oneshot vector loop — WrapExtrap specialization (2-stage optimization)
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
    x_min, x_max = extrap._x_min, extrap._x_max
    qmin, qmax = minimum(xq), maximum(xq)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(xq, output)
            output[i] = _hermite_eval_at_point(x, y, dy, xq[i], ExtendExtrap(), deriv, searcher)
        end
    else
        @inbounds for i in eachindex(xq, output)
            xi_wrapped = _wrap_to_domain(xq[i], extrap)
            output[i] = _hermite_eval_at_point(x, y, dy, xi_wrapped, ExtendExtrap(), deriv, searcher)
        end
    end
    return output
end

# Interpolant vector loop (with spacing)
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    extrap = _check_domain(x, xq, extrap)
    @inbounds for i in eachindex(xq, output)
        output[i] = _hermite_eval_at_point(x, spacing, y, dy, xq[i], extrap, deriv, searcher)
    end
    return output
end

# Interpolant vector loop — WrapExtrap specialization
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector,
        xq::AbstractVector{<:Real},
        extrap::WrapExtrap,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, O <: AbstractEvalOp, P <: Searcher}
    x_min, x_max = extrap._x_min, extrap._x_max
    qmin, qmax = minimum(xq), maximum(xq)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(xq, output)
            output[i] = _hermite_eval_at_point(x, spacing, y, dy, xq[i], ExtendExtrap(), deriv, searcher)
        end
    else
        @inbounds for i in eachindex(xq, output)
            xi_wrapped = _wrap_to_domain(xq[i], extrap)
            output[i] = _hermite_eval_at_point(x, spacing, y, dy, xi_wrapped, ExtendExtrap(), deriv, searcher)
        end
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              ONTHEFLY PATH — AbstractSlopeMethod overloads                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Mirror of the above paths but with sm::AbstractSlopeMethod instead of dy::AbstractVector.
# Uses _local_slope(sm, x, y, idx, n) to compute slopes per-cell in O(1).

# ── Oneshot scalar (no spacing) ─────────────────────────────────

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
    idx, idx_R, xL, xR = search_interval(searcher, x, xq)
    n = length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq - xL
    h = _get_h(x, xL, xR)
    inv_h = _get_inv_h(x, xL, xR)
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
    idx, idx_R, xL, xR = search_interval(searcher, x, xq)
    n = length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq - xL
    h = _get_h(x, xL, xR)
    inv_h = _get_inv_h(x, xL, xR)
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
    xq_wrapped = _wrap_to_domain(xq, extrap)
    idx, idx_R, xL, xR = search_interval(searcher, x, xq_wrapped)
    n = length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq_wrapped - xL
    h = _get_h(x, xL, xR)
    inv_h = _get_inv_h(x, xL, xR)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dyL, dyR, h, inv_h, dL)
end

# ── Interpolant scalar (with spacing) ───────────────────────────

@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, spacing, xq)
    n = length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dyL, dyR, h, inv_h, dL)
end

@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
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
    idx, idx_R, xL, _ = search_interval(searcher, x, spacing, xq)
    n = length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dyL, dyR, h, inv_h, dL)
end

@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::Tq,
        extrap::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, extrap)
    idx, idx_R, xL, _ = search_interval(searcher, x, spacing, xq_wrapped)
    n = length(x)
    dyL = _local_slope(sm, x, y, idx, n)
    dyR = _local_slope(sm, x, y, idx_R, n)
    dL = xq_wrapped - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx_R], dyL, dyR, h, inv_h, dL)
end

# ── Vector loops (delegate to scalar overloads above) ────────────

# Oneshot vector loop (no spacing)
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

# Oneshot vector loop — WrapExtrap specialization
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
    x_min, x_max = extrap._x_min, extrap._x_max
    qmin, qmax = minimum(xq), maximum(xq)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(xq, output)
            output[i] = _hermite_eval_at_point(x, y, sm, xq[i], ExtendExtrap(), deriv, searcher)
        end
    else
        @inbounds for i in eachindex(xq, output)
            xi_wrapped = _wrap_to_domain(xq[i], extrap)
            output[i] = _hermite_eval_at_point(x, y, sm, xi_wrapped, ExtendExtrap(), deriv, searcher)
        end
    end
    return output
end

# Interpolant vector loop (with spacing)
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    extrap = _check_domain(x, xq, extrap)
    @inbounds for i in eachindex(xq, output)
        output[i] = _hermite_eval_at_point(x, spacing, y, sm, xq[i], extrap, deriv, searcher)
    end
    return output
end

# Interpolant vector loop — WrapExtrap specialization
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        sm::AbstractSlopeMethod,
        xq::AbstractVector{<:Real},
        extrap::WrapExtrap,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, O <: AbstractEvalOp, P <: Searcher}
    x_min, x_max = extrap._x_min, extrap._x_max
    qmin, qmax = minimum(xq), maximum(xq)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(xq, output)
            output[i] = _hermite_eval_at_point(x, spacing, y, sm, xq[i], ExtendExtrap(), deriv, searcher)
        end
    else
        @inbounds for i in eachindex(xq, output)
            xi_wrapped = _wrap_to_domain(xq[i], extrap)
            output[i] = _hermite_eval_at_point(x, spacing, y, sm, xi_wrapped, ExtendExtrap(), deriv, searcher)
        end
    end
    return output
end
