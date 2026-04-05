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
        dy::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, xL, xR = search_interval(searcher, x, xq)
    dL = xq - xL
    h = _get_h(x, xR, xL)
    inv_h = _get_inv_h(x, xR, xL)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx + 1], dy[idx], dy[idx + 1], h, inv_h, dL)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_primal = _extract_primal(xq)
    if xq_primal < first(x)
        return _eval_extrapolation(op, first(y), extrap, xq)
    elseif xq_primal > last(x)
        return _eval_extrapolation(op, last(y), extrap, xq)
    end
    idx, xL, xR = search_interval(searcher, x, xq)
    dL = xq - xL
    h = _get_h(x, xR, xL)
    inv_h = _get_inv_h(x, xR, xL)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx + 1], dy[idx], dy[idx + 1], h, inv_h, dL)
end

# WrapExtrap: wrap query to domain → search + kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        xq::Tq,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, first(x), last(x))
    idx, xL, xR = search_interval(searcher, x, xq_wrapped)
    dL = xq_wrapped - xL
    h = _get_h(x, xR, xL)
    inv_h = _get_inv_h(x, xR, xL)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx + 1], dy[idx], dy[idx + 1], h, inv_h, dL)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  INTERPOLANT PATH (with spacing)                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# NoExtrap / ExtendExtrap / generic: direct search + kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, xL, _ = search_interval(searcher, x, spacing, xq)
    dL = xq - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx + 1], dy[idx], dy[idx + 1], h, inv_h, dL)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_primal = _extract_primal(xq)
    if xq_primal < first(x)
        return _eval_extrapolation(op, first(y), extrap, xq)
    elseif xq_primal > last(x)
        return _eval_extrapolation(op, last(y), extrap, xq)
    end
    idx, xL, _ = search_interval(searcher, x, spacing, xq)
    dL = xq - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx + 1], dy[idx], dy[idx + 1], h, inv_h, dL)
end

# WrapExtrap: wrap query to domain → search + kernel
@inline function _hermite_eval_at_point(
        x::AbstractVector{Tg},
        spacing::AbstractGridSpacing{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        xq::Tq,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, first(x), last(x))
    idx, xL, _ = search_interval(searcher, x, spacing, xq_wrapped)
    dL = xq_wrapped - xL
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds return _hermite_kernel_1d(op, y[idx], y[idx + 1], dy[idx], dy[idx + 1], h, inv_h, dL)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR LOOPS                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Oneshot vector loop (no spacing)
@inline function _hermite_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        dy::AbstractVector{Tv},
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg <: AbstractFloat, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
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
        dy::AbstractVector{Tv},
        xq::AbstractVector{<:Real},
        ::WrapExtrap,
        deriv::O,
        searcher::P
    ) where {Tg <: AbstractFloat, Tv, O <: AbstractEvalOp, P <: Searcher}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(xq), maximum(xq)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(xq, output)
            output[i] = _hermite_eval_at_point(x, y, dy, xq[i], ExtendExtrap(), deriv, searcher)
        end
    else
        @inbounds for i in eachindex(xq, output)
            xi_wrapped = _wrap_to_domain(xq[i], x_min, x_max)
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
        dy::AbstractVector{Tv},
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg <: AbstractFloat, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
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
        dy::AbstractVector{Tv},
        xq::AbstractVector{<:Real},
        ::WrapExtrap,
        deriv::O,
        searcher::P
    ) where {Tg <: AbstractFloat, Tv, O <: AbstractEvalOp, P <: Searcher}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(xq), maximum(xq)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(xq, output)
            output[i] = _hermite_eval_at_point(x, spacing, y, dy, xq[i], ExtendExtrap(), deriv, searcher)
        end
    else
        @inbounds for i in eachindex(xq, output)
            xi_wrapped = _wrap_to_domain(xq[i], x_min, x_max)
            output[i] = _hermite_eval_at_point(x, spacing, y, dy, xi_wrapped, ExtendExtrap(), deriv, searcher)
        end
    end
    return output
end
