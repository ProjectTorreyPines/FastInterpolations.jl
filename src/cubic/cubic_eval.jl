# ========================================
# Cubic Spline Evaluation Functions
# ========================================
# Internal functions for evaluating cubic splines.
# Include order: cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

# ========================================
# Core Evaluation Functions
# ========================================

"Evaluate cubic spline at a single point with operation dispatch and search policy."
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector{Tv},
        xq::Tq,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    idx, xL, xR = search_interval(searcher, x, spacing, xq)

    # Use original xq for arithmetic to preserve AD
    dL = xq - xL   # distance from Left endpoint (can be Dual for AD)
    dR = xR - xq   # distance from Right endpoint (can be Dual for AD)

    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx + 1]
        yL = y[idx]
        yR = y[idx + 1]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# Backward-compatible without searcher (uses default _search_interval)
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector{Tv},
        xq::Tq,
        op::O
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp}
    idx, xL, xR = _search_interval(x, spacing, xq)

    # Use original xq for arithmetic to preserve AD
    dL = xq - xL   # distance from Left endpoint (can be Dual for AD)
    dR = xR - xq   # distance from Right endpoint (can be Dual for AD)

    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx + 1]
        yL = y[idx]
        yR = y[idx + 1]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

"Evaluate periodic cubic spline at a single point with operation dispatch and search policy."
@inline function _eval_cubic_at_point_periodic(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector{Tv},
        xq::Tq,
        period::Tg,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, first(x), first(x) + period)
    idx, xL, xR = search_interval(searcher, x, spacing, xq_wrapped)

    # Compute offset from original xq to preserve AD (adjust for wrapping)
    # For periodic, we use the wrapped position for arithmetic since
    # wrapping is a discrete operation that AD shouldn't track
    dL = xq_wrapped - xL   # distance from Left endpoint
    dR = xR - xq_wrapped   # distance from Right endpoint
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx + 1]
        yL = y[idx]
        yR = y[idx + 1]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# ========================================
# Extrapolation-aware Evaluation
# ========================================

# Helper: result for constant/fill extrapolation outside domain
# For value: return boundary y (ClampExtrap) or fill value (FillExtrap)
# For derivatives: always return zero from y_bnd (constant function has no slope/curvature)
# Uses y_bnd (not fill value) for derivatives to avoid 0 * NaN = NaN
#
# 4th arg `xq` enables type promotion for mixed-precision (Float32 data + Float64 query).
# For Number types: zero(xq)*zero(val) promotes to promote_type(typeof(val), typeof(xq)).
# For duck types (non-Number): fallback returns raw val/0*val, since the kernel also
# returns the same Tv when data and query share the same arithmetic type.

# ========================================
# Core eval: extrap dispatch → search → kernel (no intermediate layers)
# ========================================

# NoExtrap / ExtendExtrap / other: direct search + kernel.
# _check_domain(::NoExtrap) throws if OOB; search_interval clamps idx for ExtendExtrap.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, xL, xR = search_interval(searcher, x, spacing, xq)
    dL = xq - xL
    dR = xR - xq
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds begin
        zL = z[idx]; zR = z[idx + 1]
        yL = y[idx]; yR = y[idx + 1]
    end
    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector{Tv},
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_primal = _extract_primal(xq)
    xq_primal < first(x) && return _eval_extrapolation(op, first(y), extrap, xq)
    xq_primal > last(x) && return _eval_extrapolation(op, last(y), extrap, xq)
    idx, xL, xR = search_interval(searcher, x, spacing, xq)
    dL = xq - xL
    dR = xR - xq
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds begin
        zL = z[idx]; zR = z[idx + 1]
        yL = y[idx]; yR = y[idx + 1]
    end
    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# WrapExtrap: wrap query to domain → search + kernel.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector{Tv},
        xq::Tq,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg <: AbstractFloat, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, first(x), last(x))
    idx, xL, xR = search_interval(searcher, x, spacing, xq_wrapped)
    dL = xq_wrapped - xL
    dR = xR - xq_wrapped
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds begin
        zL = z[idx]; zR = z[idx + 1]
        yL = y[idx]; yR = y[idx + 1]
    end
    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end


# ========================================
# BC-Aware Evaluation Helper
# ========================================

# --- Searcher-aware versions ---

"Evaluate with BC-aware dispatch (Periodic BC) with search policy."
@inline function _eval_with_bc(
        cache::CubicSplineCache{Tg, X, F, PeriodicData{Tg}, S},
        y::AbstractVector{Tv},
        z::AbstractVector{Tv},
        xq::Tq,
        ::AbstractExtrap,  # extrapolation ignored for periodic
        op::O,
        searcher::P
    ) where {Tg <: AbstractFloat, Tv, Tq, X, F, S <: AbstractGridSpacing{Tg}, O <: AbstractEvalOp, P <: Searcher}
    return _eval_cubic_at_point_periodic(cache.x, y, cache.spacing, z, xq, cache.bc_config.period, op, searcher)
end

"Evaluate with BC-aware dispatch (Generic Derivative BC) with search policy."
@inline function _eval_with_bc(
        cache::CubicSplineCache{Tg, X, F, BCPair{L, R}, S},
        y::AbstractVector{Tv},
        z::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::P
    ) where {Tg <: AbstractFloat, Tv, Tq, X, F, L <: PointBC, R <: PointBC, S <: AbstractGridSpacing{Tg}, O <: AbstractEvalOp, P <: Searcher}
    return _eval_cubic_at_point(cache.x, y, cache.spacing, z, xq, extrap, op, searcher)
end


# ========================================
# Vector Loop Functions
# ========================================

"Vector loop for non-periodic BC. Accepts any Real query type (AD-compatible)."
@inline function _cubic_vector_loop!(
        output::AbstractVector,
        cache::CubicSplineCache{Tg, X, F, BC, S},
        y::AbstractVector{Tv},
        z::AbstractVector{Tv},
        x_query::AbstractVector{<:Real},
        ev::E,
        op::O,
        searcher::P
    ) where {Tg <: AbstractFloat, Tv, X, F, BC, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    @boundscheck _check_domain(cache.x, x_query, ev)
    return @inbounds for k in eachindex(x_query, output)
        output[k] = _eval_with_bc(cache, y, z, x_query[k], ev, op, searcher)
    end
end

"Vector loop for Periodic BC with 2-stage optimization. Accepts any Real query type."
@inline function _cubic_vector_loop!(
        output::AbstractVector,
        cache::CubicSplineCache{Tg, X, F, PeriodicData{Tg}, S},
        y::AbstractVector{Tv},
        z::AbstractVector{Tv},
        x_query::AbstractVector{<:Real},
        ::AbstractExtrap,  # extrap ignored for periodic
        op::O,
        searcher::P
    ) where {Tg <: AbstractFloat, Tv, X, F, S <: AbstractGridSpacing{Tg}, O <: AbstractEvalOp, P <: Searcher}
    x_min = first(cache.x)
    x_max = x_min + cache.bc_config.period
    qmin, qmax = minimum(x_query), maximum(x_query)

    return if qmin >= x_min && qmax < x_max
        # Fast path: all queries inside domain
        @inbounds for k in eachindex(x_query, output)
            output[k] = _eval_cubic_at_point(cache.x, y, cache.spacing, z, x_query[k], op, searcher)
        end
    else
        # Slow path: per-element wrap
        period = cache.bc_config.period
        @inbounds for k in eachindex(x_query, output)
            output[k] = _eval_cubic_at_point_periodic(cache.x, y, cache.spacing, z, x_query[k], period, op, searcher)
        end
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
        cache::CubicSplineCache{Tg, X, F, BC, S},
        y::AbstractVector{Tv},
        x_query::Tg;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv, X, F, BC, S <: AbstractGridSpacing{Tg}}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    searcher = _resolve_search(cache.x, x_query, search, hint)
    @boundscheck _check_domain(cache.x, x_query, extrap)
    _eval_with_bc(cache, y, z, x_query, extrap, deriv, searcher)
end
