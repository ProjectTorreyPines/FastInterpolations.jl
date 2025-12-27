# ========================================
# Cubic Spline Evaluation Functions
# ========================================
# Internal functions for evaluating cubic splines.
# Include order: cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

# ========================================
# Core Evaluation Functions
# ========================================

"Evaluate natural cubic spline at a single point with operation dispatch."
@inline function _eval_cubic_at_point(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    idx, x0, x1 = _find_interval_with_bounds(x, xi)

    dt1 = xi - x0
    dt2 = x1 - xi
    h_i = h[idx+1]

    @inbounds begin
        z_i = z[idx]
        z_ip1 = z[idx+1]
        y_i = y[idx]
        y_ip1 = y[idx+1]
    end

    return _cubic_kernel(op, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)
end

"Evaluate periodic cubic spline at a single point with operation dispatch."
@inline function _eval_cubic_at_point_periodic(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    period::T,
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    xi_wrapped = _wrap_to_domain(xi, first(x), first(x) + period)
    idx, x0, x1 = _find_interval_with_bounds(x, xi_wrapped)

    dt1 = xi_wrapped - x0
    dt2 = x1 - xi_wrapped
    h_i = h[idx+1]

    @inbounds begin
        z_i = z[idx]
        z_ip1 = z[idx+1]
        y_i = y[idx]
        y_ip1 = y[idx+1]
    end

    return _cubic_kernel(op, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)
end

# ========================================
# Extrapolation-aware Evaluation
# ========================================

# Helper: result for constant extrapolation outside domain
# For value: return boundary y
# For derivatives: return zero (constant function has no slope/curvature)
@inline _constant_extrap_result(::EvalValue, y_boundary::T) where {T} = y_boundary
@inline _constant_extrap_result(::EvalDeriv1, ::T) where {T} = zero(T)
@inline _constant_extrap_result(::EvalDeriv2, ::T) where {T} = zero(T)

"Evaluate with no extrapolation - throws DomainError if outside domain."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:none},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    return _eval_cubic_at_point(x, y, h, z, xi, op)
end

"Evaluate with constant extrapolation - returns boundary values outside domain."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:constant},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    xi < first(x) && return _constant_extrap_result(op, @inbounds y[1])
    xi > last(x) && return _constant_extrap_result(op, @inbounds y[end])
    return _eval_cubic_at_point(x, y, h, z, xi, op)
end

"Evaluate with extension extrapolation - extends boundary polynomial."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:extension},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    return _eval_cubic_at_point(x, y, h, z, xi, op)
end

"Evaluate with coordinate wrapping (for natural BC with wrap extrapolation)."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:wrap},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    xi_wrapped = _wrap_to_domain(xi, first(x), last(x))
    return _eval_cubic_at_point(x, y, h, z, xi_wrapped, op)
end

# ========================================
# BC-Aware Evaluation Helper
# ========================================

"Evaluate with BC-aware dispatch (Periodic BC - ignores extrapolation) - backward compat."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ev::Val  # extrapolation ignored for periodic
) where {T<:AbstractFloat, X, F}
    _eval_with_bc(cache, y, h, z, xi, ev, EvalValue())
end

"Evaluate with BC-aware dispatch (Periodic BC) with op."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val,  # extrapolation ignored for periodic
    op::O
) where {T<:AbstractFloat, X, F, O<:AbstractEvalOp}
    _eval_cubic_at_point_periodic(cache.x, y, h, z, xi, cache.bc_data.period, op)
end

"Evaluate with BC-aware dispatch (Generic Derivative BC) - backward compat."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,BCPair{T,L,R}},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    extrap::Val
) where {T<:AbstractFloat, X, F, L<:PointBC{T}, R<:PointBC{T}}
    _eval_with_bc(cache, y, h, z, xi, extrap, EvalValue())
end

"Evaluate with BC-aware dispatch (Generic Derivative BC) with op."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,BCPair{T,L,R}},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    extrap::Val,
    op::O
) where {T<:AbstractFloat, X, F, L<:PointBC{T}, R<:PointBC{T}, O<:AbstractEvalOp}
    _eval_cubic_with_extrap(cache.x, y, h, z, xi, extrap, op)
end

# ========================================
# Vector Loop Functions
# ========================================

"Default vector loop (for :none, :constant, :extension) - backward compatible."
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BC},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ev::Val
) where {T<:AbstractFloat, X, F, BC}
    _cubic_vector_loop!(output, cache, y, z, x_query, ev, EvalValue())
end

"Default vector loop with op parameter (for :none, :constant, :extension)."
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BC},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ev::Val,
    op::O
) where {T<:AbstractFloat, X, F, BC, O<:AbstractEvalOp}
    @boundscheck _check_domain(cache.x, x_query, ev)
    @inbounds for (k, xq) in enumerate(x_query)
        output[k] = _eval_with_bc(cache, y, cache.h, z, xq, ev, op)
    end
end

"Optimized vector loop for Periodic BC - backward compat."
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ev::Val  # extrap ignored for periodic
) where {T<:AbstractFloat, X, F}
    _cubic_vector_loop!(output, cache, y, z, x_query, ev, EvalValue())
end

"Optimized vector loop for Periodic BC with op - uses 2-stage strategy."
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ::Val,  # extrap ignored for periodic
    op::O
) where {T<:AbstractFloat, X, F, O<:AbstractEvalOp}
    x_min = first(cache.x)
    x_max = x_min + cache.bc_data.period
    qmin, qmax = minimum(x_query), maximum(x_query)

    if qmin >= x_min && qmax < x_max
        # Fast path: all queries inside domain
        @inbounds for (k, xq) in enumerate(x_query)
            output[k] = _eval_cubic_at_point(cache.x, y, cache.h, z, xq, op)
        end
    else
        # Slow path: per-element wrap
        period = cache.bc_data.period
        @inbounds for (k, xq) in enumerate(x_query)
            output[k] = _eval_cubic_at_point_periodic(cache.x, y, cache.h, z, xq, period, op)
        end
    end
end

# ========================================
# Scalar Evaluation Entry Point
# ========================================

"Scalar cubic spline evaluation (solves system once, evaluates once)."
@inline function cubic_interp_scalar(
    cache::CubicSplineCache{T,X,F,BC},
    y::AbstractVector{T},
    x_query::T;
    extrap::Symbol=:none,
    order::Int=0
) where {T<:AbstractFloat, X, F, BC}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    z = _solve_system!(cache, y, cache.bc_data)

    @_dispatch_order order => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(cache.x, x_query, ev)
            _eval_with_bc(cache, y, cache.h, z, x_query, ev, op)
        end
    end
end
