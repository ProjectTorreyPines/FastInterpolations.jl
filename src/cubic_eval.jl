# ========================================
# Cubic Spline Evaluation Functions
# ========================================
# Internal functions for evaluating cubic splines.
# Include order: cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

# ========================================
# Core Evaluation Functions
# ========================================

"Evaluate natural cubic spline at a single point using pre-computed z coefficients."
@inline function _eval_cubic_at_point(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T
) where {T<:AbstractFloat}
    idx, x0, x1 = _find_interval_with_bounds(x, xi)

    dt1 = xi - x0
    dt2 = x1 - xi
    h_i = h[idx+1]

    @inbounds begin
        I = (z[idx] * dt2^3 + z[idx+1] * dt1^3) / (6 * h_i)
        C = (y[idx+1] / h_i - z[idx+1] * h_i / 6) * dt1
        D = (y[idx] / h_i - z[idx] * h_i / 6) * dt2
    end

    return I + C + D
end

"Evaluate periodic cubic spline at a single point (wraps coordinates)."
@inline function _eval_cubic_at_point_periodic(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    period::T
) where {T<:AbstractFloat}
    xi_wrapped = _wrap_to_domain(xi, first(x), first(x) + period)
    idx, x0, x1 = _find_interval_with_bounds(x, xi_wrapped)

    dt1 = xi_wrapped - x0
    dt2 = x1 - xi_wrapped
    h_i = h[idx+1]

    @inbounds begin
        I = (z[idx] * dt2^3 + z[idx+1] * dt1^3) / (6 * h_i)
        C = (y[idx+1] / h_i - z[idx+1] * h_i / 6) * dt1
        D = (y[idx] / h_i - z[idx] * h_i / 6) * dt2
    end

    return I + C + D
end

# ========================================
# Extrapolation-aware Evaluation
# ========================================

"Evaluate with no extrapolation - throws DomainError if outside domain."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:none}
) where {T<:AbstractFloat}
    return _eval_cubic_at_point(x, y, h, z, xi)
end

"Evaluate with constant extrapolation - returns boundary values outside domain."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:constant}
) where {T<:AbstractFloat}
    xi < first(x) && return @inbounds y[1]
    xi > last(x) && return @inbounds y[end]
    return _eval_cubic_at_point(x, y, h, z, xi)
end

"Evaluate with extension extrapolation - extends boundary polynomial."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:extension}
) where {T<:AbstractFloat}
    return _eval_cubic_at_point(x, y, h, z, xi)
end

"Evaluate with coordinate wrapping (for natural BC with wrap extrapolation)."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:wrap}
) where {T<:AbstractFloat}
    xi_wrapped = _wrap_to_domain(xi, first(x), last(x))
    return _eval_cubic_at_point(x, y, h, z, xi_wrapped)
end

# ========================================
# BC-Aware Evaluation Helper
# ========================================

"Evaluate with BC-aware dispatch (Periodic BC - ignores extrapolation)."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val  # extrapolation ignored for periodic
) where {T<:AbstractFloat, X, F}
    _eval_cubic_at_point_periodic(cache.x, y, h, z, xi, cache.bc_data.period)
end

"Evaluate with BC-aware dispatch (Generic Derivative BC - uses standard evaluation)."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,BCPair{T,L,R}},
    y::AbstractVector{T},
    h::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    extrap::Val
) where {T<:AbstractFloat, X, F, L<:PointBC{T}, R<:PointBC{T}}
    _eval_cubic_with_extrap(cache.x, y, h, z, xi, extrap)
end

# ========================================
# Vector Loop Functions
# ========================================

"Default vector loop (for :none, :constant, :extension)."
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BC},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ev::Val
) where {T<:AbstractFloat, X, F, BC}
    @boundscheck _check_domain(cache.x, x_query, ev)
    @inbounds for (k, xq) in enumerate(x_query)
        output[k] = _eval_with_bc(cache, y, cache.h, z, xq, ev)
    end
end

"Optimized vector loop for Periodic BC - uses 2-stage strategy."
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ::Val  # extrap ignored for periodic
) where {T<:AbstractFloat, X, F}
    x_min = first(cache.x)
    x_max = x_min + cache.bc_data.period
    qmin, qmax = minimum(x_query), maximum(x_query)

    if qmin >= x_min && qmax < x_max
        # Fast path: all queries inside domain
        @inbounds for (k, xq) in enumerate(x_query)
            output[k] = _eval_cubic_at_point(cache.x, y, cache.h, z, xq)
        end
    else
        # Slow path: per-element wrap
        period = cache.bc_data.period
        @inbounds for (k, xq) in enumerate(x_query)
            output[k] = _eval_cubic_at_point_periodic(cache.x, y, cache.h, z, xq, period)
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
    extrap::Symbol=:none
) where {T<:AbstractFloat, X, F, BC}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    z = _solve_system!(cache, y, cache.bc_data)

    @_dispatch_extrap extrap => ev begin
        @boundscheck _check_domain(cache.x, x_query, ev)
        _eval_with_bc(cache, y, cache.h, z, x_query, ev)
    end
end
