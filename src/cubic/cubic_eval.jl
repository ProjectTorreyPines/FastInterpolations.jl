# ========================================
# Cubic Spline Evaluation Functions
# ========================================
# Internal functions for evaluating cubic splines.
# Include order: cubic_types.jl → cubic_solver.jl → cubic_eval.jl → cubic_interp.jl

# ========================================
# Core Evaluation Functions
# ========================================

"Evaluate natural cubic spline at a single point with operation dispatch and search policy."
@inline function _eval_cubic_at_point(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    op::O,
    searcher::S
) where {T<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    idx, xL, xR = search_interval(searcher, x, spacing, xi)

    dL = xi - xL   # distance from Left endpoint
    dR = xR - xi   # distance from Right endpoint

    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx+1]
        yL = y[idx]
        yR = y[idx+1]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# Backward-compatible without searcher (uses default _search_interval)
@inline function _eval_cubic_at_point(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    idx, xL, xR = _search_interval(x, spacing, xi)

    dL = xi - xL   # distance from Left endpoint
    dR = xR - xi   # distance from Right endpoint

    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx+1]
        yL = y[idx]
        yR = y[idx+1]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

"Evaluate periodic cubic spline at a single point with operation dispatch and search policy."
@inline function _eval_cubic_at_point_periodic(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    period::T,
    op::O,
    searcher::S
) where {T<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    xi_wrapped = _wrap_to_domain(xi, first(x), first(x) + period)
    idx, xL, xR = search_interval(searcher, x, spacing, xi_wrapped)

    dL = xi_wrapped - xL   # distance from Left endpoint
    dR = xR - xi_wrapped   # distance from Right endpoint
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx+1]
        yL = y[idx]
        yR = y[idx+1]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# Backward-compatible without searcher
@inline function _eval_cubic_at_point_periodic(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    period::T,
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    xi_wrapped = _wrap_to_domain(xi, first(x), first(x) + period)
    idx, xL, xR = _search_interval(x, spacing, xi_wrapped)

    dL = xi_wrapped - xL   # distance from Left endpoint
    dR = xR - xi_wrapped   # distance from Right endpoint
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx+1]
        yL = y[idx]
        yR = y[idx+1]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
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
@inline _constant_extrap_result(::EvalDeriv3, ::T) where {T} = zero(T)

# --- Searcher-aware versions ---

"Evaluate with no extrapolation and search policy."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:none},
    op::O,
    searcher::S
) where {T<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    return _eval_cubic_at_point(x, y, spacing, z, xi, op, searcher)
end

"Evaluate with constant extrapolation and search policy."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:constant},
    op::O,
    searcher::S
) where {T<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    xi < first(x) && return _constant_extrap_result(op, @inbounds y[1])
    xi > last(x) && return _constant_extrap_result(op, @inbounds y[end])
    return _eval_cubic_at_point(x, y, spacing, z, xi, op, searcher)
end

"Evaluate with extension extrapolation and search policy."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:extension},
    op::O,
    searcher::S
) where {T<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    return _eval_cubic_at_point(x, y, spacing, z, xi, op, searcher)
end

"Evaluate with coordinate wrapping and search policy."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:wrap},
    op::O,
    searcher::S
) where {T<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    xi_wrapped = _wrap_to_domain(xi, first(x), last(x))
    return _eval_cubic_at_point(x, y, spacing, z, xi_wrapped, op, searcher)
end

# --- Backward-compatible versions (without searcher) ---

"Evaluate with no extrapolation - throws DomainError if outside domain."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:none},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    return _eval_cubic_at_point(x, y, spacing, z, xi, op)
end

"Evaluate with constant extrapolation - returns boundary values outside domain."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:constant},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    xi < first(x) && return _constant_extrap_result(op, @inbounds y[1])
    xi > last(x) && return _constant_extrap_result(op, @inbounds y[end])
    return _eval_cubic_at_point(x, y, spacing, z, xi, op)
end

"Evaluate with extension extrapolation - extends boundary polynomial."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:extension},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    return _eval_cubic_at_point(x, y, spacing, z, xi, op)
end

"Evaluate with coordinate wrapping (for natural BC with wrap extrapolation)."
@inline function _eval_cubic_with_extrap(
    x::AbstractVector{T},
    y::AbstractVector{T},
    spacing::AbstractGridSpacing{T},
    z::AbstractVector{T},
    xi::T,
    ::Val{:wrap},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    xi_wrapped = _wrap_to_domain(xi, first(x), last(x))
    return _eval_cubic_at_point(x, y, spacing, z, xi_wrapped, op)
end

# ========================================
# BC-Aware Evaluation Helper
# ========================================

# --- Searcher-aware versions ---

"Evaluate with BC-aware dispatch (Periodic BC) with search policy."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,PeriodicData{T},S},
    y::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val,  # extrapolation ignored for periodic
    op::O,
    searcher::P
) where {T<:AbstractFloat, X, F, S<:AbstractGridSpacing{T}, O<:AbstractEvalOp, P<:Searcher}
    _eval_cubic_at_point_periodic(cache.x, y, cache.spacing, z, xi, cache.bc_config.period, op, searcher)
end

"Evaluate with BC-aware dispatch (Generic Derivative BC) with search policy."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,BCPair{T,L,R},S},
    y::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    extrap::Val,
    op::O,
    searcher::P
) where {T<:AbstractFloat, X, F, L<:PointBC{T}, R<:PointBC{T}, S<:AbstractGridSpacing{T}, O<:AbstractEvalOp, P<:Searcher}
    _eval_cubic_with_extrap(cache.x, y, cache.spacing, z, xi, extrap, op, searcher)
end

# --- Backward-compatible versions (without searcher) ---

"Evaluate with BC-aware dispatch (Periodic BC) with op."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,PeriodicData{T},S},
    y::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    ::Val,  # extrapolation ignored for periodic
    op::O
) where {T<:AbstractFloat, X, F, S<:AbstractGridSpacing{T}, O<:AbstractEvalOp}
    _eval_cubic_at_point_periodic(cache.x, y, cache.spacing, z, xi, cache.bc_config.period, op)
end

"Evaluate with BC-aware dispatch (Generic Derivative BC) with op."
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,BCPair{T,L,R},S},
    y::AbstractVector{T},
    z::AbstractVector{T},
    xi::T,
    extrap::Val,
    op::O
) where {T<:AbstractFloat, X, F, L<:PointBC{T}, R<:PointBC{T}, S<:AbstractGridSpacing{T}, O<:AbstractEvalOp}
    _eval_cubic_with_extrap(cache.x, y, cache.spacing, z, xi, extrap, op)
end

# ========================================
# Vector Loop Functions
# ========================================

# --- Searcher-aware versions ---

"Default vector loop with search policy (for :none, :constant, :extension)."
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,BC,S},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ev::Val,
    op::O,
    searcher::P
) where {T<:AbstractFloat, X, F, BC, S<:AbstractGridSpacing{T}, O<:AbstractEvalOp, P<:Searcher}
    @boundscheck _check_domain(cache.x, x_query, ev)
    @inbounds for (k, xq) in enumerate(x_query)
        output[k] = _eval_with_bc(cache, y, z, xq, ev, op, searcher)
    end
end

"Optimized vector loop for Periodic BC with search policy - uses 2-stage strategy."
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T},S},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ::Val,  # extrap ignored for periodic
    op::O,
    searcher::P
) where {T<:AbstractFloat, X, F, S<:AbstractGridSpacing{T}, O<:AbstractEvalOp, P<:Searcher}
    x_min = first(cache.x)
    x_max = x_min + cache.bc_config.period
    qmin, qmax = minimum(x_query), maximum(x_query)

    if qmin >= x_min && qmax < x_max
        # Fast path: all queries inside domain
        @inbounds for (k, xq) in enumerate(x_query)
            output[k] = _eval_cubic_at_point(cache.x, y, cache.spacing, z, xq, op, searcher)
        end
    else
        # Slow path: per-element wrap
        period = cache.bc_config.period
        @inbounds for (k, xq) in enumerate(x_query)
            output[k] = _eval_cubic_at_point_periodic(cache.x, y, cache.spacing, z, xq, period, op, searcher)
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
    cache::CubicSplineCache{T,X,F,BC,S},
    y::AbstractVector{T},
    x_query::T;
    extrap::Symbol=:none,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {T<:AbstractFloat, X, F, BC, S<:AbstractGridSpacing{T}}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    z = similar!(pool, y)
    _solve_system!(z, cache, y, cache.bc_config)

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(cache.x, x_query, ev)
            _eval_with_bc(cache, y, z, x_query, ev, op, searcher)
        end
    end
end
