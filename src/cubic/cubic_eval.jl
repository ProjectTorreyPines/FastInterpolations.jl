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
        z::AbstractVector,
        xq::Tq,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    idx, idx_R, xL, xR = search_interval(searcher, x, spacing, xq)

    # Use original xq for arithmetic to preserve AD
    dL = xq - xL   # distance from Left endpoint (can be Dual for AD)
    dR = xR - xq   # distance from Right endpoint (can be Dual for AD)

    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx_R]
        yL = y[idx]
        yR = y[idx_R]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# Backward-compatible without searcher (uses default _search_interval)
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector,
        xq::Tq,
        op::O
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp}
    idx, idx_R, xL, xR = _search_interval(x, spacing, xq)

    # Use original xq for arithmetic to preserve AD
    dL = xq - xL   # distance from Left endpoint (can be Dual for AD)
    dR = xR - xq   # distance from Right endpoint (can be Dual for AD)

    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)

    @inbounds begin
        zL = z[idx]
        zR = z[idx_R]
        yL = y[idx]
        yR = y[idx_R]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

"""
Evaluate periodic cubic spline at a single point with operation dispatch and search policy.

`bc_config::PeriodicData{Tg, E}` provides:
- `period` for query wrap (`_wrap_to_domain`).
- `h_n` (seam-cell width) — for `:exclusive` form, the spacing object only
  carries n-1 interior cell widths; the seam cell (`idx == n`) reads from
  `bc_config.h_n` instead. Inclusive form is unaffected (spacing has the
  full n cells); the BC-aware accessor below dispatches on `E`.
"""
@inline function _eval_cubic_at_point_periodic(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector,
        xq::Tq,
        bc_config::PeriodicData{Tg},
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, first(x), first(x) + bc_config.period)
    idx, idx_R, xL, xR = search_interval(searcher, x, spacing, xq_wrapped)

    # Compute offset from original xq to preserve AD (adjust for wrapping)
    # For periodic, we use the wrapped position for arithmetic since
    # wrapping is a discrete operation that AD shouldn't track
    dL = xq_wrapped - xL   # distance from Left endpoint
    dR = xR - xq_wrapped   # distance from Right endpoint
    h, inv_h = _periodic_cell_h(spacing, idx, bc_config)

    @inbounds begin
        zL = z[idx]
        zR = z[idx_R]
        yL = y[idx]
        yR = y[idx_R]
    end

    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# Periodic cell-width accessor — BC-aware via `PeriodicData{Tg, E}` type-param.
#
# Inclusive form: spacing carries all n cells (length(x) - 1 = n), so any
#   `idx in 1..n` resolves through the standard accessor.
# Exclusive form: spacing carries only n-1 interior cells; the seam cell
#   (`idx == length(z)` where `length(z) == n`) reads `h_n` from `bc_config`.
#   `idx_R == 1` at the seam from the searcher dispatch confirms this is
#   the wrapped cell, but indexing on `idx > length(spacing entries)` is
#   simpler and uniform across ScalarSpacing/VectorSpacing.
@inline _periodic_cell_h(spacing, idx::Int, ::PeriodicData{Tg, :inclusive}) where {Tg} =
    (_get_h(spacing, idx), _get_inv_h(spacing, idx))

@inline function _periodic_cell_h(spacing::AbstractGridSpacing{Tg}, idx::Int, bc::PeriodicData{Tg, :exclusive}) where {Tg}
    # ScalarSpacing (uniform Range): all cells share `step`; seam_h equals step
    # for valid exclusive Range input. Return the spacing accessor — uniform.
    # VectorSpacing: spacing.h has n-1 entries; idx == n is OOB → seam.
    return idx > _periodic_n_real_cells(spacing) ?
        (bc.h_n, inv(bc.h_n)) :
        (_get_h(spacing, idx), _get_inv_h(spacing, idx))
end

# Number of real (spacing-resident) cell widths. For VectorSpacing this is
# `length(spacing.h)`; for ScalarSpacing (uniform), every index is "real" since
# the constant step is well-defined for any idx.
@inline _periodic_n_real_cells(s::ScalarSpacing) = typemax(Int)   # always real (uniform step)
@inline _periodic_n_real_cells(s::VectorSpacing) = length(s.h)

# ========================================
# Extrapolation-aware Evaluation
# ========================================

# Extrapolation value helpers (_eval_extrapolation, _promote_extrap_val/zero)
# are defined in core/utils.jl — shared by all interpolation methods.

# ========================================
# Core eval: extrap dispatch → search → kernel (no intermediate layers)
# ========================================

# NoExtrap / ExtendExtrap / other: direct search + kernel.
# _check_domain(::NoExtrap) throws if OOB; search_interval clamps idx for ExtendExtrap.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, idx_R, xL, xR = search_interval(searcher, x, spacing, xq)
    dL = xq - xL
    dR = xR - xq
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
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
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector,
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_primal = _extract_primal(xq)
    xq_primal < first(x) && return _eval_extrapolation(op, first(y), extrap, xq)
    xq_primal > last(x) && return _eval_extrapolation(op, last(y), extrap, xq)
    idx, idx_R, xL, xR = search_interval(searcher, x, spacing, xq)
    dL = xq - xL
    dR = xR - xq
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds begin
        zL = z[idx]; zR = z[idx_R]
        yL = y[idx]; yR = y[idx_R]
    end
    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end

# WrapExtrap: wrap query to domain → search + kernel.
# 4-arg `_wrap_to_domain` dispatches on typed vs Nothing WrapExtrap.
@inline function _eval_cubic_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        spacing::AbstractGridSpacing{Tg},
        z::AbstractVector,
        xq::Tq,
        extrap::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, extrap)
    idx, idx_R, xL, xR = search_interval(searcher, x, spacing, xq_wrapped)
    dL = xq_wrapped - xL
    dR = xR - xq_wrapped
    h = _get_h(spacing, idx)
    inv_h = _get_inv_h(spacing, idx)
    @inbounds begin
        zL = z[idx]; zR = z[idx_R]
        yL = y[idx]; yR = y[idx_R]
    end
    return _cubic_kernel(op, zL, zR, yL, yR, h, inv_h, dL, dR)
end


# ========================================
# BC-Aware Evaluation Helper
# ========================================

# --- Searcher-aware versions ---

"Evaluate with BC-aware dispatch (Periodic BC) with search policy."
@inline function _eval_with_bc(
        cache::CubicSplineCache{Tg, X, F, <:PeriodicData{Tg}, S},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        ::AbstractExtrap,  # extrapolation ignored for periodic
        op::O,
        searcher::P
    ) where {Tg, Tv, Tq, X, F, S <: AbstractGridSpacing{Tg}, O <: AbstractEvalOp, P <: Searcher}
    return _eval_cubic_at_point_periodic(cache.x, y, cache.spacing, z, xq, cache.bc_config, op, searcher)
end

"Evaluate with BC-aware dispatch (Generic Derivative BC) with search policy."
@inline function _eval_with_bc(
        cache::CubicSplineCache{Tg, X, F, BCPair{L, R}, S},
        y::AbstractVector{Tv},
        z::AbstractVector,
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::P
    ) where {Tg, Tv, Tq, X, F, L <: PointBC, R <: PointBC, S <: AbstractGridSpacing{Tg}, O <: AbstractEvalOp, P <: Searcher}
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
        z::AbstractVector,
        x_query::AbstractVector{<:Real},
        ev::E,
        op::O,
        searcher::P
    ) where {Tg, Tv, X, F, BC, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    ev = _check_domain(cache.x, x_query, ev)
    return @inbounds for k in eachindex(x_query, output)
        output[k] = _eval_with_bc(cache, y, z, x_query[k], ev, op, searcher)
    end
end

"Vector loop for Periodic BC. Accepts any Real query type."
@inline function _cubic_vector_loop!(
        output::AbstractVector,
        cache::CubicSplineCache{Tg, X, F, <:PeriodicData{Tg}, S},
        y::AbstractVector{Tv},
        z::AbstractVector,
        x_query::AbstractVector{<:Real},
        ::AbstractExtrap,  # extrap ignored for periodic
        op::O,
        searcher::P
    ) where {Tg, Tv, X, F, S <: AbstractGridSpacing{Tg}, O <: AbstractEvalOp, P <: Searcher}
    # Use the periodic kernel uniformly: it handles BC-aware seam cell widths
    # via `bc_config` (essential for `:exclusive` Vector grids where the spacing
    # only carries n-1 cells and the seam cell width is in `bc_config.h_n`).
    # `_wrap_to_domain` is a no-op when queries are already in-domain, so the
    # previous "fast path" optimization (skipping wrap for in-domain queries)
    # had no measurable benefit and is removed to keep the seam-cell handling
    # in one place.
    bc_config = cache.bc_config
    return @inbounds for k in eachindex(x_query, output)
        output[k] = _eval_cubic_at_point_periodic(cache.x, y, cache.spacing, z, x_query[k], bc_config, op, searcher)
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
        x_query::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real, X, F, BC, S <: AbstractGridSpacing{Tg}}
    @assert length(y) == length(cache.x) "y length must match cache grid"

    Tz = _output_eltype(Tv, eltype(cache.x))
    z = acquire!(pool, Tz, length(y))
    _solve_system!(z, cache, y, cache.bc_config)

    searcher = _resolve_search(cache.x, x_query, search, hint)
    extrap_eff = _resolve_extrap(extrap, cache.x)
    @boundscheck _check_domain(cache.x, x_query, extrap_eff)
    _eval_with_bc(cache, y, z, x_query, extrap_eff, deriv, searcher)
end
