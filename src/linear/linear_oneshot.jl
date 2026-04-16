# ========================================
# Linear Interpolation Oneshot API
# ========================================
# Zero-allocation linear interpolation functions.
# Type definitions in linear_types.jl.
# Callable methods (2-arg form) in linear_interpolant.jl.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH - OPTIMIZED CORE                         ║
# ║                  All arguments have same FT<:AbstractFloat                ║
# ║                      Zero type conversion overhead                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# PeriodicBC 1D dispatch uses the shared `_periodic_extend_1d_pooled!` helper
# from core/periodic.jl (no method-specific logic needed — Linear has no
# coefficient system, so extension alone suffices).

# ========================================
# Vector interpolation (in-place, zero-allocation)
# ========================================

"""
    linear_interp!(output, x, y, x_targets; bc=NoBC(), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Zero-allocation linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: Search algorithm determined by `search` parameter

# Arguments
- `output`: Pre-allocated output vector (must be floating-point type)
- `bc::AbstractBC`: Boundary condition. Default `NoBC()` (no BC). Pass
  `PeriodicBC(endpoint=:inclusive)` or `PeriodicBC(endpoint=:exclusive, period=L)`
  for periodic interpolation (extrap is forced to `WrapExtrap()` in that case).
- `extrap::AbstractExtrap`: `NoExtrap()` (default, throws DomainError), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `deriv::DerivOp`: Derivative order (`EvalValue()` default, `DerivOp(1)` first derivative, `DerivOp(2)` second derivative)
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `BinarySearch()` (default): O(log n) binary search, stateless
  - `LinearBinarySearch(linear_window=0)`: O(1) if hint valid, O(log n) fallback
  - `LinearBinarySearch(linear_window=8)`: Linear search within window, then binary fallback

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
out = Vector{Float64}(undef, 2)
linear_interp!(out, rho, y, [0.55, 0.77])  # throws error if outside domain
linear_interp!(out, rho, y, [-0.1, 1.2]; extrap=ExtendExtrap())  # linear extrapolation

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
output = zeros(1000)
linear_interp!(output, x_vec, y_vec, sorted_queries; search=LinearBinarySearch(linear_window=8))
```

# Implementation Note
- Optimized core works with `AbstractFloat` types (calls optimized scalar version)
- Integer/Real inputs automatically promoted via wrapper methods
"""
function linear_interp! end

# Unified method for AbstractVector
# Unified in-place entry point. Handles promotion internally via _promote_itp_inputs,
# so no separate Real/Mixed-type wrapper is needed (same pattern as the scalar API).
#
# Hot-path (non-periodic) stays pool-free — `@with_pool` is hoisted into the
# periodic helper below. Directly wrapping the whole entry point added ~5-25 ns
# per call on small queries even when the pool was unused, a measurable
# regression vs master on the `4_linear_oneshot/q00001` benchmark.
function linear_interp!(
        output::AbstractVector,
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AbstractVector;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    x_typed = _prepare_grid(x)
    if _is_periodic_bc(bc)
        return _linear_interp_periodic_vector!(output, x_typed, y, x_targets, bc, extrap, deriv, search)
    end
    searcher = _resolve_search(x_typed, x_targets, search, nothing)
    return _linear_interp_loop!(output, x_typed, y, x_targets, extrap, deriv, searcher)
end

# Pool-scoped helper for the periodic oneshot vector path. Isolated so the
# common non-periodic path above doesn't pay the `@with_pool` overhead.
@inline @with_pool pool function _linear_interp_periodic_vector!(
        output, x, y, x_targets, bc, extrap, deriv, search
    )
    x_eff, y_eff, extrap_eff = _periodic_extend_1d_pooled!(pool, x, y, bc, extrap)
    searcher = _resolve_search(x_eff, x_targets, search, nothing)
    return _linear_interp_loop!(output, x_eff, y_eff, x_targets, extrap_eff, deriv, searcher)
end

# Internal loop with AbstractExtrap dispatch and Searcher (type-stable)
# Supports mixed types: Tg for grid, Tv for values
@inline function _linear_interp_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_targets::AbstractVector,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, O <: AbstractEvalOp, S <: Searcher}
    extrap = _check_domain(x, x_targets, extrap)
    @inbounds for i in eachindex(x_targets, output)
        output[i] = _linear_eval_at_point(x, y, x_targets[i], extrap, op, searcher)
    end
    return output
end


# Optimized loop for WrapExtrap - uses 2-stage strategy
# Stage 1: Check if ALL queries are inside domain (cheap: ~150ns for 1000 elements)
# Stage 2: If all inside, use extension path (no wrap needed); otherwise per-element wrap
@inline function _linear_interp_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_targets::AbstractVector,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, O <: AbstractEvalOp, S <: Searcher}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(x_targets), maximum(x_targets)

    if qmin >= x_min && qmax < x_max
        # Fast path: all queries inside domain - use extension (no wrap overhead)
        @inbounds for i in eachindex(x_targets, output)
            output[i] = _linear_eval_at_point(x, y, x_targets[i], ExtendExtrap(), op, searcher)
        end
    else
        # Slow path: some queries outside - per-element wrap
        @inbounds for i in eachindex(x_targets, output)
            xi_wrapped = _wrap_to_domain(x_targets[i], x_min, x_max)
            output[i] = _linear_eval_at_point(x, y, xi_wrapped, ExtendExtrap(), op, searcher)
        end
    end
    return output
end

# AbstractRange WrapExtrap specialization removed — identical logic to the
# AbstractVector version above.  _CachedRange <: AbstractRange <: AbstractVector,
# so the AbstractVector dispatch handles all Range inputs.

# NOTE: the former AbstractRange{Tg}-specific `linear_interp!` overload (which resolved
# ambiguity with the old Real wrappers) has been removed. The unified `linear_interp!`
# above handles promotion for all grid types (Float, Int, Dual, Range) via
# `_promote_itp_inputs`, eliminating the need for type-specific overloads.

# ========================================
# Scalar interpolation (zero-allocation)
# ========================================

"""
    linear_interp(x, y, xq::Real; bc=NoBC(), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch()) -> AbstractFloat

Zero-allocation scalar linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: Search algorithm determined by `search` parameter

# Arguments
- `xq::Real`: Single interpolation query point
- `bc::AbstractBC`: Boundary condition. Default `NoBC()` (no BC). Pass
  `PeriodicBC(endpoint=:inclusive)` or `PeriodicBC(endpoint=:exclusive, period=L)`
  for periodic interpolation (extrap is forced to `WrapExtrap()` in that case).
- `extrap::AbstractExtrap`: `NoExtrap()` (default, throws DomainError), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `deriv::DerivOp`: Derivative order (`EvalValue()` default, `DerivOp(1)` first derivative)
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `BinarySearch()` (default): O(log n) binary search, stateless
  - `LinearBinarySearch(linear_window=0)`: O(1) if hint valid, O(log n) fallback
  - `LinearBinarySearch(linear_window=8)`: Linear search within window, then binary fallback

# Returns
- Always returns a floating-point type (Integer inputs auto-promoted to Float)

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
value = linear_interp(rho, y, 0.55)  # Returns Float64, zero allocation
value = linear_interp(rho, y, 1.5; extrap=WrapExtrap())  # wraps to domain

# Integer inputs auto-promoted to Float
x_int = 0:10
y_int = x_int.^2
value = linear_interp(x_int, y_int, 5.5)  # Returns Float64 (not Int)
```

# Implementation Note
- Optimized core works with `AbstractFloat` types only (zero conversion overhead)
- Integer/Real inputs automatically promoted to Float via wrapper methods
- Uses Val dispatch for extrapolation to eliminate runtime branches
"""

# ========================================
# Internal evaluation with op parameter
# ========================================
#
# TYPE PARAMETERS:
# - Tg: Grid type (AbstractFloat) - for x coordinates
# - Tv: Value type (unconstrained) - for y values
# - Tq: Query type - typically Tg but can be left unconstrained for AD support
#
# These functions form the core evaluation logic that supports:
# - Any value type Tv via duck typing (+, -, scalar *)
# - AD support (xq can be Dual{Tg})

"""
    _linear_eval_at_point(x, y, xq, extrap, op, searcher)

Core linear interpolation evaluation using kernel function and search policy.
Supports value (EvalValue), first derivative (EvalDeriv1), and second derivative (EvalDeriv2).

# Type Parameters
- `Tg`: Grid type (AbstractFloat)
- `Tv`: Value type (unconstrained)
- `Tq`: Query type (can be Tg or Dual{Tg} for AD support)

# AD Support
For ForwardDiff compatibility, `xq` can be a Dual type:
- Search uses `_extract_primal(xq)` to find interval index
- Interpolation arithmetic uses original `xq` to propagate derivatives
"""
# ========================================
# Core eval: extrap dispatch → search → kernel (no intermediate layers)
# ========================================
# _get_inv_h(x, xR, xL) dispatches to x.inv_h (_CachedRange) or inv(xR-xL) (Vector).

# NoExtrap / ExtendExtrap / other: direct search + kernel.
# _check_domain(::NoExtrap) throws if OOB; all others are no-ops.
@inline function _linear_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        extrap::AbstractExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, xL, xR = search_interval(searcher, x, xq)
    dL = xq - xL  # xq can be Dual here (preserves AD)
    @inbounds return _linear_kernel(op, y[idx], y[idx + 1], _get_inv_h(x, xR, xL), dL)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel.
@inline function _linear_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        extrap::_ClampOrFill,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_primal = _extract_primal(xq)
    if xq_primal < first(x)
        return _eval_extrapolation(op, first(y), extrap, xq)
    elseif xq_primal > last(x)
        return _eval_extrapolation(op, last(y), extrap, xq)
    end
    idx, xL, xR = search_interval(searcher, x, xq)
    dL = xq - xL
    @inbounds return _linear_kernel(op, y[idx], y[idx + 1], _get_inv_h(x, xR, xL), dL)
end

# WrapExtrap: wrap query to domain → search + kernel.
# Pass original xq (may be Dual) to _wrap_to_domain to preserve AD derivatives.
@inline function _linear_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq,
        ::WrapExtrap,
        op::O,
        searcher::S
    ) where {Tg, Tv, Tq, O <: AbstractEvalOp, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, first(x), last(x))
    idx, xL, xR = search_interval(searcher, x, xq_wrapped)
    dL = xq_wrapped - xL
    @inbounds return _linear_kernel(op, y[idx], y[idx + 1], _get_inv_h(x, xR, xL), dL)
end

# Public scalar one-shot API.
# Zero-alloc: _prepare_grid returns Vector as-is, Range → _CachedRange (stack).
# Kernel arithmetic auto-promotes Int×Float via _get_h float() wrappers.
@inline function linear_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    x_typed = _prepare_grid(x)
    if _is_periodic_bc(bc)
        return _linear_interp_periodic_scalar(x_typed, y, xq, bc, extrap, deriv, search, hint)
    end
    searcher = _resolve_search(x_typed, xq, search, hint)
    return _linear_eval_at_point(x_typed, y, xq, extrap, deriv, searcher)
end

# Pool-scoped scalar periodic helper — kept out of the hot non-periodic path.
@inline @with_pool pool function _linear_interp_periodic_scalar(
        x, y, xq, bc, extrap, deriv, search, hint
    )
    x_eff, y_eff, extrap_eff = _periodic_extend_1d_pooled!(pool, x, y, bc, extrap)
    searcher = _resolve_search(x_eff, xq, search, hint)
    return _linear_eval_at_point(x_eff, y_eff, xq, extrap_eff, deriv, searcher)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        GENERIC WRAPPERS - CONVENIENCE                     ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ║                     Integer inputs → Float64 outputs                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Vector interpolation - Allocating hot path
# Unified allocating vector one-shot. Calls the unified linear_interp! which
# handles promotion internally. Output type includes Tg for duck grids.

function linear_interp(
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AbstractVector;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    Tg = _promote_grid_float(eltype(x), eltype(y))
    T_out = _output_eltype(eltype(y), Tg)
    output = Vector{T_out}(undef, length(x_targets))
    linear_interp!(output, x, y, x_targets; bc, extrap, deriv, search)
    return output
end

# NOTE: the former in-place and allocating vector wrappers (`linear_interp!` and
# `linear_interp` with `{Tg<:Real, Tv, Tq<:Real}`) have been removed. Their
# `_promote_itp_inputs` calls are now performed by the unified `linear_interp!`
# above and the typed `linear_interp(x, y, x_targets)` below. This prevents
# dispatch ambiguity and infinite recursion on duck grids.
