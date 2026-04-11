# ========================================
# Constant Interpolation Oneshot API
# ========================================
# Zero-allocation constant interpolation functions.
# Type definitions in constant_types.jl.
# Callable methods (2-arg form) in constant_interpolant.jl.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      CONSTANT (STEP) INTERPOLATION                        ║
# ║              Piecewise constant interpolation with side options           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Core eval: extrap dispatch → search → kernel (no intermediate layers)
# ========================================
# _eval_extrapolation helper is defined in core/utils.jl (shared by all methods).
# _get_h(x, xR, xL) dispatches to x.h (_CachedRange) or xR-xL (Vector).

# NoExtrap / InBounds: domain check + search + kernel.
# (OOB impossible: NoExtrap throws, InBounds guarantees in-domain)
@inline function _constant_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq,
        extrap::AbstractExtrap,
        side::AbstractSide,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, S <: Searcher}
    xi_typed = _to_grid_type(xi, Tg)
    @boundscheck _check_domain(x, xi_typed, extrap)
    if xi_typed == last(x)
        return op isa EvalValue ? last(y) : 0 * first(y)
    end
    idx, xL, xR = search_interval(searcher, x, xi_typed)
    dL = xi - xL
    @inbounds return _constant_kernel(op, y[idx], y[idx + 1], _get_h(x, xR, xL), dL, side)
end

# ExtendExtrap: constant function has zero slope → extend = clamp.
# Cannot use catch-all because kernel is side-dependent and OOB dL > h gives wrong side.
@inline function _constant_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq,
        ::ExtendExtrap,
        side::AbstractSide,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, S <: Searcher}
    return _constant_eval_at_point(x, y, xi, ClampExtrap(), side, op, searcher)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel.
@inline function _constant_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq,
        extrap::_ClampOrFill,
        side::AbstractSide,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, S <: Searcher}
    xi_typed = _to_grid_type(xi, Tg)
    xi_typed < _extract_primal(first(x)) && return _eval_extrapolation(op, first(y), extrap, xi)
    xi_typed > _extract_primal(last(x)) && return _eval_extrapolation(op, last(y), extrap, xi)
    if xi_typed == last(x)
        return op isa EvalValue ? last(y) : 0 * first(y)
    end
    idx, xL, xR = search_interval(searcher, x, xi_typed)
    dL = xi - xL
    @inbounds return _constant_kernel(op, y[idx], y[idx + 1], _get_h(x, xR, xL), dL, side)
end

# WrapExtrap: wrap query to domain → search + kernel.
@inline function _constant_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq,
        ::WrapExtrap,
        side::AbstractSide,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, S <: Searcher}
    xi_typed = _to_grid_type(xi, Tg)
    xi_wrapped = _wrap_to_domain(xi_typed, first(x), last(x))
    idx, xL, xR = search_interval(searcher, x, xi_wrapped)
    dL = xi_wrapped - xL
    @inbounds return _constant_kernel(op, y[idx], y[idx + 1], _get_h(x, xR, xL), dL, side)
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ║                  Generic grid (supports duck types), Tv value type         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar interpolation
# ========================================

"""
    constant_interp(x, y, xi; extrap=NoExtrap(), side=NearestSide(), deriv=EvalValue(), search=AutoSearch())

Constant (step/piecewise constant) interpolation at a single point.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (same length as x)
- `xi::Real`: Query point
- `extrap::AbstractExtrap`: Extrapolation mode
  - `NoExtrap()` (default): throws DomainError if outside domain
  - `ClampExtrap()`: clamp to boundary values
  - `ExtendExtrap()`: same as ClampExtrap (slope=0)
  - `WrapExtrap()`: wrap to [x_min, x_max)
- `side::AbstractSide`: Side selection
  - `NearestSide()` (default): nearest neighbor (left tie-breaking at midpoint)
  - `LeftSide()`: always use left value
  - `RightSide()`: use right value (except at grid points)
- `deriv::DerivOp`: Derivative order (`EvalValue()`, `DerivOp(1)`, or `DerivOp(2)`). Derivatives are always 0.
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `BinarySearch()` (default): O(log n) binary search, stateless
  - `LinearBinarySearch(linear_window=0)`: O(1) if hint valid, O(log n) fallback
  - `LinearBinarySearch(linear_window=8)`: Linear search within window, then binary fallback

# Returns
- Interpolated value (Float type)

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]

constant_interp(x, y, 0.5)                    # 10.0 (nearest to left)
constant_interp(x, y, 0.5; side=LeftSide())    # 10.0
constant_interp(x, y, 0.5; side=RightSide())  # 20.0
constant_interp(x, y, 1.0)                    # 20.0 (grid point)
constant_interp(x, y, -1.0; extrap=ClampExtrap()) # 10.0 (clamped)

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = constant_interp(x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
# Public scalar one-shot API.
# Single unified entry point for every grid eltype. `_promote_itp_inputs` handles
# grid normalization (Range -> _CachedRange, Int -> Float64, Dual -> Dual{Float64},
# etc.) and optional y-value promotion for standard numeric types.
@inline function constant_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq;
        extrap::AbstractExtrap = NoExtrap(),
        side::AbstractSide = NearestSide(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    x_typed, y_typed = _promote_itp_inputs(x, y)
    searcher = _resolve_search(x_typed, xi, search, hint)
    return _constant_eval_at_point(x_typed, y_typed, xi, extrap, side, deriv, searcher)
end

# ========================================
# Vector interpolation (in-place)
# ========================================

"""
    constant_interp!(output, x, y, x_targets; extrap=NoExtrap(), side=NearestSide(), deriv=EvalValue(), search=AutoSearch())

Zero-allocation constant interpolation for multiple query points.

# Arguments
- `output`: Pre-allocated output vector
- `x, y, x_targets`: Grid and query points
- `extrap, side, deriv`: Same as `constant_interp`
- `search::AbstractSearchPolicy`: Search algorithm for interval finding

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]
out = zeros(3)
constant_interp!(out, x, y, [0.5, 1.5, 2.5])
# out == [10.0, 20.0, 30.0]

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
output = zeros(1000)
constant_interp!(output, x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
# Unified in-place entry point. Handles promotion internally via _promote_itp_inputs,
# so no separate Real/Mixed-type wrapper is needed (same pattern as the scalar API).
function constant_interp!(
        output::AbstractVector,
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
        side::AbstractSide = NearestSide(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    x_typed, y_typed, xq_typed = _promote_itp_inputs(x, y, x_targets)
    searcher = _resolve_search(x_typed, xq_typed, search, nothing)
    _constant_vector_loop!(output, x_typed, y_typed, xq_typed, extrap, side, deriv, searcher)
    return output
end

# ========================================
# Vector interpolation (allocating)
# ========================================

"""
    constant_interp(x, y, x_targets; extrap=NoExtrap(), side=NearestSide(), deriv=EvalValue(), search=AutoSearch())

Constant interpolation for multiple query points (allocating version).

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]
result = constant_interp(x, y, [0.5, 1.5, 2.5])
# result == [10.0, 20.0, 30.0]

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = constant_interp(x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
# Unified allocating vector one-shot. Calls the unified constant_interp! which
# handles promotion internally. Output type includes Tg for duck grids.
function constant_interp(
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
        side::AbstractSide = NearestSide(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    x_typed, y_typed, xq_typed = _promote_itp_inputs(x, y, x_targets)
    Tg = eltype(x_typed)
    Tv = eltype(y_typed)
    T_out = _output_eltype(Tv, Tg)
    output = Vector{T_out}(undef, length(x_targets))
    constant_interp!(output, x_typed, y_typed, xq_typed; extrap, side, deriv, search)
    return output
end
