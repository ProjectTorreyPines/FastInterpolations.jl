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
# _get_h(x, xL, xR) dispatches to x.h (_CachedRange) or xR-xL (Vector).

# Core in-bounds path: `last(x) == xi` seam handling + search + kernel.
# All non-InBounds extrap overloads delegate here after preprocessing — see
# cubic_eval.jl for the same `InBounds = core fast path` pattern. WrapExtrap
# uses a separate periodic-seam-aware core (`_raw(y)`) below.
@inline function _constant_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq,
        ::InBounds,
        side::AbstractSide,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, S <: Searcher}
    if _extract_primal(xi) == _extract_primal(last(x))
        # `last(y)` covers both raw vectors and `_ExclusivePeriodicData` (cyclic
        # `inner[1]`). `* one(xi)` propagates Tq carrier to match the kernel
        # path below (without it, Int y + Float xq returns Union{Int,Float}).
        return op isa EvalValue ? last(y) * one(xi) : 0 * first(y) * one(xi)
    end
    idx, idx_R, xL, xR = search_interval(searcher, x, xi)
    dL = xi - xL
    @inbounds return _constant_kernel(op, y[idx], y[idx_R], _get_h(x, idx, xL, xR), dL, side)
end

# NoExtrap / others matching AbstractExtrap: domain check → delegate.
@inline function _constant_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq,
        extrap::AbstractExtrap,
        side::AbstractSide,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, S <: Searcher}
    @boundscheck _check_domain(x, xi, extrap)
    return _constant_eval_at_point(x, y, xi, InBounds(), side, op, searcher)
end

# ExtendExtrap: constant has zero slope → extend = clamp. Route through
# ClampExtrap so OOB queries return the boundary value (in-bounds case
# eventually reaches the InBounds core).
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

# ClampExtrap / FillExtrap: boundary check → extrap value or delegate.
@inline function _constant_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq,
        extrap::_ClampOrFill,
        side::AbstractSide,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, S <: Searcher}
    xi_primal = _extract_primal(xi)
    xi_primal < _extract_primal(first(x)) && return _eval_extrapolation(op, first(y), extrap, xi)
    xi_primal > _extract_primal(last(x)) && return _eval_extrapolation(op, last(y), extrap, xi)
    return _constant_eval_at_point(x, y, xi, InBounds(), side, op, searcher)
end

# WrapExtrap: wrap query to domain → right-edge short-circuit → search + kernel.
# `_wrap_to_domain(xi, x, ::WrapExtrap)` reads `(first(x), last(x))` from the
# axis — `_ExclusivePeriodicAxis` exposes the precomputed virtual endpoint via
# `last(g)`, so the wrap domain naturally spans one period for `:exclusive`.
@inline function _constant_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq,
        extrap::WrapExtrap,
        side::AbstractSide,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tq <: Real, S <: Searcher}
    xi_wrapped = _wrap_to_domain(xi, x)
    # Right-edge short-circuit (closed-domain): `xi == last(x)` collapses
    # uniformly to `last(y)`, bypassing side semantics. Mirrors the InBounds
    # core's identical guard and the persistent anchor path's `aq.xq == x_last`
    # short-circuit so scalar oneshot agrees with the persistent interpolant at
    # the exact boundary. `last(_ExclusivePeriodicData) = inner[1]` so `:exclusive`
    # cyclic wrap is preserved; raw Vector yields `y[n]`.
    _extract_primal(xi_wrapped) == _extract_primal(last(x)) &&
        return op isa EvalValue ? last(y) * one(xi_wrapped) : 0 * first(y) * one(xi_wrapped)
    idx, idx_R, xL, xR = search_interval(searcher, x, xi_wrapped)
    dL = xi_wrapped - xL
    # Unwrap data once: `search_interval` already resolved the seam (idx_R = 1
    # at seam cell), so direct inner access skips the wrapper's cyclic
    # `Base.getindex` branch on each `y[idx]` / `y[idx_R]`. The 4-arg
    # `_get_h(x, idx, xL, xR)` is wrapper- and cache-aware.
    yi = _raw(y)
    @inbounds return _constant_kernel(op, yi[idx], yi[idx_R], _get_h(x, idx, xL, xR), dL, side)
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ║                  Generic grid (supports duck types), Tv value type         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar interpolation
# ========================================

"""
    constant_interp(x, y, xi; bc=NoBC(), side=NearestSide(), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Constant (step/piecewise constant) interpolation at a single point.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (same length as x)
- `xi::Real`: Query point
- `bc::AbstractBC`: Boundary condition. Default `NoBC()` (no BC). Pass
  `PeriodicBC(endpoint=:inclusive)` or `PeriodicBC(endpoint=:exclusive, period=L)`
  for periodic interpolation (extrap is forced to `WrapExtrap()` in that case).
- `extrap::AbstractExtrap`: Extrapolation mode
  - `NoExtrap()` (default): throws DomainError if outside domain
  - `ClampExtrap()`: clamp to boundary values
  - `ExtendExtrap()`: same as ClampExtrap (slope=0)
  - `WrapExtrap()`: wrap to closed domain [x_min, x_max] (xq==x_max returns y[end])
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
- Interpolated value, eltype `promote_type(eltype(y), eltype(xq))` (the
  kernel's `* one(dL)` carrier propagation; fully-Int chain preserves Int).

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
# Zero-alloc: _resolve_axis returns Vector as-is, Range → _CachedRange (stack).
# Selection kernel (no `inv_h * dL` arithmetic) — raw Int grids pass through.
@inline function constant_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xi::Tq;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    # Surface-level BC-aware resolvers (zero-alloc reference wrapping). BC info
    # lives in axis type after resolution → searcher uses `NoBC()`.
    x_eff = _resolve_axis(x, bc)
    y_eff = _resolve_data(y, bc)
    extrap_eff = _resolve_extrap(extrap, bc, x_eff, y_eff)
    searcher = _resolve_search(x_eff, xi, search, hint)
    return _constant_eval_at_point(x_eff, y_eff, xi, extrap_eff, side, deriv, searcher)
end

# ========================================
# Vector interpolation (in-place)
# ========================================

"""
    constant_interp!(output, x, y, x_targets; bc=NoBC(), side=NearestSide(), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Zero-allocation constant interpolation for multiple query points.

# Arguments
- `output`: Pre-allocated output vector
- `x, y, x_targets`: Grid and query points
- `bc, extrap, side, deriv`: Same as `constant_interp`
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
# Unified in-place entry. Resolvers normalize inputs; selection kernel
# preserves `eltype(y)`. No Real/Mixed wrapper needed.
function constant_interp!(
        output::AbstractVector,
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AbstractVector;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    # Surface-level BC-aware resolvers (same template as Linear oneshot).
    x_eff = _resolve_axis(x, bc)
    y_eff = _resolve_data(y, bc)
    extrap_eff = _resolve_extrap(extrap, bc, x_eff, y_eff)
    searcher = _resolve_search(x_eff, x_targets, search, nothing)
    _constant_vector_loop!(output, x_eff, y_eff, x_targets, extrap_eff, side, deriv, searcher)
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
# Buffer eltype via Constant's kernel shape — Julia infers the return type
# from `_constant_kernel_shape(xL, yv, xq) = yv * one(xq - xL)`, matching the
# actual kernel reality (Int×Int×Int → Int; SVector × Dual → SVector{Dual};
# Float y × Dual grid → Dual carrier via `xq - xL`).
function constant_interp(
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AbstractVector;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    output = Vector{_output_eltype(_constant_kernel_shape, eltype(x), eltype(y), eltype(x_targets))}(
        undef, length(x_targets)
    )
    constant_interp!(output, x, y, x_targets; bc, extrap, side, deriv, search)
    return output
end
