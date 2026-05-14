# ========================================
# Quadratic Interpolation Oneshot API
# ========================================
# Zero-allocation quadratic spline interpolation functions.
# QuadraticBC type alias and _compute_quadratic_coeffs are in quadratic_solver.jl.
# Type definitions in quadratic_types.jl.
# Callable methods (2-arg form) in quadratic_interpolant.jl.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    QUADRATIC SPLINE INTERPOLATION API                     ║
# ║              C1 piecewise quadratic with single-endpoint BC               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Internal Evaluation Functions
# ========================================

# Note: _eval_extrapolation helper is defined in core/utils.jl (shared)

# ========================================
# Core eval: extrap dispatch → search → kernel (no intermediate layers)
# ========================================

# NoExtrap / ExtendExtrap / other: direct search + kernel.
# _check_domain(::NoExtrap) throws if OOB; search_interval clamps idx for ExtendExtrap.
@inline function _quadratic_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        a::AbstractVector{Tc},
        d::AbstractVector{Tc},
        xq::Tq,
        extrap::AbstractExtrap,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tc, Tq, S <: Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, _, xL, _ = search_interval(searcher, x, xq)
    dt = xq - xL  # Can be Dual for AD
    @inbounds return _quadratic_kernel(op, a[idx], d[idx], y[idx], dt)
end

# ClampExtrap / FillExtrap: boundary check → extrap value or kernel.
@inline function _quadratic_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        a::AbstractVector{Tc},
        d::AbstractVector{Tc},
        xq::Tq,
        extrap::_ClampOrFill,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tc, Tq, S <: Searcher}
    xq_primal = _extract_primal(xq)
    xq_primal < _extract_primal(first(x)) && return _eval_extrapolation(op, first(y), extrap, xq)
    xq_primal > _extract_primal(last(x)) && return _eval_extrapolation(op, last(y), extrap, xq)
    idx, _, xL, _ = search_interval(searcher, x, xq)
    dt = xq - xL
    @inbounds return _quadratic_kernel(op, a[idx], d[idx], y[idx], dt)
end

# WrapExtrap: wrap query to domain → search + kernel.
# Wrap domain `[first(x), last(x))` is read directly from the axis.
@inline function _quadratic_eval_at_point(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        a::AbstractVector{Tc},
        d::AbstractVector{Tc},
        xq::Tq,
        extrap::WrapExtrap,
        op::AbstractEvalOp,
        searcher::S
    ) where {Tg, Tv, Tc, Tq, S <: Searcher}
    xq_wrapped = _wrap_to_domain(xq, x)
    idx, _, xL, _ = search_interval(searcher, x, xq_wrapped)
    dt = xq_wrapped - xL
    @inbounds return _quadratic_kernel(op, a[idx], d[idx], y[idx], dt)
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ║                 TYPED CORE — all grid types                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar interpolation
# ========================================

"""
    quadratic_interp(x, y, xi; bc=Left(QuadraticFit()), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

C1 piecewise quadratic spline interpolation at a single point.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (same length as x)
- `xi::Real`: Query point
- `bc`: Boundary condition (one of):
  - `Left(QuadraticFit())`: 3-point parabola fit at left (default, exact for polynomials)
  - `Right(QuadraticFit())`: 3-point parabola fit at right
  - `Left(Deriv1(v))`: First derivative = v at left endpoint
  - `Left(Deriv2(v))`: Second derivative = v at left endpoint
  - `Right(Deriv1(v))`: First derivative = v at right endpoint
  - `Right(Deriv2(v))`: Second derivative = v at right endpoint
  - `MinCurvFit()`: Minimize total curvature (globally smooth)
- `extrap::AbstractExtrap`: Extrapolation mode
  - `NoExtrap()` (default): throws DomainError if outside domain
  - `ClampExtrap()`: clamp to boundary values
  - `ExtendExtrap()`: extend the boundary polynomial
- `deriv::DerivOp`: Derivative order -- use `EvalValue()` (default), `DerivOp(1)`, or `DerivOp(2)`
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `BinarySearch()` (default): O(log n) binary search, stateless
  - `LinearBinarySearch(linear_window=0)`: O(1) if hint valid, O(log n) fallback
  - `LinearBinarySearch(linear_window=8)`: Linear search within window, then binary fallback

# Returns
- Interpolated value (Float type)

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = x.^2  # [0, 1, 4, 9]

# Default: QuadraticFit (exact for quadratic polynomials)
quadratic_interp(x, y, 1.5)  # ≈ 2.25 (exact)

# With specific BC
quadratic_interp(x, y, 1.5; bc=Left(Deriv1(0.0)))  # zero slope at left
quadratic_interp(x, y, 1.5; bc=MinCurvFit())        # minimize curvature

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = quadratic_interp(x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
@inline @with_pool pool function quadratic_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::Tq;  # Accepts Tg, Real, or Dual for AD (Dual <: Real)
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))
    @boundscheck length(x) >= 2 || throw(ArgumentError("x must have at least 2 elements"))

    x = _resolve_axis(x)
    # Compute coefficients using temporary arrays from pool. The grid `x`
    # carries cached `h`/`inv_h` when wrapped, or computes them on the fly.
    nx = length(x)
    Tcoeff = _output_eltype(eltype(y), eltype(x))
    d = acquire!(pool, Tcoeff, nx)
    a = acquire!(pool, Tcoeff, nx - 1)
    bc_promoted = _normalize_bc(bc, first(y))
    _compute_quadratic_coeffs!(d, a, x, y, bc_promoted)

    searcher = _resolve_search(x, xq, search, hint)
    # Materialize WrapExtrap{Nothing} against the grid before reaching the kernel.
    extrap_eff = _resolve_extrap(extrap, x)
    _quadratic_eval_at_point(x, y, a, d, xq, extrap_eff, deriv, searcher)
end

# ========================================
# Vector interpolation (in-place)
# ========================================

"""
    quadratic_interp!(output, x, y, x_targets; bc=Left(QuadraticFit()), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

In-place quadratic spline interpolation for multiple query points.

# Arguments
- `output`: Pre-allocated output vector
- Other arguments same as `quadratic_interp`

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = x.^2
out = zeros(3)
quadratic_interp!(out, x, y, [0.5, 1.5, 2.5])
# out ≈ [0.25, 2.25, 6.25]

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
output = zeros(1000)
quadratic_interp!(output, x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
@with_pool pool function quadratic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        x_targets::AbstractVector{Tq};
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv, Tq <: Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"
    @assert length(x) >= 2 "x must have at least 2 elements"

    x = _resolve_axis(x)
    # Compute coefficients using temporary arrays from pool. The grid `x`
    # carries cached `h`/`inv_h` when wrapped, or computes them on the fly.
    nx = length(x)
    Tcoeff = _output_eltype(eltype(y), eltype(x))
    d = acquire!(pool, Tcoeff, nx)
    a = acquire!(pool, Tcoeff, nx - 1)
    bc_promoted = _normalize_bc(bc, first(y))
    _compute_quadratic_coeffs!(d, a, x, y, bc_promoted)

    searcher = _resolve_search(x, x_targets, search, nothing)
    extrap_eff = _resolve_extrap(extrap, x)
    _quadratic_vector_loop!(output, x, y, a, d, x_targets, extrap_eff, deriv, searcher)
    return output
end

# ========================================
# Vector interpolation (allocating)
# ========================================

"""
    quadratic_interp(x, y, x_targets; bc=Left(QuadraticFit()), extrap=NoExtrap(), deriv=EvalValue(), search=AutoSearch())

Quadratic spline interpolation for multiple query points (allocating version).

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = x.^2
result = quadratic_interp(x, y, [0.5, 1.5, 2.5])
# result ≈ [0.25, 2.25, 6.25]

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = quadratic_interp(x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
function quadratic_interp(
        x::AbstractVector{Tg},
        y::AbstractVector,
        x_targets::AbstractVector{Tq};
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    Tr = _output_eltype(eltype(y), _promote_grid_float(Tg, eltype(y)), Tq)
    output = Vector{Tr}(undef, length(x_targets))
    quadratic_interp!(output, x, y, x_targets; bc, extrap, deriv, search)
    return output
end
