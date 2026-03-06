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

# Note: _constant_extrap_result helper is defined in cubic_eval.jl (shared)

"""
    _quadratic_eval_core(x, y, a, d, xq, op, searcher)

Core quadratic spline evaluation at a single point with search policy.
Uses interval clamping for extension extrapolation (matches cubic pattern).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type for x
- `Tq`: Query type (can be Tg or ForwardDiff.Dual for AD)
- `Tv`: Value type for y, a, d (unconstrained)
"""
@inline function _quadratic_eval_core(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    xq::Tq,
    op::AbstractEvalOp,
    searcher::S
) where {Tg<:AbstractFloat, Tv, Tq, S<:Searcher}
    # search_interval clamps idx to [1, n-1]
    # This handles both normal evaluation and extension extrapolation
    idx, xL, _ = search_interval(searcher, x, xq)
    # Use original xq for arithmetic to preserve AD
    dt = xq - xL  # Can be Dual for AD
    @inbounds return _quadratic_kernel(op, a[idx], d[idx], y[idx], dt)
end

# ========================================
# Extrapolation-aware Evaluation (matches cubic pattern)
# ========================================

"Evaluate with no extrapolation - throws DomainError if outside domain."
@inline function _quadratic_eval_with_extrap(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    xq::Tq,
    ::NoExtrap,
    op::AbstractEvalOp,
    searcher::S
) where {Tg<:AbstractFloat, Tv, Tq, S<:Searcher}
    return _quadratic_eval_core(x, y, a, d, xq, op, searcher)
end

"Evaluate with constant extrapolation - returns boundary values outside domain."
@inline function _quadratic_eval_with_extrap(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    xq::Tq,
    extrap::_ClampOrFill,
    op::AbstractEvalOp,
    searcher::S
) where {Tg<:AbstractFloat, Tv, Tq, S<:Searcher}
    # Use primal for boundary comparisons (Dual needs real value for comparison)
    xq_primal = _extract_primal(xq)
    xq_primal < first(x) && return _constant_extrap_result(op, @inbounds(y[1]), extrap)
    xq_primal > last(x) && return _constant_extrap_result(op, @inbounds(y[end]), extrap)
    return _quadratic_eval_core(x, y, a, d, xq, op, searcher)
end

"Evaluate with extension extrapolation - extends boundary polynomial."
@inline function _quadratic_eval_with_extrap(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    xq::Tq,
    ::ExtendExtrap,
    op::AbstractEvalOp,
    searcher::S
) where {Tg<:AbstractFloat, Tv, Tq, S<:Searcher}
    # Interval clamping in search_interval handles extension
    return _quadratic_eval_core(x, y, a, d, xq, op, searcher)
end

"Evaluate with wrap extrapolation - wraps query to domain using mod()."
@inline function _quadratic_eval_with_extrap(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    xq::Tq,
    ::WrapExtrap,
    op::AbstractEvalOp,
    searcher::S
) where {Tg<:AbstractFloat, Tv, Tq, S<:Searcher}
    # Wrap query to domain, then evaluate with extension
    xq_wrapped = _wrap_to_domain(xq, first(x), last(x))
    return _quadratic_eval_core(x, y, a, d, xq_wrapped, op, searcher)
end

"""
    _quadratic_eval_at_point(x, y, h, a, d, xq, extrap, op, searcher)

Entry point for quadratic spline evaluation with extrapolation dispatch and search policy.
Note: `h` parameter kept for API compatibility but not used (interval info from x).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type for x, h
- `Tq`: Query type (can be Tg or ForwardDiff.Dual for AD)
- `Tv`: Value type for y, a, d (unconstrained)
"""
@inline function _quadratic_eval_at_point(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    ::AbstractVector{Tg},  # h - unused, kept for API compatibility
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    xq::Tq,
    extrap::AbstractExtrap,
    op::AbstractEvalOp,
    searcher::S
) where {Tg<:AbstractFloat, Tv, Tq, S<:Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    return _quadratic_eval_with_extrap(x, y, a, d, xq, extrap, op, searcher)
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ║                 Tg<:AbstractFloat grid, Tv value type                     ║
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
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))
    @boundscheck length(x) >= 2 || throw(ArgumentError("x must have at least 2 elements"))

    # Compute coefficients using temporary arrays from pool
    # h is grid-typed (Tg), d and a are value-typed (Tv)
    nx = length(x)
    h = acquire!(pool, Tg, nx-1)
    d = acquire!(pool, Tv, nx)
    a = acquire!(pool, Tv, nx-1)
    bc_promoted = _normalize_bc(bc, first(y))
    _compute_quadratic_coeffs!(h, d, a, x, y, bc_promoted)

    searcher = _resolve_search(x, xq, search, hint)
    _quadratic_eval_at_point(x, y, h, a, d, xq, extrap, deriv, searcher)
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
    output::AbstractVector{Tv},
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tg};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:AbstractFloat, Tv}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"
    @assert length(x) >= 2 "x must have at least 2 elements"

    # Compute coefficients using temporary arrays from pool
    # h is grid-typed (Tg), d and a are value-typed (Tv)
    nx = length(x)
    h = acquire!(pool, Tg, nx-1)
    d = acquire!(pool, Tv, nx)
    a = acquire!(pool, Tv, nx-1)
    bc_promoted = _normalize_bc(bc, first(y))
    _compute_quadratic_coeffs!(h, d, a, x, y, bc_promoted)

    searcher = _resolve_search(x, x_targets, search, nothing)
    _quadratic_vector_loop!(output, x, y, h, a, d, x_targets, extrap, deriv, searcher)
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
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tg};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:AbstractFloat, Tv}
    output = Vector{Tv}(undef, length(x_targets))
    quadratic_interp!(output, x, y, x_targets; bc, extrap, deriv, search)
    return output
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GENERIC WRAPPERS - CONVENIENCE                        ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# BC promotion uses _normalize_bc(Left/Right/MinCurvFit, Tv) from core/bc_types.jl

# ========================================
# Scalar → typed wrappers
# ========================================
# Unified wrapper for non-AbstractFloat inputs (Int, mixed types, Complex, etc.)
# POLICY: Tg is computed from x/y ONLY, not from xq

# Wrapper for non-AbstractFloat inputs (Int, mixed types, etc.)
# Preserves original xq type for AD support (Dual types flow through)
@inline function quadratic_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    xq::Tq;  # Accepts Tg, Real, or Dual for AD (Dual <: Real)
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:Real, Tv, Tq<:Real}
    x_typed, y_typed = _promote_itp_inputs(x, y)
    bc_promoted = _normalize_bc(bc, first(y_typed))
    # Pass xq directly to preserve Dual type for AD
    return quadratic_interp(x_typed, y_typed, xq; bc=bc_promoted, extrap, deriv, search, hint)
end

# ========================================
# Vector → typed wrappers (allocating)
# ========================================
# POLICY: Tg is computed from x/y ONLY, not from x_targets

function quadratic_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tq};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:Real, Tv, Tq<:Real}
    x_typed, y_typed, xq_typed = _promote_itp_inputs(x, y, x_targets)
    Tv_float = eltype(y_typed)
    output = Vector{Tv_float}(undef, length(x_targets))
    bc_promoted = _normalize_bc(bc, first(y_typed))
    quadratic_interp!(output, x_typed, y_typed, xq_typed; bc=bc_promoted, extrap, deriv, search)
    return output
end

# ========================================
# Vector → typed wrappers (in-place)
# ========================================

function quadratic_interp!(
    output::AbstractVector,
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tq};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:Real, Tv, Tq<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    x_typed, y_typed, xq_typed = _promote_itp_inputs(x, y, x_targets)
    Tv_float = eltype(y_typed)

    # Validate output can hold result type
    Tout = eltype(output)
    if promote_type(Tout, Tv_float) !== Tout
        throw(ArgumentError(
            "output eltype $Tout cannot hold interpolation result type $Tv_float. " *
            "Use Vector{$Tv_float} or a wider type."
        ))
    end

    bc_promoted = _normalize_bc(bc, first(y_typed))
    quadratic_interp!(output, x_typed, y_typed, xq_typed; bc=bc_promoted, extrap, deriv, search)
end
