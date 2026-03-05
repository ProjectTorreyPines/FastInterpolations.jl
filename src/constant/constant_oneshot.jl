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
# Internal Evaluation Functions
# ========================================

"""
    _constant_eval_extrap(y, xi, x_min, x_max, extrap, side, op)

Handle extrapolation for constant interpolation.
- EvalValue: returns boundary value (y[1] or y[end])
- EvalDeriv1/EvalDeriv2: returns zero (constant function)

Note: NoExtrap and WrapExtrap modes never reach this function.
- NoExtrap throws DomainError via _check_domain
- WrapExtrap is handled in _constant_eval_at_point before extrap check

Type parameters:
- Tg: Grid type (AbstractFloat) for xi, x_min, x_max
- Tv: Value type (unconstrained)
"""
@inline function _constant_eval_extrap(
    y::AbstractVector{Tv}, xi::Tg, x_min::Tg, x_max::Tg,
    extrap::_ClampOrFill, ::AbstractSide, op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    y_bnd = xi < x_min ? @inbounds(y[1]) : @inbounds(y[end])
    return _constant_extrap_result(op, y_bnd, extrap)
end

# ExtendExtrap delegates to ClampedExtrap (slope=0 for constant function)
@inline function _constant_eval_extrap(
    y::AbstractVector{Tv}, xi::Tg, x_min::Tg, x_max::Tg,
    ::ExtendExtrap, side::AbstractSide, op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    return _constant_eval_extrap(y, xi, x_min, x_max, ClampedExtrap(), side, op)
end


"""
    _constant_eval_at_point(x, y, xi, extrap, side, op, searcher)

Core constant interpolation evaluation with search policy.

Evaluation flow:
1. Domain check (NoExtrap → DomainError if outside)
2. WrapExtrap → wrap to [x_min, x_max) and evaluate
3. Boundary check (xi == x_max → y[end] for non-wrap modes)
4. Extrapolation check (xi outside domain → extrap handling)
5. Interval search → kernel evaluation

Type parameters:
- Tg: Grid type (AbstractFloat) for x
- Tv: Value type (unconstrained)

AD Support:
- xi can be any Real (including ForwardDiff.Dual)
- Comparisons use _extract_primal(xi) for Float value
- Original xi is passed to kernel for AD compatibility
"""
@inline function _constant_eval_at_point(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    xi::Tq,
    extrap::AbstractExtrap,
    side::AbstractSide,
    op::AbstractEvalOp,
    searcher::S
) where {Tg<:AbstractFloat, Tv, Tq<:Real, S<:Searcher}
    # Extract primal for comparisons and search (supports ForwardDiff.Dual)
    xi_primal = _extract_primal(xi)
    xi_typed = Tg(xi_primal)

    # Domain check for NoExtrap mode (throws DomainError)
    @boundscheck _check_domain(x, xi_typed, extrap)

    x_min, x_max = first(x), last(x)

    # WrapExtrap mode handles all cases (inside and outside domain)
    # For WrapExtrap, the query is wrapped to domain, so use wrapped position for dL
    # (AD derivative is zero for constant interp regardless)
    if extrap isa WrapExtrap
        xi_wrapped = _wrap_to_domain(xi_typed, x_min, x_max)
        idx, xL, xR = search_interval(searcher, x, xi_wrapped)
        h = xR - xL
        dL = xi_wrapped - xL  # Use wrapped position for correct interval calculation
        @inbounds return _constant_kernel(op, y[idx], y[idx+1], h, dL, side)
    end

    # Boundary special case: xi == x[end] → y[end] directly
    # (avoids _search_interval returning idx=n-1, dL=h)
    if xi_typed == x_max
        return op isa EvalValue ? (@inbounds y[end]) : 0 * first(y)
    end

    # Extrapolation handling (ClampedExtrap, ExtendExtrap)
    if xi_typed < x_min || xi_typed > x_max
        return _constant_eval_extrap(y, xi_typed, x_min, x_max, extrap, side, op)
    end

    # Normal case: interval search and kernel evaluation
    idx, xL, xR = search_interval(searcher, x, xi_typed)
    h = xR - xL
    dL = xi - xL  # Use original xi for AD
    @inbounds return _constant_kernel(op, y[idx], y[idx+1], h, dL, side)
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ║                 Tg<:AbstractFloat grid, Tv value type                     ║
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
  - `ClampedExtrap()`: clamp to boundary values
  - `ExtendExtrap()`: same as ClampedExtrap (slope=0)
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
constant_interp(x, y, -1.0; extrap=ClampedExtrap()) # 10.0 (clamped)

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = constant_interp(x, y, sorted_queries; search=LinearBinarySearch(linear_window=8))
```
"""
# AD Support: xi can be any Real (including ForwardDiff.Dual)
# Note: Tq<:Real constraint resolves method ambiguity with generic Real wrapper
@inline function constant_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    xi::Tq;
    extrap::AbstractExtrap=NoExtrap(),
    side::AbstractSide=NearestSide(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    searcher = _resolve_search(x, xi, search, hint)
    _constant_eval_at_point(x, y, xi, extrap, side, deriv, searcher)
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
function constant_interp!(
    output::AbstractVector{Tv},
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tg};
    extrap::AbstractExtrap=NoExtrap(),
    side::AbstractSide=NearestSide(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:AbstractFloat, Tv}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    searcher = _resolve_search(x, x_targets, search, nothing)
    @boundscheck _check_domain(x, x_targets, extrap)
    _constant_vector_loop!(output, x, y, x_targets, extrap, side, deriv, searcher)
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
function constant_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tg};
    extrap::AbstractExtrap=NoExtrap(),
    side::AbstractSide=NearestSide(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:AbstractFloat, Tv}
    output = Vector{Tv}(undef, length(x_targets))
    constant_interp!(output, x, y, x_targets; extrap, side, deriv, search)
    return output
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GENERIC WRAPPERS - CONVENIENCE                        ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar → typed wrappers
# ========================================
# Unified wrapper for non-AbstractFloat inputs (Int, mixed types, Complex, etc.)
# POLICY: Tg is computed from x/y ONLY, not from xq
# AD Support: Pass xi directly without Tg conversion to preserve Dual type

@inline function constant_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    xi::Tq;
    extrap::AbstractExtrap=NoExtrap(),
    side::AbstractSide=NearestSide(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:Real, Tv, Tq<:Real}
    x_typed, y_typed = _promote_itp_inputs(x, y)
    # Pass xi directly (not converted) to preserve ForwardDiff.Dual for AD
    return constant_interp(x_typed, y_typed, xi; extrap, side, deriv, search, hint)
end

# ========================================
# Vector → typed wrappers (allocating)
# ========================================
# POLICY: Tg is computed from x/y ONLY, not from x_targets

function constant_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tq};
    extrap::AbstractExtrap=NoExtrap(),
    side::AbstractSide=NearestSide(),
    deriv::DerivOp=EvalValue(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:Real, Tv, Tq<:Real}
    x_typed, y_typed, xq_typed = _promote_itp_inputs(x, y, x_targets)
    Tv_float = eltype(y_typed)
    output = Vector{Tv_float}(undef, length(x_targets))
    constant_interp!(output, x_typed, y_typed, xq_typed; extrap, side, deriv, search)
    return output
end

# ========================================
# Vector → typed wrappers (in-place)
# ========================================

function constant_interp!(
    output::AbstractVector,
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tq};
    extrap::AbstractExtrap=NoExtrap(),
    side::AbstractSide=NearestSide(),
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

    constant_interp!(output, x_typed, y_typed, xq_typed; extrap, side, deriv, search)
end
