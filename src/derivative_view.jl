# ========================================
# DerivativeView: Wrapper for Broadcast Support
# ========================================
#
# This file contains:
# - DerivativeView struct for derivative evaluation
# - Factory functions for CubicInterpolant and LinearInterpolant
# - Callable methods delegating to parent's deriv keyword
#
# The wrapper enables:
# - Broadcast: deriv1(itp).(xs)
# - HOF composition: map(deriv1(itp), xs)
# - Fused broadcast: @. coef * deriv1(itp)(xs)

# ========================================
# DerivativeView Struct
# ========================================

"""
    AbstractDerivativeView

Abstract supertype for all derivative view wrappers.

Useful for dispatch when you want to accept any derivative view regardless of order or interpolant type.

# Example
```julia
# Accept any derivative view
function integrate(d::AbstractDerivativeView, a, b)
    # works with deriv1, deriv2, deriv3 of any interpolant
end
```

See also: [`DerivativeView`](@ref), [`deriv1`](@ref), [`deriv2`](@ref), [`deriv3`](@ref)
"""
abstract type AbstractDerivativeView end

"""
    DerivativeView{Order, ITP} <: AbstractDerivativeView

Lightweight callable wrapper for derivative evaluation with a fixed derivative order.
Enables broadcast and higher-order function composition.

# Type Parameters
- `Order`: Derivative order (1, 2, or 3)
- `ITP`: Parent interpolant type

# Construction
Use `deriv1(itp)`, `deriv2(itp)`, or `deriv3(itp)` to create instances.

# Type Dispatch Patterns
```julia
# Match any derivative view
function foo(d::AbstractDerivativeView) ... end

# Match specific derivative order (any interpolant)
function foo(d::DerivativeView{1}) ... end

# Match specific order AND interpolant type
function foo(d::DerivativeView{1, <:CubicInterpolant}) ... end
```

# Example
```julia
itp = cubic_interp(x, y)
d1 = deriv1(itp)     # DerivativeView{1, ...}
d2 = deriv2(itp)     # DerivativeView{2, ...}

# Broadcast evaluation
slopes = d1.(query_points)

# Fused broadcast (zero allocation)
result = @. coef * d1(xs) * other_func(xs)
```

See also: [`AbstractDerivativeView`](@ref), [`deriv1`](@ref), [`deriv2`](@ref), [`deriv3`](@ref)
"""
struct DerivativeView{Order, ITP} <: AbstractDerivativeView
    parent::ITP
end

# ========================================
# Factory Functions
# ========================================

"""
    deriv1(itp::AbstractInterpolant)
    deriv2(itp::AbstractInterpolant)
    deriv3(itp::AbstractInterpolant)

Create a callable, zero-allocation derivative view of the interpolant for the 1st, 2nd, or 3rd derivative.

`DerivativeView` is a lightweight wrapper that delegates all evaluation calls to the underlying interpolant
using the `deriv` keyword argument (e.g., `itp(xq; deriv=1)`). This enables a more functional syntax
without the overhead of full object copying.

# Keyword Arguments
All keyword arguments are forwarded to the parent interpolant. Common options include:
- `search`: Search policy for interval lookup (default: parent's `search_policy`)
- `hint::Union{Nothing,Base.RefValue{Int}}`: Mutable hint for sequential access patterns (default: `nothing`)

# Features
- **Functional API**: Pass derivative views into high-order functions like `map` or `find_zeros`.
- **Broadcast Fusion**: Enables evaluation without temporary allocations when combined with other operations using dot syntax (e.g., `@. d1(xs) + d2(xs)`).
- **Unified Interface**: Supports both single-series and multi-series (SeriesInterpolant) objects.
  - For `AbstractInterpolant`, returns a scalar derivative.
  - For `AbstractSeriesInterpolant`, returns a vector of derivatives.
- **In-place Evaluation**: Supports in-place evaluation for both single-series and multi-series interpolants.

# Notes
- **Order 3**: For cubic splines, the 3rd derivative is piecewise constant and may jump at knot points. For lower-order interpolants, it returns zero.

# Example
```julia
itp = cubic_interp(x, y)
d1 = deriv1(itp)

# Evaluate at a point
val = d1(0.5)

# Override search policy
val = d1(0.5; search=LinearBinary())

# Use hint for sequential access
hint = Ref(1)
for xq in sorted_queries
    val = d1(xq; hint=hint)
end

# Use in higher-order functions
using Roots
root = find_zero(d1, 0.5)

# Fused broadcast (no temporary allocations)
query_pts = range(0.0, 1.0, 100)
results = @. itp(query_pts) + 0.1 * d1(query_pts)

# In-place evaluation (single-series)
output = similar(query_pts)
d1(output, query_pts)
d1(output, query_pts; search=LinearBinary())
```
"""
(deriv1, deriv2, deriv3)

@inline deriv1(itp::AbstractInterpolant) = DerivativeView{1, typeof(itp)}(itp)
@inline deriv2(itp::AbstractInterpolant) = DerivativeView{2, typeof(itp)}(itp)
@inline deriv3(itp::AbstractInterpolant) = DerivativeView{3, typeof(itp)}(itp)

# ========================================
# Callable Methods (kwargs forwarding for future-proofing)
# ========================================
# Note: `deriv` keyword is captured and rejected - DerivativeView always uses
# its compile-time Order parameter. This check is zero-cost due to constant folding.

@inline _check_no_deriv_override(::Val, ::Nothing) = nothing
@inline _check_no_deriv_override(::Val{Order}, ::Any) where {Order} = throw(ArgumentError(
    "This DerivativeView already evaluates the $(Order == 1 ? "1st" : Order == 2 ? "2nd" : "3rd") derivative (deriv=$Order). " *
    "The `deriv` keyword argument is not accepted. " *
    "To evaluate a different derivative order, create a new view: deriv1(itp), deriv2(itp), or deriv3(itp)."
))

# Out-of-place calls (Scalar or Vector)
@inline function (d::DerivativeView{Order, ITP})(
    xq::Union{Real, AbstractArray{<:Real}}; deriv=nothing, kwargs...
) where {Order, ITP}
    _check_no_deriv_override(Val(Order), deriv)
    d.parent(xq; deriv=Order, kwargs...)
end

# In-place vector query => vector output (single-series interpolants)
# Note: No element type constraint - parent handles type checking/conversion
@inline function (d::DerivativeView{Order, ITP})(
    output::AbstractVector, xq::AbstractVector{<:Real}; deriv=nothing, kwargs...
) where {Order, ITP}
    _check_no_deriv_override(Val(Order), deriv)
    d.parent(output, xq; deriv=Order, kwargs...)
end

# In-place scalar query => array output (SeriesInterpolant)
# Note: No element type constraint - parent handles type checking/conversion
@inline function (d::DerivativeView{Order, ITP})(
    out::AbstractArray, xq::Real; deriv=nothing, kwargs...
) where {Order, ITP}
    _check_no_deriv_override(Val(Order), deriv)
    d.parent(out, xq; deriv=Order, kwargs...)
end

