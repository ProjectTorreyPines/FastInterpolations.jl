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
    DerivativeView{Order, ITP}

Lightweight wrapper for derivative evaluation.
Enables broadcast and higher-order function composition.

# Note
This is an internal type. Use `deriv1(itp)` or `deriv2(itp)` to create.
For vector evaluation, use broadcast: `deriv1(itp).(xs)`

# Example
```julia
itp = cubic_interp(x, y)
d1 = deriv1(itp)      # First derivative view
d2 = deriv2(itp)     # Second derivative view

# Broadcast evaluation
slopes = d1.(query_points)

# Fused broadcast
result = @. coef * d1(xs) * other_func(xs)
```
"""
struct DerivativeView{Order, ITP}
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

# Features
- **Functional API**: Pass derivative views into high-order functions like `map` or `find_zeros`.
- **Broadcast Fusion**: Enables evaluation without temporary allocations when combined with other operations using dot syntax (e.g., `@. d1(xs) + d2(xs)`).
- **Unified Interface**: Supports both single-series and multi-series (SeriesInterpolant) objects.
  - For `AbstractInterpolant`, returns a scalar derivative.
  - For `AbstractSeriesInterpolant`, returns a vector of derivatives.

# Notes
- **Order 3**: For cubic splines, the 3rd derivative is piecewise constant and may jump at knot points. For lower-order interpolants, it returns zero.
- **In-place**: Supports in-place evaluation for SeriesInterpolant via `d(out, xq)`.

# Example
```julia
itp = cubic_interp(x, y)
d1 = deriv1(itp)

# Evaluate at a point
val = d1(0.5)

# Use in higher-order functions
using Roots
root = find_zero(d1, 0.5)

# Fused broadcast (no temporary allocations)
query_pts = range(0.0, 1.0, 100)
results = @. itp(query_pts) + 0.1 * d1(query_pts)
```
"""
(deriv1, deriv2, deriv3)

@inline deriv1(itp::AbstractInterpolant) = DerivativeView{1, typeof(itp)}(itp)
@inline deriv2(itp::AbstractInterpolant) = DerivativeView{2, typeof(itp)}(itp)
@inline deriv3(itp::AbstractInterpolant) = DerivativeView{3, typeof(itp)}(itp)

# ========================================
# Callable Methods
# ========================================

# Out-of-place calls (Scalar or Vector)
# Delegates to parent's callable with the appropriate `deriv` value.
# - Scalar: Used for single points or broadcasting `d.(xs)`.
# - Vector: Convenience for standalone evaluation; returns a newly allocated array.
@inline function (d::DerivativeView{Order, ITP})(xq::Union{Real, AbstractArray{<:Real}}) where {Order, ITP}
    d.parent(xq; deriv=Order)
end

# In-place scalar query => vector output (SeriesInterpolant)
@inline function (d::DerivativeView{Order, ITP})(out::AbstractArray{<:Real}, xq::Real) where {Order, ITP}
    d.parent(out, xq; deriv=Order)
end

# In-place vector query => vector-of-vectors output (SeriesInterpolant)
@inline function (d::DerivativeView{Order, ITP})(out::AbstractArray{<:AbstractArray{<:Real}}, xq::AbstractArray{<:Real}) where {Order, ITP}
    d.parent(out, xq; deriv=Order)
end
