# ========================================
# DerivativeView: Wrapper for Broadcast Support
# ========================================
#
# This file contains:
# - DerivativeView struct for derivative evaluation
# - Factory functions for CubicInterpolant and LinearInterpolant
# - Callable methods delegating to parent's order keyword
#
# The wrapper enables:
# - Broadcast: derivative(itp).(xs)
# - HOF composition: map(derivative(itp), xs)
# - Fused broadcast: @. coef * derivative(itp)(xs)

# ========================================
# DerivativeView Struct
# ========================================

"""
    DerivativeView{Order, ITP}

Lightweight wrapper for derivative evaluation.
Enables broadcast and higher-order function composition.

# Note
This is an internal type. Use `derivative(itp)` or `derivative2(itp)` to create.
For vector evaluation, use broadcast: `derivative(itp).(xs)`

# Example
```julia
itp = cubic_interp(x, y)
d1 = derivative(itp)      # First derivative view
d2 = derivative2(itp)     # Second derivative view

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
    derivative(itp::CubicInterpolant) -> DerivativeView{1, ...}

Create a callable first-derivative view of the interpolant.

Returns a lightweight wrapper that can be broadcast or used in HOF composition.

# Example
```julia
itp = cubic_interp(x, y)
d1 = derivative(itp)
slopes = d1.(xs)  # Broadcast over query points
```
"""
@inline derivative(itp::CubicInterpolant) = DerivativeView{1, typeof(itp)}(itp)

"""
    derivative2(itp::CubicInterpolant) -> DerivativeView{2, ...}

Create a callable second-derivative view of the interpolant.

# Example
```julia
itp = cubic_interp(x, y)
d2 = derivative2(itp)
curvatures = d2.(xs)  # Broadcast over query points
```
"""
@inline derivative2(itp::CubicInterpolant) = DerivativeView{2, typeof(itp)}(itp)

"""
    derivative(itp::LinearInterpolant) -> DerivativeView{1, ...}

Create a callable first-derivative view of the linear interpolant.

# Example
```julia
litp = linear_interp(x, y)
d1 = derivative(litp)
slopes = d1.(xs)  # Piecewise constant slopes
```
"""
@inline derivative(itp::LinearInterpolant) = DerivativeView{1, typeof(itp)}(itp)

"""
    derivative2(itp::LinearInterpolant) -> DerivativeView{2, ...}

Create a callable second-derivative view of the linear interpolant.
Always returns 0.0 (linear has no curvature).

# Example
```julia
litp = linear_interp(x, y)
d2 = derivative2(litp)
@assert all(d2.(xs) .== 0.0)
```
"""
@inline derivative2(itp::LinearInterpolant) = DerivativeView{2, typeof(itp)}(itp)

# ========================================
# Callable Methods
# ========================================

# First derivative view - scalar call only (broadcast-friendly)
@inline function (d::DerivativeView{1, ITP})(xi::Real) where {ITP}
    d.parent(xi; order=1)
end

# Second derivative view - scalar call only (broadcast-friendly)
@inline function (d::DerivativeView{2, ITP})(xi::Real) where {ITP}
    d.parent(xi; order=2)
end

# Note: Vector methods are intentionally NOT defined.
# Use broadcast for vector evaluation: d.(xs)
# This preserves the "one way to do it" principle and enables fused broadcast.
