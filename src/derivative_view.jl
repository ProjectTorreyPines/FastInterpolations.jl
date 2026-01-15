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
    deriv1(itp::CubicInterpolant) -> DerivativeView{1, ...}

Create a callable first-derivative view of the interpolant.

Returns a lightweight wrapper that can be broadcast or used in HOF composition.

# Example
```julia
itp = cubic_interp(x, y)
d1 = deriv1(itp)
slopes = d1.(xs)  # Broadcast over query points
```
"""
@inline deriv1(itp::CubicInterpolant) = DerivativeView{1, typeof(itp)}(itp)

"""
    deriv2(itp::CubicInterpolant) -> DerivativeView{2, ...}

Create a callable second-derivative view of the interpolant.

# Example
```julia
itp = cubic_interp(x, y)
d2 = deriv2(itp)
curvatures = d2.(xs)  # Broadcast over query points
```
"""
@inline deriv2(itp::CubicInterpolant) = DerivativeView{2, typeof(itp)}(itp)

"""
    deriv1(itp::LinearInterpolant) -> DerivativeView{1, ...}

Create a callable first-derivative view of the linear interpolant.

# Example
```julia
litp = linear_interp(x, y)
d1 = deriv1(litp)
slopes = d1.(xs)  # Piecewise constant slopes
```
"""
@inline deriv1(itp::LinearInterpolant) = DerivativeView{1, typeof(itp)}(itp)

"""
    deriv2(itp::LinearInterpolant) -> DerivativeView{2, ...}

Create a callable second-derivative view of the linear interpolant.
Always returns 0.0 (linear has no curvature).

# Example
```julia
litp = linear_interp(x, y)
d2 = deriv2(litp)
@assert all(d2.(xs) .== 0.0)
```
"""
@inline deriv2(itp::LinearInterpolant) = DerivativeView{2, typeof(itp)}(itp)

"""
    deriv1(itp::ConstantInterpolant) -> DerivativeView{1, ...}

Create a callable first-derivative view of the constant interpolant.
Always returns zero (constant functions have zero derivative).

# Example
```julia
citp = constant_interp(x, y)
d1 = deriv1(citp)
@assert all(d1.(xs) .== 0.0)
```
"""
@inline deriv1(itp::ConstantInterpolant) = DerivativeView{1, typeof(itp)}(itp)

"""
    deriv2(itp::ConstantInterpolant) -> DerivativeView{2, ...}

Create a callable second-derivative view of the constant interpolant.
Always returns zero (constant functions have zero curvature).

# Example
```julia
citp = constant_interp(x, y)
d2 = deriv2(citp)
@assert all(d2.(xs) .== 0.0)
```
"""
@inline deriv2(itp::ConstantInterpolant) = DerivativeView{2, typeof(itp)}(itp)

"""
    deriv1(itp::QuadraticInterpolant) -> DerivativeView{1, ...}

Create a callable first-derivative view of the quadratic interpolant.

# Example
```julia
qitp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
d1 = deriv1(qitp)
slopes = d1.(xs)  # First derivative at query points
```
"""
@inline deriv1(itp::QuadraticInterpolant) = DerivativeView{1, typeof(itp)}(itp)

"""
    deriv2(itp::QuadraticInterpolant) -> DerivativeView{2, ...}

Create a callable second-derivative view of the quadratic interpolant.

# Example
```julia
qitp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
d2 = deriv2(qitp)
curvatures = d2.(xs)  # Second derivative (constant per interval)
```
"""
@inline deriv2(itp::QuadraticInterpolant) = DerivativeView{2, typeof(itp)}(itp)

# ========================================
# Third Derivative Factories
# ========================================

"""
    deriv3(itp::CubicInterpolant) -> DerivativeView{3, ...}

Create a callable third-derivative view of the cubic interpolant.

# Note
Third derivative of cubic spline is constant within each interval.
Values may jump at knot points (this is mathematically expected).

# Example
```julia
itp = cubic_interp(x, y)
d3 = deriv3(itp)
jerks = d3.(xs)  # Third derivative (jerk) at query points
```
"""
@inline deriv3(itp::CubicInterpolant) = DerivativeView{3, typeof(itp)}(itp)

"""
    deriv3(itp::LinearInterpolant) -> DerivativeView{3, ...}

Third derivative of linear interpolation is always zero.
"""
@inline deriv3(itp::LinearInterpolant) = DerivativeView{3, typeof(itp)}(itp)

"""
    deriv3(itp::ConstantInterpolant) -> DerivativeView{3, ...}

Third derivative of constant interpolation is always zero.
"""
@inline deriv3(itp::ConstantInterpolant) = DerivativeView{3, typeof(itp)}(itp)

"""
    deriv3(itp::QuadraticInterpolant) -> DerivativeView{3, ...}

Third derivative of quadratic spline is always zero.
"""
@inline deriv3(itp::QuadraticInterpolant) = DerivativeView{3, typeof(itp)}(itp)

# ========================================
# Callable Methods
# ========================================

# First derivative view - scalar call only (broadcast-friendly)
@inline function (d::DerivativeView{1, ITP})(xi::Real) where {ITP}
    d.parent(xi; deriv=1)
end

# Second derivative view - scalar call only (broadcast-friendly)
@inline function (d::DerivativeView{2, ITP})(xi::Real) where {ITP}
    d.parent(xi; deriv=2)
end

# Third derivative view - scalar call only (broadcast-friendly)
@inline function (d::DerivativeView{3, ITP})(xi::Real) where {ITP}
    d.parent(xi; deriv=3)
end

# Note: Vector methods are intentionally NOT defined.
# Use broadcast for vector evaluation: d.(xs)
# This preserves the "one way to do it" principle and enables fused broadcast.
