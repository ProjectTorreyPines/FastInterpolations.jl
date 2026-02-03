# Complex Number Support

All interpolation methods support complex-valued data natively. The type system automatically handles the distinction between the grid type (which must be real) and the value type (which can be real or complex).

## Basic Usage

When interpolating complex data, the interpolant will return complex values. Analytic derivatives (`deriv=1`, `deriv=2`) also return complex values corresponding to the derivative of the real and imaginary parts respectively.

```julia
using FastInterpolations

x = 0.0:1.0:4.0
y = [1.0+0.5im, 0.5-1.0im, -2.0+1.5im, -1.0-0.5im, 2.0+2.0im]

# Create cubic spline with complex values
itp = cubic_interp(x, y)

# Evaluate at x = 1.5
val = itp(1.5)  # Returns ComplexF64
```

## Type System

Internally, the interpolant separates the grid type (`Tg`) from the value type (`Tv`):

*   **Grid (`Tg`)**: Always `AbstractFloat` (`Float32` or `Float64`). This ensures that grid search and interval calculations remain performant and ordered.
*   **Values (`Tv`)**: Can be `Real` or `Complex`.

If you provide inputs with different precisions (e.g., `Float64` grid and `ComplexF32` values), they will be promoted appropriately to ensure compatibility.

## Boundary Conditions with Complex Values

Boundary conditions (`BCPair`) can specify complex values for derivatives. You can even mix real and complex boundary conditions if needed (though typically they match the data type).

```julia
# Left: First derivative = 1.0 + 2.0im (Complex)
# Right: Natural boundary (Second derivative = 0.0) (Real)
bc = BCPair(Deriv1(1.0 + 2.0im), Deriv2(0.0))

itp = cubic_interp(x, y; bc=bc)
```

**Automatic Estimation:**
The `CubicFit()` boundary condition works seamlessly with complex data, estimating the complex derivatives at the endpoints using a 4-point polynomial fit.

```julia
# Estimates complex derivatives from the data points at boundaries
itp_auto = cubic_interp(x, y; bc=CubicFit())
```

## Applications: Periodic Trajectories

A common use case in physics and engineering is interpolating a closed trajectory in the complex plane.

```julia
# Circle in complex plane
theta = range(0, 2π, 101)
trajectory = exp.(im .* theta)

# PeriodicBC ensures C² continuity at the wrap-around point (0 ↔ 2π)
itp_periodic = cubic_interp(theta, trajectory; bc=PeriodicBC())
```
