# Extrapolation

Extrapolation controls what happens when query points fall outside the data domain `[x[1], x[end]]`. FastInterpolations.jl provides four extrapolation modes for both linear and cubic interpolation.

## Extrapolation Options

| Option | Behavior |
|--------|----------|
| `:none` | Throws `DomainError` (default) |
| `:extension` | Extends boundary segments |
| `:constant` | Returns boundary values |
| `:wrap` | Wraps coordinates periodically |

## Linear Extrapolation

### Extension vs Constant

```@example extrap
using FastInterpolations
using Plots

# Sample data (Vector with non-uniform y values)
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0, 2.5, 1.5, 3.0, 2.0]

# Query points including extrapolation region
xq = range(-1.0, 5.0, 200)

# Different extrapolation methods (Range works directly)
y_extension = linear_interp(x, y, xq; extrap=:extension)
y_constant = linear_interp(x, y, xq; extrap=:constant)

# Visualize
p = plot(title="Linear Extrapolation: Extension vs Constant", xlabel="x", ylabel="y")
vspan!([-1.0, 0.0], alpha=0.1, color=:gray, label="extrapolation region")
vspan!([4.0, 5.0], alpha=0.1, color=:gray, label=nothing)
plot!(xq, y_extension, label="extrap=:extension", color=:blue, linewidth=2)
plot!(xq, y_constant, label="extrap=:constant", color=:red, linestyle=:dash, linewidth=2)
scatter!(x, y, label="data points", color=:black, markersize=8)
```

**Key differences:**
- `:extension` continues the slope of the boundary segment (blue line)
- `:constant` returns `y[1]` for `x < x[1]` and `y[end]` for `x > x[end]` (red dashed)

### Wrap Extrapolation

When `extrap=:wrap`, query points are wrapped to the domain `[x[1], x[end])`. This is ideal for periodic functions like trigonometric functions or any cyclic data.

!!! note "Smooth Wrapping"
    For seamless wrapping, ensure `y[1] ≈ y[end]`.

```@example extrap
# Periodic data: one complete sine wave (Range works directly)
x_periodic = range(0.0, 2π, 21)
y_periodic = sin.(x_periodic)  # y[1] = y[end] ≈ 0

# Query points extending beyond the domain
xq_extended = range(-π, 3π, 300)

# Compare wrap vs extension
y_wrap = linear_interp(x_periodic, y_periodic, xq_extended; extrap=:wrap)
y_ext = linear_interp(x_periodic, y_periodic, xq_extended; extrap=:extension)
y_true = sin.(xq_extended)

# Visualize
p2 = plot(title="Linear Extrapolation: Wrap vs Extension", xlabel="x", ylabel="y")
vspan!([-π, 0.0], alpha=0.1, color=:gray, label="outside domain")
vspan!([2π, 3π], alpha=0.1, color=:gray, label=nothing)
plot!(xq_extended, y_true, label="sin(x) reference", color=:lightgray, linewidth=3)
plot!(xq_extended, y_wrap, label="extrap=:wrap", color=:blue, linewidth=2)
plot!(xq_extended, y_ext, label="extrap=:extension", color=:red, linestyle=:dash, linewidth=2)
vline!([0.0, 2π], color=:black, linestyle=:dot, alpha=0.5, label=nothing)
scatter!(x_periodic, y_periodic, label="data points", color=:black, markersize=5)
```

Notice how `:wrap` (blue) repeats the interpolated pattern outside the domain, while `:extension` (red) diverges from the true sine wave.

## Cubic Extrapolation

### How Boundary Conditions Affect Extrapolation

The choice of boundary condition significantly impacts extrapolation behavior for cubic splines.

```@example extrap
# Fewer points to see the difference clearly
x_cubic = range(0.0, 2π, 11)
y_cubic = sin.(x_cubic)

# Query points extending beyond domain
xq_cubic = range(-π, 3π, 400)

# Different boundary conditions with extension extrapolation
y_natural = cubic_interp(x_cubic, y_cubic, xq_cubic; bc=NaturalBC(), extrap=:extension)
y_periodic_bc = cubic_interp(x_cubic, y_cubic, xq_cubic; bc=PeriodicBC(), extrap=:extension)
y_sin = sin.(xq_cubic)

# Visualize
p3 = plot(title="Cubic Extrapolation: Natural vs Periodic BC", xlabel="x", ylabel="y")
vspan!([-π, 0.0], alpha=0.1, color=:gray, label="outside domain")
vspan!([2π, 3π], alpha=0.1, color=:gray, label=nothing)
plot!(xq_cubic, y_sin, label="sin(x) reference", color=:lightgray, linewidth=3)
plot!(xq_cubic, y_natural, label="NaturalBC", color=:red, linestyle=:dash, linewidth=2)
plot!(xq_cubic, y_periodic_bc, label="PeriodicBC", color=:blue, linewidth=2)
vline!([0.0, 2π], color=:black, linestyle=:dot, alpha=0.5, label=nothing)
scatter!(x_cubic, y_cubic, label="data points", color=:black, markersize=6)
```

**Key observations:**
- **NaturalBC** (red): Zero curvature at endpoints causes the spline to "flatten out" during extrapolation
- **PeriodicBC** (blue): Matched derivatives create smoother continuation, closer to the true sine wave

### Curvature Comparison

The second derivative (curvature) reveals why the extrapolation behaviors differ. FastInterpolations.jl provides **analytical derivatives**—no finite difference approximation needed:

```@example extrap
# More complex periodic function
x_c2 = range(0.0, 2π, 17)
y_c2 = sin.(x_c2) .+ 0.3 .* cos.(3 .* x_c2)

# Fine query grid
xq_c2 = range(-0.5, 2π + 0.5, 500)

# Interpolate with both BC types
y_nat = cubic_interp(x_c2, y_c2, xq_c2; bc=NaturalBC(), extrap=:extension)
y_per = cubic_interp(x_c2, y_c2, xq_c2; bc=PeriodicBC(), extrap=:extension)

# Analytical second derivative using deriv=2 (no finite difference needed!)
d2y_nat = cubic_interp(x_c2, y_c2, xq_c2; bc=NaturalBC(), extrap=:extension, deriv=2)
d2y_per = cubic_interp(x_c2, y_c2, xq_c2; bc=PeriodicBC(), extrap=:extension, deriv=2)

# Two subplots
p4a = plot(title="Interpolation Results", xlabel="x", ylabel="y", legend=:topright)
plot!(p4a, xq_c2, y_nat, label="NaturalBC", color=:red, linewidth=2)
plot!(p4a, xq_c2, y_per, label="PeriodicBC", color=:blue, linewidth=2)
scatter!(p4a, x_c2, y_c2, label="data", color=:black, markersize=5)
vline!(p4a, [0.0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)

p4b = plot(title="Second Derivative (Curvature)", xlabel="x", ylabel="y''", legend=:topright)
plot!(p4b, xq_c2, d2y_nat, label="NaturalBC (→0 at boundaries)", color=:red, linewidth=2)
plot!(p4b, xq_c2, d2y_per, label="PeriodicBC (matches)", color=:blue, linewidth=2)
vline!(p4b, [0.0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
hline!(p4b, [0.0], color=:black, linestyle=:dash, alpha=0.3, label=nothing)

plot(p4a, p4b, layout=(2, 1), size=(700, 500))
```

The bottom plot shows clearly how `NaturalBC` forces the curvature to zero at boundaries, while `PeriodicBC` maintains continuous curvature.

!!! tip "Analytical Derivatives"
    Use `deriv=1` or `deriv=2` for analytical derivatives instead of finite difference approximations. See the [Derivatives](derivatives.md) page for details.

## Using with Interpolants

Create interpolant objects with the desired extrapolation mode for repeated evaluation.

```@example extrap
x_call = range(0.0, 2π, 51)
y_call = sin.(x_call)

# Create interpolants with different extrapolation modes
linear_ext = linear_interp(x_call, y_call; extrap=:extension)
linear_wrap_itp = linear_interp(x_call, y_call; extrap=:wrap)
cubic_nat = cubic_interp(x_call, y_call; extrap=:extension)
cubic_per = cubic_interp(x_call, y_call; bc=PeriodicBC(), extrap=:extension)

# Evaluate at test points (including outside domain)
test_points = [-0.5, 0.5, π, 2π + 0.5]

println("| x | Linear (ext) | Linear (wrap) | Cubic (natural) | Cubic (periodic) |")
println("|---|--------------|---------------|-----------------|------------------|")
for xi in test_points
    v1 = round(linear_ext(xi), digits=4)
    v2 = round(linear_wrap_itp(xi), digits=4)
    v3 = round(cubic_nat(xi), digits=4)
    v4 = round(cubic_per(xi), digits=4)
    println("| $xi | $v1 | $v2 | $v3 | $v4 |")
end
```

## Choosing the Right Extrapolation

| Data Type | Recommended |
|-----------|-------------|
| Periodic/cyclic data | `:wrap` with `PeriodicBC()` |
| Physical constraints at boundaries | `:constant` |
| Need smooth continuation | `:extension` |
| Strict domain enforcement | `:none` (default) |

!!! tip "Default Behavior"
    The default `extrap=:none` throws a `DomainError` for out-of-domain queries. This is the safest option when extrapolation is not expected.

## See Also

- **[Linear Interpolation](interpolation/linear.md)**: Basic usage and derivative evaluation
- **[Standard BC](interpolation/cubic/standard.md)**: NaturalBC, ClampedBC, and custom boundary conditions
- **[Periodic BC](interpolation/cubic/periodic.md)**: Smooth wrap-around for cyclic data
