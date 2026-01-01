# Extrapolation

Extrapolation controls behavior when query points fall outside the data domain `[x[1], x[end]]`.

```@example extrap
using FastInterpolations
using Plots

# Common data for all examples
x = range(0.0, 2π, 15)
y = sin.(x)
xq = range(-1.0, 2π + 1.0, 300)  # extends beyond domain
```

## `extrap=:none` (Default)

Throws `DomainError` for out-of-domain queries. Use when extrapolation is unexpected or invalid.

```@example extrap
try
    cubic_interp(x, y, -0.5; extrap=:none)
catch e
    println("Error: ", e)
end
```

## `extrap=:extension`

Extends the boundary polynomial beyond the domain. The spline continues smoothly using its edge behavior.

```@example extrap
y_ext = cubic_interp(x, y, xq; extrap=:extension)

plot(title="extrap=:extension", xlabel="x", ylabel="y", legend=:topleft)
vspan!([-1.0, 0.0], alpha=0.1, color=:gray, label="extrapolation")
vspan!([2π, 2π+1.0], alpha=0.1, color=:gray, label=nothing)
plot!(xq, y_ext, label="cubic spline", linewidth=2, color=:blue)
scatter!(x, y, label="data", color=:black, markersize=5)
vline!([0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## `extrap=:constant`

Returns the boundary value. Left of domain returns `y[1]`, right of domain returns `y[end]`.

```@example extrap
y_const = cubic_interp(x, y, xq; extrap=:constant)

plot(title="extrap=:constant", xlabel="x", ylabel="y", legend=:topleft)
vspan!([-1.0, 0.0], alpha=0.1, color=:gray, label="extrapolation")
vspan!([2π, 2π+1.0], alpha=0.1, color=:gray, label=nothing)
plot!(xq, y_const, label="cubic spline", linewidth=2, color=:blue)
scatter!(x, y, label="data", color=:black, markersize=5)
vline!([0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## `extrap=:wrap`

Wraps coordinates periodically to `[x[1], x[end])`. Ideal for cyclic data where `y[1] ≈ y[end]`.

```@example extrap
y_wrap = cubic_interp(x, y, xq; extrap=:wrap)

plot(title="extrap=:wrap", xlabel="x", ylabel="y", legend=:topleft)
vspan!([-1.0, 0.0], alpha=0.1, color=:gray, label="extrapolation")
vspan!([2π, 2π+1.0], alpha=0.1, color=:gray, label=nothing)
plot!(xq, y_wrap, label="cubic spline", linewidth=2, color=:blue)
scatter!(x, y, label="data", color=:black, markersize=5)
vline!([0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## Comparison

All four modes on the same plot:

```@example extrap
y_ext = cubic_interp(x, y, xq; extrap=:extension)
y_const = cubic_interp(x, y, xq; extrap=:constant)
y_wrap = cubic_interp(x, y, xq; extrap=:wrap)

plot(title="Extrapolation Comparison", xlabel="x", ylabel="y", legend=:topleft, size=(700, 400))
vspan!([-1.0, 0.0], alpha=0.08, color=:gray, label="extrapolation region")
vspan!([2π, 2π+1.0], alpha=0.08, color=:gray, label=nothing)
plot!(xq, y_ext, label="extrap=:extension", linewidth=2)
plot!(xq, y_const, label="extrap=:constant", linewidth=2, linestyle=:dash)
plot!(xq, y_wrap, label="extrap=:wrap", linewidth=2, linestyle=:dashdot)
scatter!(x, y, label="data", color=:black, markersize=5)
vline!([0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## Effect of Boundary Conditions

For cubic splines, boundary conditions affect extrapolation behavior:

```@example extrap
xq_bc = range(-0.5, 2π + 0.5, 300)
y_nat = cubic_interp(x, y, xq_bc; bc=NaturalBC(), extrap=:extension)
y_per = cubic_interp(x, y, xq_bc; bc=PeriodicBC(), extrap=:extension)

plot(title="BC Effect on Extrapolation", xlabel="x", ylabel="y", legend=:topleft)
vspan!([-0.5, 0.0], alpha=0.1, color=:gray, label="extrapolation")
vspan!([2π, 2π+0.5], alpha=0.1, color=:gray, label=nothing)
plot!(xq_bc, sin.(xq_bc), label="sin(x) true", color=:lightgray, linewidth=3)
plot!(xq_bc, y_nat, label="NaturalBC", linewidth=2, linestyle=:dash)
plot!(xq_bc, y_per, label="PeriodicBC", linewidth=2)
scatter!(x, y, label="data", color=:black, markersize=5)
vline!([0, 2π], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

- **NaturalBC**: Zero curvature at endpoints causes "flattening"
- **PeriodicBC**: Matched derivatives create smoother continuation

## Summary

| Mode | Behavior | Use Case |
|------|----------|----------|
| `:none` | Throws `DomainError` | Strict domain enforcement (default) |
| `:extension` | Extends boundary polynomial | Smooth continuation needed |
| `:constant` | Returns boundary values | Physical constraints at edges |
| `:wrap` | Wraps coordinates periodically | Cyclic data (angles, phases) |

## See Also

- **[Derivatives](derivatives.md)**: Analytical derivatives with extrapolation
- **[Standard BC](interpolation/cubic/standard.md)**: NaturalBC, ClampedBC, and custom constraints
- **[Periodic BC](interpolation/cubic/periodic.md)**: Smooth wrap-around for cyclic data
