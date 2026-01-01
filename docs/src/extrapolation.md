# Extrapolation

Extrapolation controls behavior when query points fall outside the data domain `[x[1], x[end]]`.

```@example extrap
using FastInterpolations
using Plots

# Sample data with non-uniform values
x = [0.0, 0.7, 1.5, 2.3, 3.0, 4.2, 5.0, 6.0]
y = [0.2, 1.1, 0.6, 1.8, 1.2, 0.4, 1.5, 0.8]

# Query points: interior vs exterior
xq_in = range(x[1], x[end], 200)          # inside domain
xq_left = range(x[1] - 1.5, x[1], 50)     # left of domain
xq_right = range(x[end], x[end] + 1.5, 50) # right of domain
xq_out = vcat(xq_left, xq_right)          # all exterior points
```

## `extrap=:none` (Default)

Throws `DomainError` for out-of-domain queries. Use when extrapolation is unexpected.

```@example extrap
y_in = cubic_interp(x, y, xq_in; extrap=:none)

plot(title="extrap=:none", xlabel="x", ylabel="y", legend=:topright)
plot!(xq_in, y_in, label="interpolation", linewidth=2, color=:blue)
scatter!(x, y, label="data", markersize=7, color=:blue)
```

Querying outside the domain throws an error:

```@example extrap
cubic_interp(x, y, -1.0; extrap=:none)  # DomainError
```

## `extrap=:extension`

Extends the boundary polynomial beyond the domain.

```@example extrap
y_in = cubic_interp(x, y, xq_in; extrap=:extension)
y_out = cubic_interp(x, y, xq_out; extrap=:extension)

plot(title="extrap=:extension", xlabel="x", ylabel="y", legend=:topright)
plot!(xq_in, y_in, label="interpolation", linewidth=2, color=:blue)
scatter!(x, y, label="data", markersize=7, color=:blue)
plot!(xq_left, cubic_interp(x, y, xq_left; extrap=:extension),
      label="extrapolation", linewidth=2, linestyle=:dash, color=:red)
plot!(xq_right, cubic_interp(x, y, xq_right; extrap=:extension),
      label=nothing, linewidth=2, linestyle=:dash, color=:red)
```

## `extrap=:constant`

Returns boundary values: `y[1]` for left, `y[end]` for right.

```@example extrap
y_in = cubic_interp(x, y, xq_in; extrap=:constant)

plot(title="extrap=:constant", xlabel="x", ylabel="y", legend=:topright)
plot!(xq_in, y_in, label="interpolation", linewidth=2, color=:blue)
scatter!(x, y, label="data", markersize=7, color=:blue)
plot!(xq_left, cubic_interp(x, y, xq_left; extrap=:constant),
      label="extrapolation", linewidth=2, linestyle=:dash, color=:red)
plot!(xq_right, cubic_interp(x, y, xq_right; extrap=:constant),
      label=nothing, linewidth=2, linestyle=:dash, color=:red)
```

## `extrap=:wrap`

Wraps coordinates periodically to `[x[1], x[end])`. Best for cyclic data where `y[1] ≈ y[end]`.

```@example extrap
y_in = cubic_interp(x, y, xq_in; extrap=:wrap)

plot(title="extrap=:wrap", xlabel="x", ylabel="y", legend=:topright)
plot!(xq_in, y_in, label="interpolation", linewidth=2, color=:blue)
scatter!(x, y, label="data", markersize=7, color=:blue)
plot!(xq_left, cubic_interp(x, y, xq_left; extrap=:wrap),
      label="extrapolation", linewidth=2, linestyle=:dash, color=:red)
plot!(xq_right, cubic_interp(x, y, xq_right; extrap=:wrap),
      label=nothing, linewidth=2, linestyle=:dash, color=:red)
```

## Comparison

```@example extrap
xq_all = range(x[1] - 1.5, x[end] + 1.5, 300)

plot(title="Extrapolation Comparison", xlabel="x", ylabel="y",
     legend=:topright, size=(700, 400))
plot!(xq_all, cubic_interp(x, y, xq_all; extrap=:extension),
      label=":extension", linewidth=2)
plot!(xq_all, cubic_interp(x, y, xq_all; extrap=:constant),
      label=":constant", linewidth=2, linestyle=:dash)
plot!(xq_all, cubic_interp(x, y, xq_all; extrap=:wrap),
      label=":wrap", linewidth=2, linestyle=:dashdot)
scatter!(x, y, label="data", markersize=7, color=:black)
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## Summary

| Mode | Behavior | Use Case |
|------|----------|----------|
| `:none` | `DomainError` | Strict domain enforcement (default) |
| `:extension` | Continues boundary polynomial | Smooth continuation |
| `:constant` | Returns boundary values | Physical constraints |
| `:wrap` | Wraps periodically | Cyclic data (angles, phases) |

## See Also

- **[Derivatives](derivatives.md)**: Analytical derivatives with extrapolation
- **[Standard BC](interpolation/cubic/standard.md)**: Boundary conditions for cubic splines
- **[Periodic BC](interpolation/cubic/periodic.md)**: Smooth wrap-around for cyclic data
