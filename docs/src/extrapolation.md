# Extrapolation

Extrapolation controls behavior when query points fall outside the data domain `[x[1], x[end]]`.

## Overview

Use the `extrap` keyword argument to specify extrapolation behavior:

```julia-repl
# One-shot: specify extrap per call
cubic_interp(x, y, xq; extrap=:constant)
linear_interp(x, y, xq; extrap=:extension)

# Interpolant: extrap is fixed at creation
itp = cubic_interp(x, y; extrap=:extension)  # all future calls use :extension
itp(xq)  # uses :extension
```

Both `linear_interp` and `cubic_interp` support the same extrapolation modes.

| Mode | Behavior |
|------|----------|
| `:none` | Throws `DomainError` (default) |
| `:constant` | Returns boundary values |
| `:extension` | Extends boundary polynomial |
| `:wrap` | Wraps coordinates periodically (no smoothness enforced) |

## Examples

```@example extrap
using FastInterpolations
using Plots

# Sample data
x = [0.0, 0.7, 1.5, 2.3, 3.0, 4.2, 5.0, 6.0]
y = [0.2, 1.1, 0.6, 1.8, 1.2, 0.4, 1.5, 0.8]

# Query points (full range including extrapolation region)
xq = range(x[1] - 1.5, x[end] + 1.5, 300)
```

## `extrap=:none` (Default)

Throws `DomainError` for out-of-domain queries. Use when extrapolation is unexpected.

```julia
julia> cubic_interp(x, y, -1.0; extrap=:none)  # scalar query outside domain
ERROR: DomainError with -1.0:
query point outside interpolation domain [0.0, 6.0]

julia> cubic_interp(x, y, xq; extrap=:none)  # vector query (xq includes out-of-domain points)
ERROR: DomainError with -1.5:
query point outside interpolation domain [0.0, 6.0]
```

Only interior queries succeed:

```@example extrap
yq = cubic_interp(x, y, range(x[1], x[end], 200); extrap=:none)

plot(title="extrap=:none", xlabel="x", ylabel="y", legend=:topright,
     xlims=(x[1] - 1.5, x[end] + 1.5))
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label="out of domain")
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing)
plot!(range(x[1], x[end], 200), yq, label="spline", linewidth=2)
scatter!(x, y, label="data", markersize=7, color=:black)
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## `extrap=:constant`

Returns boundary values: `y[1]` for left, `y[end]` for right.

```@example extrap
yq = cubic_interp(x, y, xq; extrap=:constant)

plot(title="extrap=:constant", xlabel="x", ylabel="y", legend=:topright)
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label="out of domain")
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing)
plot!(xq, yq, label="spline", linewidth=2)
scatter!(x, y, label="data", markersize=7, color=:black)
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## `extrap=:extension`

Extends the boundary polynomial beyond the domain.

```@example extrap
yq = cubic_interp(x, y, xq; extrap=:extension)

plot(title="extrap=:extension", xlabel="x", ylabel="y", legend=:topright)
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label="out of domain")
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing)
plot!(xq, yq, label="spline", linewidth=2)
scatter!(x, y, label="data", markersize=7, color=:black)
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## `extrap=:wrap`

Wraps coordinates periodically:

```math
S(x + \tau) = S(x), \quad \tau = x_{\text{end}} - x_1
```

This is **purely coordinate mapping**—it does not enforce any physical conditions at the boundary. The spline may have discontinuities in value, slope, or curvature at the wrap point.

!!! note "For Smooth Periodicity"
    If you need C² continuity at the periodic boundary, use [`bc=PeriodicBC()`](interpolation/cubic/periodic.md) with `cubic_interp`. This enforces ``S(x_1) = S(x_{\text{end}})``, ``S'(x_1) = S'(x_{\text{end}})``, and ``S''(x_1) = S''(x_{\text{end}})``.

```@example extrap
yq = cubic_interp(x, y, xq; extrap=:wrap)

plot(title="extrap=:wrap", xlabel="x", ylabel="y", legend=:topright)
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label="out of domain")
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing)
plot!(xq, yq, label="spline", linewidth=2)
scatter!(x, y, label="data", markersize=7, color=:black)
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## Comparison

```@example extrap
plot(title="Extrapolation Comparison", xlabel="x", ylabel="y",
     legend=:topright, size=(700, 400))
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label=nothing)
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing)
plot!(xq, cubic_interp(x, y, xq; extrap=:constant),
      label=":constant", linewidth=2)
plot!(xq, cubic_interp(x, y, xq; extrap=:extension),
      label=":extension", linewidth=2, linestyle=:dash)
plot!(xq, cubic_interp(x, y, xq; extrap=:wrap),
      label=":wrap", linewidth=2, linestyle=:dashdot)
scatter!(x, y, label="data", markersize=7, color=:black)
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## Summary

| Mode | Behavior | Use Case |
|------|----------|----------|
| `:none` | `DomainError` | Strict domain enforcement (default) |
| `:constant` | Returns boundary values | Physical constraints |
| `:extension` | Continues boundary polynomial | Smooth continuation |
| `:wrap` | Wraps coordinates (no smoothness) | Cyclic data (see [`PeriodicBC`](interpolation/cubic/periodic.md) for C² continuity) |

## See Also

- **[Derivatives](derivatives.md)**: Analytical derivatives with extrapolation
- **[Standard BC](interpolation/cubic/standard.md)**: Boundary conditions for cubic splines
- **[Periodic BC](interpolation/cubic/periodic.md)**: Smooth wrap-around for cyclic data
