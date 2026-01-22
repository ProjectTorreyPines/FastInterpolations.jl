# Visualization

FastInterpolations.jl provides built-in [Plots.jl](https://docs.juliaplots.org/) recipes for automatic visualization of interpolants. Simply call `plot()` on any interpolant to generate publication-ready figures.

## Quick Start

```@example viz
using FastInterpolations
using Plots

# Create sample data
x = [0.0, 0.7, 1.5, 2.3, 3.0, 4.2, 5.0, 6.0]
y = [0.2, 1.1, 0.6, 1.8, 1.2, 0.4, 1.5, 0.8]

# Create interpolant and plot
itp = cubic_interp(x, y; extrap=:constant)
plot(itp)
```

The recipe automatically generates:
- **Interpolation curve** — high-resolution spline visualization
- **Data points** — original scatter data (auto-hidden for large datasets)
- **Domain boundaries** — dashed vertical lines at `x[1]` and `x[end]`
- **Out-of-domain shading** — gray regions beyond the data domain (when extrapolation enabled)

## Discovering Available Options

Use `help_plot()` to discover all available recipe options at runtime:

```@example viz
help_plot(itp)
```

Different interpolant types show their specific options:

```@example viz
# Multi-series interpolant (includes series_idx option)
sitp = cubic_interp(x, [y y.^2]; extrap=:constant)
help_plot(sitp)
```

```@example viz
# Derivative view (show_data defaults to false)
help_plot(deriv1(itp))
```

## Single-Series Interpolants

All single-series interpolants (`LinearInterpolant`, `ConstantInterpolant`, `QuadraticInterpolant`, `CubicInterpolant`) share the same recipe options.

### Basic Usage

```@example viz
# Different interpolation methods
itp_linear = linear_interp(x, y; extrap=:extension)
itp_cubic = cubic_interp(x, y; extrap=:extension)

plot(
    plot(itp_linear, title="Linear"),
    plot(itp_cubic, title="Cubic"),
    layout=(1, 2), size=(800, 350)
)
```

### Recipe Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `show_data` | `Bool` or `nothing` | `nothing` (auto) | Show data points. Auto-hidden when `n ≥ 200` |
| `show_bounds` | `Bool` | `true` | Show vertical lines at domain boundaries |
| `show_outside` | `Bool` | `true` | Show gray shading for out-of-domain regions |
| `samples` | `Int` or `nothing` | `nothing` (auto) | Curve sample count. Default: `clamp(50*(n-1), 200, 2000)` |
| `domain_margin` | `Real` or `nothing` | `nothing` (auto) | Domain extension. Default: 25% of domain width |

### Customization Examples

```@example viz
itp = cubic_interp(x, y; extrap=:extension)

# Minimal: curve only
p1 = plot(itp; show_data=false, show_bounds=false, show_outside=false, title="Curve only")

# Dense sampling
p2 = plot(itp; samples=1000, title="Dense sampling")

# Extended domain
p3 = plot(itp; domain_margin=3.0, title="Wide margin")

# Custom styling (Plots.jl attributes work normally)
p4 = plot(itp; color=:red, linewidth=3, linestyle=:dash, title="Custom style")

plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
```

### Extrapolation Mode Effects

The recipe adapts to the interpolant's extrapolation mode:

```@example viz
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = sin.(x)

plot(
    plot(cubic_interp(x, y; extrap=:none), title="extrap=:none"),
    plot(cubic_interp(x, y; extrap=:constant), title="extrap=:constant"),
    plot(cubic_interp(x, y; extrap=:extension), title="extrap=:extension"),
    plot(cubic_interp(x, y; extrap=:wrap), title="extrap=:wrap"),
    layout=(2, 2), size=(800, 600)
)
```

## Multi-Series Interpolants

Series interpolants (`LinearSeriesInterpolant`, `CubicSeriesInterpolant`, etc.) plot multiple curves with distinct colors.

### Basic Usage

```@example viz
x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
Y = [sin.(x) cos.(x) sin.(x .+ 1)]  # 3 series

sitp = cubic_interp(x, Y; extrap=:extension)
plot(sitp, title="Multi-Series (3 curves)")
```

### Series Selection

Use `series_idx` to control which series to display:

```@example viz
x = range(0, 2π, 10)
Y = [sin.(x) cos.(x) tan.(x) ./ 5]

sitp = cubic_interp(collect(x), Y; extrap=:extension)

plot(
    plot(sitp; series_idx=:all, title="All series"),
    plot(sitp; series_idx=:first, title="First only"),
    plot(sitp; series_idx=1:2, title="Series 1-2"),
    plot(sitp; series_idx=[1, 3], title="Series 1 & 3"),
    layout=(2, 2), size=(800, 600)
)
```

### Multi-Series Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `series_idx` | `Symbol`, `Int`, `Range`, `Vector{Int}` | `:all` | Which series to plot |

Values for `series_idx`:
- `:all` — Plot all series (default)
- `:first` — Plot only the first series
- `n::Int` — Plot series `n` only
- `1:3` — Plot series 1 through 3
- `[1, 3, 5]` — Plot specific series

## Derivative Views

Derivative views (`deriv1`, `deriv2`, `deriv3`) have their own recipe with derivative-specific defaults.

### Basic Usage

```@example viz
x = range(0, 2π, 15)
y = sin.(x)

itp = cubic_interp(collect(x), y; extrap=:extension)

plot(
    plot(itp, title="S(x)"),
    plot(deriv1(itp), title="S'(x)"),
    plot(deriv2(itp), title="S''(x)"),
    plot(deriv3(itp), title="S'''(x)"),
    layout=(2, 2), size=(800, 600)
)
```

### Derivative-Specific Defaults

| Option | Default (Derivatives) | Default (Interpolants) |
|--------|----------------------|------------------------|
| `show_data` | `false` | `true` (auto) |
| Line style | `:dash` | `:solid` |
| Color | `:steelblue` | `:blue` |

Derivatives hide data points by default since the original `y` values don't represent derivative values.

### Combined Visualization

Plot the interpolant and its derivative together:

```@example viz
x = range(0, 2π, 12)
y = sin.(x)
itp = cubic_interp(collect(x), y; extrap=:extension)

p = plot(itp, label="sin(x)", title="Function and Derivative")
plot!(p, deriv1(itp), label="cos(x) ≈ S'(x)")
```

## Intelligent Defaults

The recipe includes smart defaults that adapt to your data:

### Automatic Marker Scaling

Marker size and opacity scale inversely with data count to prevent visual clutter:

```@example viz
# Small dataset: large, opaque markers
x_small = range(0, 5, 8)
y_small = sin.(x_small)

# Large dataset: smaller, transparent markers
x_large = range(0, 5, 50)
y_large = sin.(x_large)

plot(
    plot(cubic_interp(collect(x_small), y_small), title="n=8 (large markers)"),
    plot(cubic_interp(collect(x_large), y_large), title="n=50 (small markers)"),
    layout=(1, 2), size=(800, 350)
)
```

### Automatic Scatter Hiding

For datasets with 200+ points, scatter points are hidden by default:

```julia
x = range(0, 10, 250)
y = sin.(x)
itp = cubic_interp(collect(x), y)

plot(itp)                    # Scatter hidden (n ≥ 200)
plot(itp; show_data=true)    # Force show scatter
```

### Automatic Curve Sampling

Sample count is computed as `clamp(50*(n-1), 200, 2000)`:

| Data Points | Auto Samples |
|-------------|--------------|
| 5 | 200 |
| 10 | 450 |
| 20 | 950 |
| 50+ | 2000 |

## Combining with Plots.jl

All standard Plots.jl attributes work normally:

```@example viz
x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
y = [0.0, 0.8, 0.9, 0.4, 0.2, 0.6]
itp = cubic_interp(x, y; extrap=:constant)

plot(itp;
    title="Styled Interpolant",
    xlabel="Time (s)",
    ylabel="Amplitude",
    color=:crimson,
    linewidth=3,
    legend=:topright,
    size=(600, 400),
    dpi=150
)
```

### Overlaying Multiple Interpolants

```@example viz
x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
y = [0.0, 0.8, 0.9, 0.4, 0.2, 0.6]

itp_const = constant_interp(x, y)
itp_linear = linear_interp(x, y)
itp_quad = quadratic_interp(x, y)
itp_cubic = cubic_interp(x, y)

# Note: overlay using plot!() after initial plot()
p = plot(itp_const; show_bounds=false, show_outside=false, label="constant", color=:gray)
plot!(p, itp_linear; show_data=false, show_bounds=false, show_outside=false, label="linear", color=:blue)
plot!(p, itp_quad; show_data=false, show_bounds=false, show_outside=false, label="quadratic", color=:green)
plot!(p, itp_cubic; show_data=false, show_bounds=false, show_outside=false, label="cubic", color=:red)
plot!(p; title="Method Comparison", legend=:topright)
```

## Summary

| Interpolant Type | Recipe Features |
|------------------|-----------------|
| Single-series (`CubicInterpolant`, etc.) | Curve, scatter, bounds, out-of-domain shading |
| Multi-series (`CubicSeriesInterpolant`, etc.) | Multiple colored curves, `series_idx` selection |
| Derivative views (`deriv1`, `deriv2`, `deriv3`) | Dashed curves, `show_data=false` default |

## See Also

- **[Extrapolation](extrapolation.md)**: How extrapolation modes affect visualization
- **[Derivatives](interpolation/derivatives.md)**: Analytical derivative computation
- **[Series Interpolant](guides/series_interpolant.md)**: Multi-series interpolant usage
