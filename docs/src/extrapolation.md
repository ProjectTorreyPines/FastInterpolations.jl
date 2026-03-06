# Extrapolation

Extrapolation controls behavior when query points fall outside the data domain `[x[1], x[end]]`.

## Overview

All extrapolation modes are concrete subtypes of [`AbstractExtrap`](@ref). Pass them via the `extrap` keyword argument:

```julia-repl
# Using Extrap() factory (recommended)
cubic_interp(x, y, xq; extrap=Extrap(:clamped))
linear_interp(x, y, xq; extrap=Extrap(:extend))

# Using direct types (also supported)
cubic_interp(x, y, xq; extrap=ClampExtrap())
linear_interp(x, y, xq; extrap=ExtendExtrap())

# Interpolant: extrap is fixed at creation
itp = cubic_interp(x, y; extrap=Extrap(:extend))
itp(xq)  # uses ExtendExtrap
```

!!! tip "Factory Functions"
    The [`Extrap()`](@ref factory_functions) factory provides a single discoverable entry point. For ND per-axis configuration, use the multi-arg form: `Extrap(:extend, :clamped, :none)`. See [Factory Functions](@ref factory_functions) for details.

All interpolation methods (`linear_interp`, `quadratic_interp`, `cubic_interp`, `constant_interp`) support the same extrapolation modes.

| Type | Behavior |
|------|----------|
| [`NoExtrap()`](@ref NoExtrap) | Throws `DomainError` (default) |
| [`ClampExtrap()`](@ref ClampExtrap) | Returns boundary values |
| [`ExtendExtrap()`](@ref ExtendExtrap) | Extends boundary polynomial |
| [`WrapExtrap()`](@ref WrapExtrap) | Wraps coordinates periodically (no smoothness enforced) |

```
AbstractExtrap
├── NoExtrap       # DomainError on out-of-domain (default)
├── ClampExtrap    # clamp to y[1] / y[end]
├── ExtendExtrap   # continue boundary polynomial
└── WrapExtrap     # modular coordinate mapping
```

## Examples

```@example extrap
using FastInterpolations
using Plots # hide

# Sample data
x = [0.0, 0.7, 1.5, 2.3, 3.0, 4.2, 5.0, 6.0]
y = [0.2, 1.1, 0.6, 1.8, 1.2, 0.4, 1.5, 0.8]

# Query points (including extrapolation region)
xq = range(x[1] - 1.5, x[end] + 1.5, 300)
nothing # hide
```

## `NoExtrap()` (Default)

Throws `DomainError` for out-of-domain queries. Use when extrapolation is unexpected.

```julia
julia> cubic_interp(x, y, -1.0; extrap=NoExtrap())  # scalar query outside domain
ERROR: DomainError with -1.0:
query point outside interpolation domain [0.0, 6.0]

julia> cubic_interp(x, y, xq; extrap=NoExtrap())  # vector query (xq includes out-of-domain points)
ERROR: DomainError with -1.5:
query point outside interpolation domain [0.0, 6.0]
```

Only interior queries succeed:

```@example extrap
yq = cubic_interp(x, y, range(x[1], x[end], 200); extrap=NoExtrap())

plot(title="NoExtrap()", xlabel="x", ylabel="y", legend=:topright, xlims=(x[1] - 1.5, x[end] + 1.5)) # hide
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label="out of domain") # hide
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing) # hide
plot!(range(x[1], x[end], 200), yq, label="spline", linewidth=2) # hide
scatter!(x, y, label="data", markersize=7, color=:black) # hide
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing) # hide
```

## `ClampExtrap()`

Returns boundary values: `y[1]` for left, `y[end]` for right.

```@example extrap
yq = cubic_interp(x, y, xq; extrap=ClampExtrap())

plot(title="ClampExtrap()", xlabel="x", ylabel="y", legend=:topright) # hide
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label="out of domain") # hide
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing) # hide
plot!(xq, yq, label="spline", linewidth=2) # hide
scatter!(x, y, label="data", markersize=7, color=:black) # hide
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing) # hide
```

## `ExtendExtrap()`

Extends the boundary polynomial beyond the domain.

```@example extrap
yq = cubic_interp(x, y, xq; extrap=ExtendExtrap())

plot(title="ExtendExtrap()", xlabel="x", ylabel="y", legend=:topright) # hide
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label="out of domain") # hide
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing) # hide
plot!(xq, yq, label="spline", linewidth=2) # hide
scatter!(x, y, label="data", markersize=7, color=:black) # hide
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing) # hide
```

## `WrapExtrap()`

Wraps coordinates periodically:

```math
S(x + \tau) = S(x), \quad \tau = x_{\text{end}} - x_1
```

This is **purely coordinate mapping**—it does not enforce any physical conditions at the boundary. The spline may have discontinuities in value, slope, or curvature at the wrap point.

!!! note "For Smooth Periodicity"
    If you need C² continuity at the periodic boundary, use [`bc=PeriodicBC()`](interpolation/cubic.md) with `cubic_interp`. This enforces ``S(x_1) = S(x_{\text{end}})``, ``S'(x_1) = S'(x_{\text{end}})``, and ``S''(x_1) = S''(x_{\text{end}})``.

```@example extrap
yq = cubic_interp(x, y, xq; extrap=WrapExtrap())

plot(title="WrapExtrap()", xlabel="x", ylabel="y", legend=:topright) # hide
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label="out of domain") # hide
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing) # hide
plot!(xq, yq, label="spline", linewidth=2) # hide
scatter!(x, y, label="data", markersize=7, color=:black) # hide
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing) # hide
```

## Comparison

```@example extrap
# All three modes on same plot
y_const = cubic_interp(x, y, xq; extrap=ClampExtrap())
y_ext   = cubic_interp(x, y, xq; extrap=ExtendExtrap())
y_wrap  = cubic_interp(x, y, xq; extrap=WrapExtrap())

plot(title="Extrapolation Comparison", xlabel="x", ylabel="y", legend=:topright, size=(700, 400)) # hide
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label=nothing) # hide
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing) # hide
plot!(xq, y_const, label="ClampExtrap", linewidth=2) # hide
plot!(xq, y_ext, label="ExtendExtrap", linewidth=2, linestyle=:dash) # hide
plot!(xq, y_wrap, label="WrapExtrap", linewidth=2, linestyle=:dashdot) # hide
scatter!(x, y, label="data", markersize=7, color=:black) # hide
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing) # hide
```

## Summary

| Mode | Behavior | Use Case |
|------|----------|----------|
| `NoExtrap()` | `DomainError` | Strict domain enforcement (default) |
| `ClampExtrap()` | Returns boundary values | Physical constraints |
| `ExtendExtrap()` | Continues boundary polynomial | Smooth continuation |
| `WrapExtrap()` | Wraps coordinates (no smoothness) | Cyclic data (see [`PeriodicBC`](interpolation/cubic.md) for C² continuity) |

## See Also

- **[Derivatives](interpolation/derivatives.md)**: Analytical derivatives with extrapolation
- **[Cubic Splines](interpolation/cubic.md)**: Boundary conditions and smooth periodicity
