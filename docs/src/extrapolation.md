# Extrapolation

Extrapolation controls behavior when query points fall outside the data domain `[x[1], x[end]]`.

## Overview

All extrapolation modes are concrete subtypes of [`AbstractExtrap`](@ref). Pass them via the `extrap` keyword argument:

```julia-repl
# Using Extrap() factory (recommended)
cubic_interp(x, y, xq; extrap=Extrap(:clamp))
linear_interp(x, y, xq; extrap=Extrap(:extend))

# Using direct types (also supported)
cubic_interp(x, y, xq; extrap=ClampExtrap())
linear_interp(x, y, xq; extrap=ExtendExtrap())

# Interpolant: stored extrap is fixed at creation; a per-call `InBounds()` skips the domain check
itp = cubic_interp(x, y; extrap=Extrap(:extend))
itp(xq)                      # uses ExtendExtrap
itp(xq; extrap=InBounds())   # in-domain fast-path (query must be in-domain)
```

!!! tip "Factory Functions"
    The [`Extrap()`](@ref factory_functions) factory provides a single discoverable entry point. For ND per-axis configuration, use the multi-arg form: `Extrap(:extend, :clamp, :none)`. See [Factory Functions](@ref factory_functions) for details.

All interpolation methods (`linear_interp`, `quadratic_interp`, `cubic_interp`, `constant_interp`) support the same extrapolation modes.

| Type | Behavior |
|------|----------|
| [`NoExtrap()`](@ref NoExtrap) | Throws `DomainError` (default) |
| [`ClampExtrap()`](@ref ClampExtrap) | Returns boundary values |
| [`ExtendExtrap()`](@ref ExtendExtrap) | Extends boundary polynomial |
| [`WrapExtrap()`](@ref WrapExtrap) | Wraps coordinates periodically (no smoothness enforced) |
| [`FillExtrap(value)`](@ref FillExtrap) | Returns a constant fill value (e.g. `0.0`, `NaN`) |
| [`InBounds()`](@ref InBounds) | Skip domain checks (caller guarantees in-domain) |

```
AbstractExtrap
├── NoExtrap         # DomainError on out-of-domain (default)
├── ClampExtrap      # clamp to y[1] / y[end]
├── ExtendExtrap     # continue boundary polynomial
├── WrapExtrap       # modular coordinate mapping
├── FillExtrap{T}    # return constant fill value
└── InBounds         # skip domain checks (advanced/internal)
```

`InBounds` additionally accepts endpoint keywords that narrow the promised interval:
`InBounds(last = :exclusive)` promises `first(x) ≤ xq < last(x)` and unlocks a leaner
direct search on unit-step range grids (`1:n`, `Base.OneTo`, offset unit ranges). Like
`Base.@inbounds`, violating the promise is undefined behavior — see the
[`InBounds`](@ref) docstring for the full contract table.

## Examples

```@example extrap
using FastInterpolations
using Plots

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

Create an interpolant and plot directly — the recipe renders only the interior domain:

```@example extrap
itp = cubic_interp(x, y; extrap=NoExtrap())
plot(itp)
```

## `ClampExtrap()`

Returns boundary values: `y[1]` for left, `y[end]` for right.

```julia
# One-shot evaluation
yq = cubic_interp(x, y, xq; extrap=ClampExtrap())
```

Create an interpolant and plot directly:

```@example extrap
itp = cubic_interp(x, y; extrap=ClampExtrap())
plot(itp)
```

## `ExtendExtrap()`

Extends the boundary polynomial beyond the domain.

```julia
# One-shot evaluation
yq = cubic_interp(x, y, xq; extrap=ExtendExtrap())
```

Create an interpolant and plot directly:

```@example extrap
itp = cubic_interp(x, y; extrap=ExtendExtrap())
plot(itp)
```

## `WrapExtrap()`

Wraps coordinates periodically:

```math
S(x + \tau) = S(x), \quad \tau = x_{\text{end}} - x_1
```

This is **purely coordinate mapping**—it does not enforce any physical conditions at the boundary. The spline may have discontinuities in value, slope, or curvature at the wrap point.

!!! note "For Smooth Periodicity"
    If you need C² continuity at the periodic boundary, use [`bc=PeriodicBC()`](interpolation/cubic.md) with `cubic_interp`. This enforces ``S(x_1) = S(x_{\text{end}})``, ``S'(x_1) = S'(x_{\text{end}})``, and ``S''(x_1) = S''(x_{\text{end}})``.

```julia
# One-shot evaluation
yq = cubic_interp(x, y, xq; extrap=WrapExtrap())
```

Create an interpolant and plot directly:

```@example extrap
itp = cubic_interp(x, y; extrap=WrapExtrap())
plot(itp)
```

## `FillExtrap(value)`

Returns a constant fill value for all out-of-domain queries. The fill value can be any numeric type — `0.0`, `42.0`, `NaN`, etc.

```julia
cubic_interp(x, y; extrap=FillExtrap(0.0))       # zero outside domain
cubic_interp(x, y; extrap=FillExtrap(NaN))        # NaN outside domain
cubic_interp(x, y; extrap=Extrap(:fill; fill_value=0)) # factory form
```

Derivatives in the out-of-domain region are exactly zero (flat constant), and `integrate` accumulates `fill_value × interval_length` over any OOB segment.

```julia
# One-shot evaluation
yq = cubic_interp(x, y, xq; extrap=FillExtrap(0.0))
```

Create an interpolant and plot directly:

```@example extrap
itp = cubic_interp(x, y; extrap=FillExtrap(0.0))
plot(itp)
```

## Comparison

```@example extrap
# All modes on same plot
y_clamp = cubic_interp(x, y, xq; extrap=ClampExtrap())
y_ext   = cubic_interp(x, y, xq; extrap=ExtendExtrap())
y_wrap  = cubic_interp(x, y, xq; extrap=WrapExtrap())
y_fill  = cubic_interp(x, y, xq; extrap=FillExtrap(0.0))

plot(title="Extrapolation Comparison", xlabel="x", ylabel="y",
     legend=:topright, size=(700, 400))
vspan!([x[1] - 1.5, x[1]], alpha=0.1, color=:gray, label=nothing)
vspan!([x[end], x[end] + 1.5], alpha=0.1, color=:gray, label=nothing)
plot!(xq, y_clamp, label="ClampExtrap", linewidth=2)
plot!(xq, y_ext,   label="ExtendExtrap", linewidth=2, linestyle=:dash)
plot!(xq, y_wrap,  label="WrapExtrap", linewidth=2, linestyle=:dashdot)
plot!(xq, y_fill,  label="FillExtrap(0.0)", linewidth=2, linestyle=:dot)
scatter!(x, y, label="data", markersize=7, color=:black)
vline!([x[1], x[end]], color=:gray, linestyle=:dot, alpha=0.5, label=nothing)
```

## Summary

| Mode | Behavior | Use Case |
|------|----------|----------|
| `NoExtrap()` | `DomainError` | Strict domain enforcement (default) |
| `ClampExtrap()` | Returns boundary values | Physical constraints |
| `ExtendExtrap()` | Continues boundary polynomial | Smooth continuation |
| `WrapExtrap()` | Wraps coordinates (no smoothness) | Cyclic data (see [`PeriodicBC`](interpolation/cubic.md) for C² continuity) |
| `FillExtrap(v)` | Returns constant `v` outside domain | Masking (`NaN`), zero-padding, sentinel values |
| `InBounds()` | Skip domain checks | Pre-validated queries, batch inner loops |
| `InBounds(last = :exclusive)` | Skip checks + promise `xq < last(x)` | Fastest unit-step range search for strictly-interior queries |

## See Also

- **[Derivatives](interpolation/derivatives.md)**: Analytical derivatives with extrapolation
- **[Cubic Splines](interpolation/cubic.md)**: Boundary conditions and smooth periodicity
