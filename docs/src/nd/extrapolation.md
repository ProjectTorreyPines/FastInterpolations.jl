# Extrapolation (ND)

Extrapolation in ND works per-axis, using the same modes as [1D extrapolation](../extrapolation.md).

---

## Modes

| Mode | Behavior |
|:-----|:---------|
| `:none` | `DomainError` (default) |
| `:constant` | Clamp to boundary value |
| `:extension` | Extend boundary polynomial |
| `:wrap` | Wrap coordinates periodically |

---

## Visualization

All examples use a **Franke-lite** test function — two Gaussian peaks arranged diagonally on ``[0, 1]^2``:

```@example nd_extrap
using FastInterpolations
using Plots # hide

f(x, y) = 0.75exp(-((4x-1)^2 + (4y-1)^2)) + 0.75exp(-((4x-3)^2 + (4y-3)^2)/2)

xs = [0.0, 0.1, 0.4, 0.5, 0.82, 1.0]
ys = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]

data = [f(xi, yj) for xi in xs, yj in ys]

kw = (size=(480, 400), xlims=(-0.3,1.3), ylims=(-0.3,1.3))
nothing # hide
```

### `extrap=:none` (Default)

Domain-only view — queries outside ``[0, 1]^2`` throw `DomainError`:

```@example nd_extrap
itp = cubic_interp((xs, ys), data; bc=CubicFit(), extrap=:none)
plot(itp; kw...)
```

### `extrap=:constant`

Boundary values are held constant — flat color bands extend from each edge:

```@example nd_extrap
itp = cubic_interp((xs, ys), data; bc=CubicFit(), extrap=:constant)
plot(itp; kw...)
```

### `extrap=:extension`

Boundary polynomials extend beyond the domain — the cubic pieces continue their natural curvature:

```@example nd_extrap
itp = cubic_interp((xs, ys), data; bc=CubicFit(), extrap=:extension)
plot(itp; kw...)
```

### `extrap=:wrap`

Coordinates wrap periodically. Since the data is not periodic, a visible **seam** appears at the boundaries:

```@example nd_extrap
itp = cubic_interp((xs, ys), data; bc=CubicFit(), extrap=:wrap)
plot(itp; kw...)
```

!!! tip "Wrap vs Periodic — coordinate repetition is not periodic interpolation"
    `:wrap` alone simply maps coordinates back into the domain via modular arithmetic.
    The interpolant itself is **unaware** of periodicity, so a C⁰ seam (jump in derivatives) is expected at the boundary.

    For **true periodic interpolation**, use [`PeriodicBC()`](boundary_conditions.md) with cubic splines.
    `PeriodicBC()` enforces **C² continuity** at the wrap boundary — matching function value,
    first derivative, and second derivative across the period — producing a seamless, smooth result.

    When `bc=PeriodicBC()` is set on an axis, `extrap=:wrap` is automatically applied for that axis.

---

## Per-Axis Combinations

Different extrapolation modes can be assigned independently to each axis via a tuple:

```julia
itp = cubic_interp((x, y), data; extrap=(:constant, :extension))
```

### Constant × None

Constant extrapolation on x₁, strict domain on x₂ — the heatmap extends **only horizontally**:

```@example nd_extrap
itp = cubic_interp((xs, ys), data; bc=CubicFit(), extrap=(:constant, :none))
plot(itp; kw...)
```

### Constant x Wrap

Constant extrapolation on x₁, wrap periodically on x₂:

```@example nd_extrap
itp = cubic_interp((xs, ys), data; bc=CubicFit(), extrap=(:constant, :wrap))
plot(itp; kw...)
```

---

## See Also

- **[1D Extrapolation](../extrapolation.md)** — Detailed mode descriptions with visualizations
- **[Overview](overview.md)** — ND API introduction
