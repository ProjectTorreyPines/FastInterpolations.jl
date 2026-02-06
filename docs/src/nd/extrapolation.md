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

## Broadcast vs Per-Axis

```julia
# Broadcast: same mode on all axes
itp = cubic_interp((x, y), data; extrap=:constant)

# Per-axis: different mode per axis
itp = cubic_interp((x, y), data;
    extrap=(:extension, :wrap))
```

### Example: Physical Boundary

A common pattern in physics simulations — extend in the radial direction, wrap in the angular direction:

```julia
r = range(0.1, 1.0, 50)       # radial
θ = range(0.0, 2π, 60)        # angular (periodic)

data = [sin(ri * cos(θj)) for ri in r, θj in θ]

itp = cubic_interp((r, θ), data;
    bc=(NaturalBC(), PeriodicBC()),
    extrap=(:extension, :wrap))
```

!!! note "PeriodicBC and :wrap"
    When `bc=PeriodicBC()` is set on an axis, `extrap=:wrap` is automatically enforced for that axis, regardless of the `extrap` argument.

---

## See Also

- **[1D Extrapolation](../extrapolation.md)** — Detailed mode descriptions with visualizations
- **[Overview](overview.md)** — ND API introduction
