# Boundary Conditions (ND)

In ND, boundary conditions are specified **per-axis** via Tuples. A single BC value is broadcast to all axes.

!!! note "Method Applicability"
    - **Constant** / **Linear**: No BC needed
    - **Quadratic**: One BC per axis (`Left(...)` or `Right(...)`)
    - **Cubic**: Paired BCs per axis (`NaturalBC()`, `PeriodicBC()`, `BCPair(...)`, etc.)

    For BC type details, see the [1D Boundary Conditions](../boundary-conditions/overview.md).

---

## Broadcast vs Per-Axis

```julia
# Broadcast: same BC on all axes
itp = cubic_interp((x, y, z), data; bc=NaturalBC())
# Equivalent to: bc=(NaturalBC(), NaturalBC(), NaturalBC())

# Per-axis: different BC per axis
itp = cubic_interp((x, y, z), data;
    bc=(NaturalBC(), PeriodicBC(), ClampedBC()))
```

---

## Cubic ND

All [1D cubic BCs](../boundary-conditions/overview.md) are available per-axis:

```julia
# Natural in x, periodic in y
itp = cubic_interp((x, y), data;
    bc=(NaturalBC(), PeriodicBC()))

# Custom: known slope at x-left, auto-fit in y
itp = cubic_interp((x, y), data;
    bc=(BCPair(Deriv1(0.0), CubicFit()), CubicFit()))
```

!!! note "PeriodicBC + Extrapolation"
    `PeriodicBC()` on an axis automatically forces `extrap=:wrap` on that axis.

---

## Quadratic ND

Quadratic BCs use `Left(...)` / `Right(...)` wrappers per-axis:

```julia
# Broadcast: same BC on all axes
itp = quadratic_interp((x, y), data; bc=Left(QuadraticFit()))

# Per-axis
itp = quadratic_interp((x, y), data;
    bc=(Left(QuadraticFit()), Right(Deriv1(0.0))))
```

| BC | Description |
|:---|:------------|
| `Left(QuadraticFit())` | 3-point auto-fit at left (default) |
| `Right(Deriv1(v))` | Known slope at right |
| `MinCurvFit()` | Minimize total curvature |

---

## See Also

- **[1D Boundary Conditions](../boundary-conditions/overview.md)** — Full BC type reference
- **[Overview](overview.md)** — ND API introduction
