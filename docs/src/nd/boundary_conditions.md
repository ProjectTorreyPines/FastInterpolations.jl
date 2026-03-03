# Boundary Conditions (ND)

In ND, boundary conditions are specified **per-axis** via Tuples. A single BC value is broadcast to all axes.

!!! note "Method Applicability"
    - **Constant** / **Linear**: No BC needed
    - **Quadratic**: One BC per axis (`Left(...)` or `Right(...)`)
    - **Cubic**: Paired BCs per axis (`CubicFit()`, `ZeroCurvBC()`, `PeriodicBC()`, `BCPair(...)`, etc.)

    For BC type details, see the [1D Boundary Conditions](../boundary-conditions/overview.md).

---

## Example Data
```@example nd_boundary
using FastInterpolations 
x = range(1, 10, length = 10)
y = range(0, 2pi, length = 5) 
z = [0, 1, 3, 5, 10]
data2d = [cos(xi)*cos(yi) for xi in x, yi in y] # Real 2D data
data3d = [cos(xi)*cos(yi)+ zi*1im for xi in x, yi in y, zi in z] # Complex 3D data
nothing #hide
```

## Broadcast vs Per-Axis

```@example nd_boundary
# Broadcast: same BC on all axes (CubicFit is default, so this is equivalent to no bc kwarg)
itp = cubic_interp((x, y, z), data3d; bc=CubicFit())
# Equivalent to: bc=(CubicFit(), CubicFit(), CubicFit())
```

```@example nd_boundary
# Per-axis: different BC per axis
itp = cubic_interp((x, y, z), data3d;
    bc=(CubicFit(), PeriodicBC(), ZeroCurvBC()))
```

---

## Cubic ND

All [1D cubic BCs](../boundary-conditions/overview.md) are available per-axis:

```@example nd_boundary
# ZeroCurv in x, periodic in y
bc = (ZeroCurvBC(), PeriodicBC())
itp = cubic_interp((x, y), data2d; bc=bc)
```
!!! note "PeriodicBC + Extrapolation"
    `PeriodicBC()` on an axis automatically forces `WrapExtrap()` on that axis.

```@example nd_boundary
# Custom: known slope at x-left, auto-fit in y
bc = (BCPair(Deriv1(0.0), CubicFit()), CubicFit())
itp = cubic_interp((x, y), data2d; bc=bc)
```

---

## Quadratic ND

Quadratic BCs use `Left(...)` / `Right(...)` wrappers per-axis:

```@example nd_boundary
# Broadcast: same BC on all axes
itp = quadratic_interp((x, y), data2d; bc=Left(QuadraticFit()))

# Per-axis
itp = quadratic_interp((x, y), data2d;
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
