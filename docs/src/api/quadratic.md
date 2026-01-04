# Quadratic Interpolation API

## Overview

### One-shot (construction + evaluation)

| Function | Description |
|----------|-------------|
| `quadratic_interp(x, y, xq)` | Quadratic interpolation at point(s) `xq` |
| `quadratic_interp(x, y, xq; bc=...)` | With boundary condition |
| `quadratic_interp!(out, x, y, xq)` | In-place quadratic interpolation |
| `quadratic_interp!(out, x, y, xq; bc)` | In-place with BC |

### Re-usable interpolant

| Function | Description |
|----------|-------------|
| `itp = quadratic_interp(x, y)` | Create quadratic interpolant |
| `itp = quadratic_interp(x, y; bc=...)` | Create with boundary condition |
| `itp(xq)` | Evaluate at point(s) `xq` |
| `itp(out, xq)` | Evaluate at `xq`, store result in `out` |

### Derivatives

| Function | Description |
|----------|-------------|
| `quadratic_interp(x, y, xq; deriv=1)` | First derivative (continuous) |
| `quadratic_interp(x, y, xq; deriv=2)` | Second derivative (piecewise constant) |
| `deriv1(itp)` | First derivative view |
| `deriv2(itp)` | Second derivative view |

---

## Functions

```@docs
quadratic_interp
quadratic_interp!
```

## Interpolant Type

```@docs
QuadraticInterpolant
```

## Boundary Condition Types

```@docs
ParabolaFit
MinCurvFit
```
