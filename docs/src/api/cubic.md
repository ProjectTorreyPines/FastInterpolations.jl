# Cubic Spline API

## Overview

### One-shot (construction + evaluation)

| Function | Description |
|----------|-------------|
| `cubic_interp(x, y, xq)` | Cubic spline at point(s) `xq` (default: NaturalBC) |
| `cubic_interp(x, y, xq; bc=...)` | With specified BC |
| `cubic_interp!(out, x, y, xq; bc=...)` | In-place version |

### Re-usable interpolant

| Function | Description |
|----------|-------------|
| `itp = cubic_interp(x, y; bc=...)` | Create interpolant |
| `itp(xq)` | Evaluate at point(s) `xq` |
| `itp(out, xq)` | Evaluate at `xq`, store result in `out` |

### Derivatives

| Function | Description |
|----------|-------------|
| `cubic_interp(x, y, xq; deriv=1)` | First derivative (continuous) |
| `cubic_interp(x, y, xq; deriv=2)` | Second derivative (continuous) |
| `cubic_interp(x, y, xq; deriv=3)` | Third derivative (piecewise constant) |
| `deriv1(itp)` | First derivative view |
| `deriv2(itp)` | Second derivative view |
| `deriv3(itp)` | Third derivative view |

---

## Functions

```@docs
cubic_interp
cubic_interp!
```

## Interpolant Type

```@docs
CubicInterpolant
CubicSplineCache
```

## Cache Management

```@docs
set_cubic_cache_size!
get_cubic_cache_size
clear_cubic_cache!
```

## Derivative Views

See [Derivatives](../interpolation/derivatives.md) for `deriv1`, `deriv2`, `deriv3` API reference.
