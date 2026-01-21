# Linear Interpolation API

## Overview

### One-shot (construction + evaluation)

| Function | Description |
|----------|-------------|
| `linear_interp(x, y, xq)` | Linear interpolation at point(s) `xq` |
| `linear_interp!(out, x, y, xq)` | In-place linear interpolation |

### Re-usable interpolant

| Function | Description |
|----------|-------------|
| `itp = linear_interp(x, y)` | Create linear interpolant |
| `itp(xq)` | Evaluate at point(s) `xq` |
| `itp(out, xq)` | Evaluate at `xq`, store result in `out` |

### Derivatives

| Function | Description |
|----------|-------------|
| `linear_interp(x, y, xq; deriv=1)` | First derivative (piecewise constant) |
| `linear_interp(x, y, xq; deriv=2)` | Second derivative (always 0) |
| `deriv1(itp)` | First derivative view |
| `deriv2(itp)` | Second derivative view |
| `deriv3(itp)` | Third derivative view (always 0) |

---

## Functions

```@docs
linear_interp
linear_interp!
```

## Interpolant Type

```@docs
LinearInterpolant
```

## Derivative Views

See [Derivatives](../interpolation/derivatives.md) for `deriv1`, `deriv2`, `deriv3` API reference.
