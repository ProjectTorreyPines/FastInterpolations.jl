# Constant Interpolation API

## Overview

### One-shot (construction + evaluation)

| Function | Description |
|----------|-------------|
| `constant_interp(x, y, xq)` | Constant interpolation at point(s) `xq` |
| `constant_interp(x, y, xq; side=:left)` | With side mode (`:nearest`, `:left`, `:right`) |
| `constant_interp!(out, x, y, xq)` | In-place constant interpolation |
| `constant_interp!(out, x, y, xq; side)` | In-place with side mode |

### Re-usable interpolant

| Function | Description |
|----------|-------------|
| `itp = constant_interp(x, y)` | Create constant interpolant |
| `itp = constant_interp(x, y; side=:left)` | Create with side mode |
| `itp(xq)` | Evaluate at point(s) `xq` |
| `itp(out, xq)` | Evaluate at `xq`, store result in `out` |

### Derivatives

| Function | Description |
|----------|-------------|
| `constant_interp(x, y, xq; deriv=1)` | First derivative (always 0) |
| `constant_interp(x, y, xq; deriv=2)` | Second derivative (always 0) |
| `deriv1(itp)` | First derivative view |
| `deriv2(itp)` | Second derivative view |
| `deriv3(itp)` | Third derivative view (always 0) |

---

## Functions

```@docs
constant_interp
constant_interp!
```

## Interpolant Type

```@docs
ConstantInterpolant
ConstantInterpolantND
```

## Derivative Views

See [Derivatives](../interpolation/derivatives.md) for `deriv1`, `deriv2`, `deriv3` API reference.
