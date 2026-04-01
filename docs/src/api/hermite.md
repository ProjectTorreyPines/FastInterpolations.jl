# Local Cubic Hermite API

Four C^1 interpolation methods sharing the same Hermite basis kernel, differing only in how slopes are determined.

## Overview

### Hermite (user-supplied slopes)

| Function | Description |
|----------|-------------|
| `cubic_interp(x, Hermite(y, dy), xq)` | Hermite interpolation at point(s) `xq` |
| `cubic_interp!(out, x, Hermite(y, dy), xq)` | In-place Hermite interpolation |
| `itp = cubic_interp(x, Hermite(y, dy))` | Create callable interpolant |

### PCHIP (monotone-preserving)

| Function | Description |
|----------|-------------|
| `pchip_interp(x, y, xq)` | PCHIP interpolation at point(s) `xq` |
| `pchip_interp!(out, x, y, xq)` | In-place PCHIP interpolation |
| `itp = pchip_interp(x, y)` | Create callable interpolant |

### Cardinal / Catmull-Rom

| Function | Description |
|----------|-------------|
| `cardinal_interp(x, y, xq; tension=0.0)` | Cardinal spline at point(s) `xq` |
| `cardinal_interp!(out, x, y, xq; tension=0.0)` | In-place cardinal interpolation |
| `itp = cardinal_interp(x, y; tension=0.0)` | Create callable interpolant |

### Akima (outlier-robust)

| Function | Description |
|----------|-------------|
| `akima_interp(x, y, xq)` | Akima interpolation at point(s) `xq` |
| `akima_interp!(out, x, y, xq)` | In-place Akima interpolation |
| `itp = akima_interp(x, y)` | Create callable interpolant |

---

## Data Wrapper

```@docs
Hermite
```

## Functions — Hermite

Hermite interpolation uses `cubic_interp` with a `Hermite(y, dy)` wrapper.
See the [Cubic API](cubic.md) for the `cubic_interp` / `cubic_interp!` reference.

## Functions — PCHIP

```@docs
pchip_interp
pchip_interp!
```

## Functions — Cardinal

```@docs
cardinal_interp
cardinal_interp!
```

## Functions — Akima

```@docs
akima_interp
akima_interp!
```

## Interpolant Types

```@docs
CubicHermiteInterpolant1D
PchipInterpolant1D
CardinalInterpolant1D
AkimaInterpolant1D
```
