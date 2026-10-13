# Polyharmonic Spline API

## Overview

### One-shot (construction + evaluation)

| Function | Description |
|----------|-------------|
| `phs_interp(grids, data, (xq, yq, ...))` | PHS value at a single N-D point |
| `phs_interp(grids, data, queries)` | PHS values at many points (SoA tuple of vectors or AoS vector) |
| `phs_interp!(out, grids, data, queries)` | In-place version |

### Re-usable interpolant

| Function | Description |
|----------|-------------|
| `itp = phs_interp(grids, data; stencil_size=8, degree=3, ...)` | Create interpolant |
| `itp((x, y, ...))` | Evaluate at a single point |
| `itp(queries)` / `itp(out, queries)` | Batch evaluation (allocating / in-place) |
| `itp((x, y); deriv=(DerivOp(1), DerivOp(0)))` | Partial derivative (per-axis `DerivOp` tuple) |

### Log-transform

| Function | Description |
|----------|-------------|
| `phs_interp(grids, data; log_reference = c)` | Interpolate `log(data/c)` for a nonzero constant `c` with the sign of `data`, return `data` scale |
| `phs_interp(grids, data; log_reference = ρ₀)` | Interpolate `log(data/ρ₀)` for a callable `ρ₀(q)`, such as an interpolant; derivative queries call `ρ₀(q; deriv = ops)` |

See [Polyharmonic Splines (PHS)](../interpolation/phs.md) for the method, tuning guidance, and worked examples.

---

## Functions

```@docs
phs_interp
phs_interp!
```

## Interpolant Type

```@docs
PHSInterpolantND
```
