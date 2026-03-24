# Unified API: `interp` / `interp!`

`interp` is a **unified entry point** for N-dimensional interpolation with per-axis method control.

```@example unified
using FastInterpolations

x = range(0.0, 2pi, 50)
y = range(0.0, 1.0, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]
nothing  # hide
```

## Homogeneous Methods

When all axes use the same method, `interp` delegates to the existing optimized type. No performance penalty — identical to calling `cubic_interp`, `linear_interp`, etc. directly.

```@example unified
# These are equivalent:
itp1 = cubic_interp((x, y), data)
itp2 = interp((x, y), data; method=CubicInterp())

itp1((1.0, 0.5)) ≈ itp2((1.0, 0.5))
```

A scalar `method` is broadcast to all axes.

## Heterogeneous Methods

Specify a tuple to use different methods per axis:

```@example unified
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()))
itp((1.0, 0.5))
```

All existing keywords (`extrap`, `search`, `deriv`, `hint`) work the same way — pass a scalar to broadcast or a tuple for per-axis control.

```@example unified
z = range(0.0, 5.0, 20)
data3d = [sin(xi) * cos(yj) * exp(-zk) for xi in x, yj in y, zk in z]

itp3d = interp((x, y, z), data3d;
    method = (CubicInterp(bc=PeriodicBC()), QuadraticInterp(), LinearInterp()),
    extrap = (WrapExtrap(), ClampExtrap(), NoExtrap()))
itp3d((1.0, 0.5, 2.0))
```

---

## `NoInterp` and `GridIdx`: Discrete Axes

Some axes represent discrete indices (ensemble member, vertical level, time snapshot) rather than continuous coordinates. `NoInterp()` skips interpolation on that axis — at build time and eval time.

```@example unified
# Axis 2 is discrete: no interpolation, queried by grid index
itp_ni = interp((x, y), data; method=(CubicInterp(), NoInterp()))

# Query: real coordinate for interpolated axes, GridIdx(k) for discrete axes
itp_ni((0.5, GridIdx(10)))
```

**`GridIdx(k)`** wraps an integer grid index. `NoInterp` axes must be queried with `GridIdx`; interpolated axes take real-valued coordinates as usual.

```@example unified
itp_ni(0.5, GridIdx(10))  # vararg form also works
```

### Why use `NoInterp`?

- **One build, many slices**: build the interpolant once, vary the slice index at query time
- **Dimension reduction**: the eval pipeline operates on fewer dimensions (e.g., 3D → 2D), reducing both compute and memory
- **Skipped precomputation**: no tridiagonal solve or derivative storage on discrete axes

```@example unified
# 3D data: interpolate x+z, select y by index
itp_3ni = interp((x, y, z), data3d;
    method = (CubicInterp(), NoInterp(), LinearInterp()))

# Loop over all y-slices — no rebuild
[itp_3ni((1.0, GridIdx(k), 2.0)) for k in 1:5]
```

### Derivatives and Vector Calculus

Derivatives on `NoInterp` axes return zero (the axis has no spatial interpolation). All other axes compute real derivatives:

```@example unified
itp_ni((0.5, GridIdx(10)); deriv=(DerivOp(1), DerivOp(0)))  # df/dx on the slice
```

```@example unified
gradient(itp_ni, (0.5, GridIdx(10)))
```

```@example unified
laplacian(itp_ni, (0.5, GridIdx(10)))
```

### One-Shot

```@example unified
interp((x, y), data, (0.5, GridIdx(10)); method=(CubicInterp(), NoInterp()))
```

### Batch Queries

For batch evaluation with fixed `GridIdx` slices, use `interp_batch_grididx!`:

```@example unified
output = zeros(5)
xq = collect(range(0.5, 5.0, 5))
interp_batch_grididx!(output, (x, y), data, (xq, GridIdx(10));
    method=(CubicInterp(), NoInterp()))
output
```

---

## Available Methods

| Method | Description | Derivative Order |
|:-------|:-----------|:----------------|
| `CubicInterp(; bc=CubicFit())` | C2 cubic spline | Up to 3rd |
| `QuadraticInterp(; bc=...)` | C1 quadratic spline | Up to 2nd |
| `LinearInterp()` | Piecewise linear | 1st (2nd+ = 0) |
| `ConstantInterp(; side=NearestSide())` | Step function | Always 0 |
| `NoInterp()` | Discrete axis — index-based, no interpolation | N/A |

---

## Visual Comparison

Per-axis method mixing on a 6x7 non-uniform grid for $f(x,y) = \sin(2\pi x)\cos(2\pi y)$:

![Heterogeneous 2D Comparison](../images/hetero_2d_comparison.png)

Linear x Linear (top-left) shows faceted cells. Adding cubic smoothing per axis progressively improves the result — Cubic x Cubic (bottom-right) captures the extrema accurately.
