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
nothing  # hide
```

A scalar `method` is broadcast to all axes.

## Heterogeneous Methods

Specify a tuple to use different methods per axis:

```@example unified
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()))
itp((1.0, 0.5))
nothing  # hide
```

All existing keywords (`extrap`, `search`, `deriv`, `hint`) work the same way — pass a scalar to broadcast or a tuple for per-axis control.

```@example unified
z = range(0.0, 5.0, 20)
data3d = [sin(xi) * cos(yj) * exp(-zk) for xi in x, yj in y, zk in z]

itp = interp((x, y, z), data3d;
    method = (CubicInterp(bc=PeriodicBC()), QuadraticInterp(), LinearInterp()),
    extrap = (WrapExtrap(), ClampExtrap(), NoExtrap()))
itp((1.0, 0.5, 2.0))
nothing  # hide
```

---

## `GridIdx`: Index-Based Queries

`GridIdx(k)` queries an axis by grid index instead of coordinate value. It works with **any interpolant type** — no special construction needed.

```@example unified
# Any interpolant: query axis 2 at its 10th grid point
itp = cubic_interp((x, y), data)
itp((0.5, GridIdx(10)))
nothing  # hide
```

### One-Shot Slicing

`GridIdx` is especially powerful in one-shot evaluation. Each `GridIdx` axis is sliced out of the data first — reducing the problem to fewer dimensions with **no interpolant construction** on those axes.

```@example unified
# One-shot: slice axis 2 at index 10, cubic on axis 1
interp((x, y), data, (0.5, GridIdx(10)); method=CubicInterp())
nothing  # hide
```

For a 50×100×20 cubic interpolation, the speedup from slicing is dramatic:

```julia
# Full 3D cubic: ~2.7 ms (one-shot construction + eval on all axes)
interp(grids, data3D, (20.5, 5.0, 1.0); method=CubicInterp())

# Slice 1 axis → 2D cubic: ~12 μs (224× faster)
interp(grids, data3D, (20.5, GridIdx(5), 1.0); method=CubicInterp())

# Slice 2 axes → 1D cubic: ~430 ns (6200× faster)
interp(grids, data3D, (20.5, GridIdx(5), GridIdx(1)); method=CubicInterp())
```

This works because `GridIdx` bypasses the one-shot construction (e.g., tridiagonal solve for cubic) on sliced axes. Any single method works — `GridIdx` pre-slices the data and delegates to the appropriate lower-dimensional one-shot.

### Derivatives and Vector Calculus

`gradient`, `hessian`, and `laplacian` also accept `GridIdx`:

```@example unified
itp = cubic_interp((x, y), data)
gradient(itp, (0.5, GridIdx(10)))   # full (df/dx, df/dy) at grid point
nothing  # hide
```

### Batch Queries

Batch queries with `GridIdx` work through the standard `interp!`:

```@example unified
output = zeros(5)
xq = collect(range(0.5, 5.0, 5))
interp!(output, (x, y), data, (xq, GridIdx(10)); method=CubicInterp())
nothing  # hide
```

---

## `NoInterp`: Discrete Axes

`NoInterp()` declares an axis as discrete at build time — no interpolation is performed on that axis, ever. It must be paired with `GridIdx` queries.

```@example unified
# Axis 2 is discrete: no interpolation, queried by grid index
itp = interp((x, y), data; method=(CubicInterp(), NoInterp()))
itp((0.5, GridIdx(10)))
nothing  # hide
```

### Why use `NoInterp` over plain `GridIdx`?

Plain `GridIdx` (previous section) slices at query time but still builds the full interpolant. `NoInterp` goes further:

- **Skipped precomputation**: no tridiagonal solve, no derivative storage on discrete axes
- **Smaller memory footprint**: HeteroPartials array is reduced
- **One build, many slices**: build once, vary the slice index at query time

```@example unified
# 3D data: interpolate x+z, select y by index
itp = interp((x, y, z), data3d;
    method = (CubicInterp(), NoInterp(), LinearInterp()))

# Loop over all y-slices — no rebuild
for k in 1:5
    itp((1.0, GridIdx(k), 2.0))
end
nothing  # hide
```

### NoInterp Derivatives

Derivatives on `NoInterp` axes return zero (the axis has no spatial interpolation):

```@example unified
itp = interp((x, y), data; method=(CubicInterp(), NoInterp()))
gradient(itp, (0.5, GridIdx(10)))   # → (df/dx, 0.0)
laplacian(itp, (0.5, GridIdx(10)))  # → d²f/dx²
nothing  # hide
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
