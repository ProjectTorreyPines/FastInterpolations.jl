# Unified API: `interp` / `interp!`

The `interp` function is a **unified entry point** for N-dimensional interpolation. It combines per-axis method specification with automatic optimization dispatch.

## Why `interp`?

Method-specific functions (`cubic_interp`, `linear_interp`, etc.) require the same method on all axes. `interp` removes this restriction:

| API | Use Case |
|:----|:---------|
| `cubic_interp((x,y), data)` | All axes cubic (fastest, full feature set) |
| `linear_interp((x,y), data)` | All axes linear |
| `interp((x,y), data; method=CubicInterp())` | Same as `cubic_interp` (auto-dispatched) |
| `interp((x,y), data; method=(CubicInterp(), LinearInterp()))` | **Mixed methods per axis** |

When all axes use the same method, `interp` delegates to the existing optimized type (e.g., `CubicInterpolantND`). No performance penalty.

---

## Interpolant Construction

```julia
using FastInterpolations

x = range(0.0, 2pi, 50)
y = range(0.0, 1.0, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

# Single method broadcast to all axes
itp = interp((x, y), data; method=CubicInterp())  # returns CubicInterpolantND

# Per-axis methods
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()))  # TensorProductInterpolantND

# With per-axis BCs
itp = interp((x, y), data; method=(CubicInterp(bc=ZeroCurvBC()), CubicInterp(bc=CubicFit())))
```

### Coefficient Strategy

For heterogeneous methods, two build strategies are available:

| Strategy | Build Cost | Eval Cost | When to Use |
|:---------|:-----------|:----------|:------------|
| `PreCompute()` (default) | O(n) | O(1) | Repeated queries on fixed data |
| `OnTheFly()` | O(1) | O(n) | Few queries, or data changes frequently |

```julia
itp_fast = interp((x, y), data; method=(CubicInterp(), LinearInterp()), coeffs=PreCompute())
itp_lazy = interp((x, y), data; method=(CubicInterp(), LinearInterp()), coeffs=OnTheFly())
```

---

## One-Shot Evaluation

Evaluate directly from grids + data without constructing an interpolant. Uses pool-based memory management for **zero allocation** after warmup.

### Scalar

```julia
val = interp((x, y), data, (0.5, 0.3); method=(CubicInterp(), LinearInterp()))
```

### Batch (Allocating)

```julia
queries = ([0.5, 1.0, 1.5], [0.2, 0.4, 0.6])  # SoA format
vals = interp((x, y), data, queries; method=CubicInterp())
```

### Batch (In-Place)

```julia
output = zeros(3)
interp!(output, (x, y), data, queries; method=CubicInterp())
```

Both SoA (`(xqs, yqs)`) and AoS (`[(x1,y1), (x2,y2), ...]`) query formats are supported.

---

## Derivatives and Vector Calculus

All derivative features work with `interp`-constructed interpolants:

```julia
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()))

# Per-axis derivative specification
itp((0.5, 0.3); deriv=(DerivOp(1), DerivOp(0)))  # df/dx

# Vector calculus
gradient(itp, (0.5, 0.3))   # (df/dx, df/dy)
hessian(itp, (0.5, 0.3))    # 2x2 Hessian matrix
laplacian(itp, (0.5, 0.3))  # d^2f/dx^2 + d^2f/dy^2

# One-shot derivative
interp((x, y), data, (0.5, 0.3);
    method=(CubicInterp(), LinearInterp()),
    deriv=(DerivOp(1), DerivOp(0)))
```

---

## Keyword Arguments

All keywords follow the **singular name, tuple-or-scalar value** convention:

| Keyword | Type | Default | Description |
|:--------|:-----|:--------|:------------|
| `method` | `AbstractInterpMethod` or `Tuple` | *required* | Interpolation method per axis |
| `coeffs` | `AbstractCoeffStrategy` | `PreCompute()` | Build strategy (constructor only) |
| `extrap` | `AbstractExtrap` or `Tuple` | `NoExtrap()` | Extrapolation mode per axis |
| `search` | `AbstractSearchPolicy` or `Tuple` | `AutoSearch()` | Search algorithm per axis |
| `deriv` | `DerivOp` or `Tuple` | `EvalValue()` | Derivative order per axis (one-shot) |
| `hint` | `Nothing` or `NTuple{N, RefValue{Int}}` | `nothing` | Persistent search hints (one-shot) |

A scalar value is broadcast to all axes: `method=CubicInterp()` is equivalent to `method=(CubicInterp(), CubicInterp())` in 2D.

---

## Available Methods

| Type | Constructor | Has BC? | Derivative Kernel |
|:-----|:-----------|:--------|:------------------|
| `CubicInterp(; bc=CubicFit())` | C2 cubic spline | Yes | Up to 3rd order |
| `LinearInterp()` | C0 piecewise linear | No | 1st order (2nd+ = 0) |
| `QuadraticInterp(; bc=Left(QuadraticFit()))` | C1 quadratic spline | Yes | Up to 2nd order |
| `ConstantInterp(; side=NearestSide())` | Step function | No (side only) | Always 0 |

---

## Visual Comparison

Per-axis method mixing on a 6x7 non-uniform grid for $f(x,y) = \sin(2\pi x)\cos(2\pi y)$:

![Heterogeneous 2D Comparison](../images/hetero_2d_comparison.png)

Linear x Linear (top-left) shows faceted cells. Adding cubic smoothing per axis progressively improves the result — Cubic x Cubic (bottom-right) captures the extrema accurately.

---

## Examples

### 3D Mixed Methods

```julia
x = range(0, 2pi, 40)
y = range(0, 1, 20)
z = range(0, 5, 30)
data3d = [sin(xi) * yj^2 * exp(-zk) for xi in x, yj in y, zk in z]

itp = interp((x, y, z), data3d;
    method = (CubicInterp(bc=PeriodicBC()), QuadraticInterp(), LinearInterp()),
    extrap = (WrapExtrap(), ClampExtrap(), NoExtrap()),
)

itp((1.0, 0.5, 2.0))
gradient(itp, (1.0, 0.5, 2.0))
```

### Hot Loop with One-Shot

```julia
x, y = range(0, 1, 100), range(0, 1, 50)
method = (CubicInterp(), LinearInterp())

for t in 1:10_000
    data = [f(xi, yj, t) for xi in x, yj in y]  # data changes each step
    val = interp((x, y), data, (0.5, 0.3); method=method)  # zero-alloc
end
```
