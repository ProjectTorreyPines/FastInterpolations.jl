# Adjoint via AD (∂f/∂y)

This page covers differentiating interpolation output **with respect to data values** `y` using automatic differentiation.

For differentiating with respect to **query coordinates** `xq`, see
[AD Support (1D)](autodiff_support.md) and [AD Support (ND)](autodiff_nd.md).

## Background

Interpolation is a **linear operation** on data values: the output is a weighted sum of nearby data points. The gradient `∂f/∂y` recovers these weights — equivalent to the **adjoint** (transpose) of the interpolation operator.

| Direction | Operation | Description |
|-----------|-----------|-------------|
| **Forward** (gather) | `y_interp = A * y_data` | Weighted sum of nearby data → interpolated value |
| **Adjoint** (scatter) | `∇y_data = Aᵀ * δ` | Distribute residuals back to data nodes |

!!! tip "Zygote & Enzyme support for cubic splines (1D)"
    `cubic_interp(x, y, xq; ...)` provides an analytical `rrule` via [`CubicAdjoint`](@ref),
    so **Zygote** and **Enzyme** can differentiate `∂f/∂y` for 1D cubic splines without
    tracing through the tridiagonal solve. See [Backend Support](@ref) below for details.

## Quick Start

```@example adjoint
using FastInterpolations, ForwardDiff

x = 0.0:1.0:5.0
y = cos.(collect(x))

f(y) = linear_interp(x, y, 1.25)
∇y = ForwardDiff.gradient(f, y)
```

`linear_interp(x, y, 1.25)` falls between `y[2]` and `y[3]` with fractional position `t = 0.25`, so the gradient is `(1-t, t) = (0.75, 0.25)` at those indices.

## All Methods Work

```@example adjoint
ForwardDiff.gradient(y -> constant_interp(x, y, 1.25), y)
```

```@example adjoint
ForwardDiff.gradient(y -> quadratic_interp(x, y, 1.25), y)
```

```@example adjoint
ForwardDiff.gradient(y -> cubic_interp(x, y, 1.25), y)
```

For **cubic** and **quadratic** splines, the gradient is non-local: the tridiagonal solve couples all data points, so changing any `y[i]` can affect the output. The dominant weights are still near the query point, with exponentially decaying influence further away.

### ND Interpolants

```@example adjoint
xg = range(0.0, 1.0, 5)
yg = range(0.0, 1.0, 5)
data = [cos(xi) * cos(yj) for xi in xg, yj in yg]

∇data = ForwardDiff.gradient(d -> linear_interp((xg, yg), d, (0.3, 0.7)), data)
```

### Periodic Boundary Conditions

Both inclusive and exclusive `PeriodicBC` work:

```@example adjoint
t = range(0.0, 2π, 7)
y_per = cos.(collect(t))   # cos(0) == cos(2π) == 1.0

ForwardDiff.gradient(y -> cubic_interp(t, y, 1.0; bc=PeriodicBC()), y_per)
```

## Batch Queries: Jacobian

For multiple query points, use `ForwardDiff.jacobian` to build the interpolation matrix:

```@example adjoint
xqs = [1.25, 3.7]

J = ForwardDiff.jacobian(y -> [linear_interp(x, y, xq) for xq in xqs], y)
```

```@example adjoint
# Adjoint: scatter residuals back to data nodes
δ = [1.0, -0.5]
∇y = J' * δ              # Aᵀ * δ
```

## Example: Loss Function

```@example adjoint
xq_obs = [0.3, 1.2, 2.7, 3.8, 4.5]
y_obs = [0.1, 0.8, 0.3, -0.5, -0.9]

function loss(y)
    predictions = [linear_interp(x, y, xq) for xq in xq_obs]
    return sum((predictions .- y_obs) .^ 2)
end

ForwardDiff.gradient(loss, y)
```

This gradient tells you how to adjust each data point to reduce the loss — the foundation for iterative fitting and inverse problems.

## Performance

`ForwardDiff` processes `y` in chunks (default chunk size = 8), reusing the existing optimized interpolation code paths. Because `FastInterpolations` is highly optimized for forward evaluation, the Dual number overhead remains modest.

Representative timings (Apple M1 Pro, single query `∂f/∂y`, Julia 1.12):

| N (grid points) | linear | cubic |
|:---:|:---:|:---:|
| 11 | 0.4 μs | 0.6 μs |
| 51 | 1.4 μs | 4.6 μs |
| 201 | 10 μs | 67 μs |
| 1001 | 203 μs | 1.7 ms |

Cubic is slower per chunk because each chunk re-runs the tridiagonal forward/backward substitution with Dual-valued data. The LU structure (grid spacing, cache) is reused, but the solve itself must execute for each chunk of Dual numbers.

## Backend Support

Computing `∂f/∂y` **requires the one-shot API** — there is no alternative.
Pre-building an interpolant `itp = cubic_interp(x, y)` bakes `y` into the struct as fixed constants;
calling `gradient(itp, xq)` then gives `∂f/∂xq` (coordinate sensitivity), not `∂f/∂y`.
To differentiate with respect to data, the AD backend must see `y` as a live variable, which means
it must trace through the entire construction path: `f(y) = cubic_interp(x, y, xq)`.

This construction path includes in-place array mutations (tridiagonal solve for cubic/quadratic,
slope recurrence for quadratic), which limits backend compatibility:

| Backend | constant | linear | quadratic | cubic (1D) |
|---------|:--------:|:------:|:---------:|:----------:|
| **ForwardDiff** | ✅ (1D/ND) | ✅ (1D/ND) | ✅ (1D/ND) | ✅ (1D/ND) |
| **Zygote** | ✅ | ✅ | ❌¹ | ✅² |
| **Enzyme** | ❌³ | ❌³ | ❌³ | ✅² |

¹ Quadratic one-shot mutates arrays during the spline solve, which Zygote's source-to-source transformation cannot differentiate through.
² Cubic (1D) uses an analytical `rrule` via [`CubicAdjoint`](@ref) — the AD backend never traces through the tridiagonal solve. Enzyme support requires `using Enzyme, ChainRulesCore` to load the extension.
³ Enzyme encounters LLVM codegen errors on the one-shot construction path.

## How It Works

`ForwardDiff` replaces each `y[i]` with a **Dual number** `y[i] + ε`. The interpolation code runs unchanged because all operations are generic over `<: Real`, and `ForwardDiff.Dual <: Real` satisfies the 4 [duck typing](custom_value_types.md) operations (`+`, `-`, `*`, `muladd`).

## `∂f/∂xq` vs `∂f/∂y`

| | `∂f/∂xq` (coordinate) | `∂f/∂y` (data) |
|---|---|---|
| **Use for** | Position optimization, sensitivity | Inverse problems, data fitting |
| **Fastest method** | `deriv` keyword (analytical) | Cubic: `CubicAdjoint` via `rrule`; others: `ForwardDiff` |
| **Docs** | [1D](autodiff_support.md), [ND](autodiff_nd.md) | This page |