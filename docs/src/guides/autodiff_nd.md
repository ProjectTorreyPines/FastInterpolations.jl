# AD Support for N-Dimensional Interpolants

This page covers automatic differentiation for `CubicInterpolantND` (2D, 3D, and higher dimensions).

For 1D interpolants, see [Automatic Differentiation Support](autodiff_support.md).

!!! tip "Consider Built-in Analytical Derivatives First"
    For **derivatives with respect to query coordinates** (gradient, Hessian, mixed partials),
    FastInterpolations provides analytical functions that are much faster than AD:

    ```julia
    itp(pt; deriv=DerivOp(1, 0))              # partial df/dx, zero-allocation
    FastInterpolations.gradient(itp, pt)       # full gradient (tuple input → zero-alloc)
    FastInterpolations.hessian(itp, pt)        # Hessian matrix
    ```

    AD (ForwardDiff, Zygote) is useful for:
    - **Adjoint computation** (`df/dy`): differentiating w.r.t. *data values*, not query points.
      A built-in analytical adjoint operator is on the roadmap.
    - **Composite pipelines**: when the interpolant is embedded inside a larger differentiable function.

    For query-coordinate derivatives, the built-in analytical functions are the recommended approach.

```@setup autodiff_nd
using FastInterpolations, ForwardDiff

# 2D interpolant
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 20)
data = [sin(xi) * cos(yj) for xi in x, yj in y]
itp = cubic_interp((x, y), data)
```

## Quick Start

```@example autodiff_nd
using FastInterpolations, ForwardDiff  # hide

# Vector API for AD
itp([0.5, 0.5])                           # evaluation
ForwardDiff.gradient(itp, [0.5, 0.5])     # gradient
ForwardDiff.hessian(itp, [0.5, 0.5])      # hessian

# Fast analytical derivatives (much faster; tuple form is zero-allocation)
FastInterpolations.gradient(itp, (0.5, 0.5))   # analytical gradient
FastInterpolations.hessian(itp, (0.5, 0.5))    # analytical hessian
```

## Supported APIs

### Tuple API (Original)

```@example autodiff_nd
itp((0.5, 0.5))                                    # evaluation
ForwardDiff.derivative(q -> itp((q, 0.5)), 0.5)   # partial derivative
```

### Vector API (AD-friendly)

```@example autodiff_nd
itp([0.5, 0.5])                           # evaluation
ForwardDiff.gradient(itp, [0.5, 0.5])     # full gradient
ForwardDiff.hessian(itp, [0.5, 0.5])      # hessian matrix
```

## Performance Comparison

| Method | Allocation | Notes |
|--------|-----------|-------|
| `FastInterpolations.gradient(itp, tuple)` | **0 B** | Analytical, tuple output |
| `FastInterpolations.gradient(itp, vector)` | small | Analytical, allocates output vector |
| `FastInterpolations.hessian(itp, x)` | minimal | Analytical, symmetric matrix |
| `deriv` keyword | minimal | Analytical, single partial |
| ForwardDiff.gradient | moderate | Dual number propagation |
| ForwardDiff.hessian | moderate | Dual number propagation |
| Zygote (w/ rrule) | moderate | Uses ChainRulesCore |
| Zygote (w/o rrule) | **heavy** | Source transformation fallback |

Built-in analytical methods are **much faster** than any AD backend for query-coordinate derivatives.

!!! tip "ChainRulesCore Extension"
    When `ChainRulesCore` is loaded (automatically with Zygote), the extension provides
    a **significant speedup** for reverse-mode AD by using analytical derivatives internally.

## Analytical Gradient & Hessian (Recommended)

FastInterpolations provides built-in `gradient` and `hessian` functions that use analytical
derivatives internally. These are **much faster** than ForwardDiff equivalents. The tuple-query form is zero-allocation; the vector-query form allocates the output vector.

!!! note "Qualified names"
    `gradient` and `hessian` are exported by FastInterpolations, so `gradient(itp, x)` works
    when using FastInterpolations alone. However, since many AD packages (ForwardDiff, Zygote)
    also provide functions with the same name, we use the qualified form
    `FastInterpolations.gradient` throughout this guide to avoid ambiguity.

```@example autodiff_nd
# Gradient: returns (df/dx, df/dy)
grad = FastInterpolations.gradient(itp, (0.5, 0.5))   # Tuple input -> Tuple output
grad = FastInterpolations.gradient(itp, [0.5, 0.5])   # Vector input -> Vector output
```

```@example autodiff_nd
# Hessian: returns NxN symmetric matrix
H = FastInterpolations.hessian(itp, (0.5, 0.5))
# H = [d2f/dx2    d2f/dxdy]
#     [d2f/dxdy   d2f/dy2 ]
```

### When to Use

| Use Case | Recommended Method |
|----------|-------------------|
| Performance-critical code | `FastInterpolations.gradient(itp, x)`, `FastInterpolations.hessian(itp, x)` |
| Optimization with Optim.jl | `FastInterpolations.gradient(itp, x)` for custom gradient |
| General AD compatibility | `ForwardDiff.gradient` |
| Reverse-mode AD (Zygote) | Zygote with ChainRulesCore extension |

### 3D and Higher Dimensions

Works for any dimension:

```@example autodiff_nd
z = range(0.0, 1.0, 20)
data3d = [xi + yj + zk for xi in x, yj in y, zk in z]
itp3d = cubic_interp((x, y, z), data3d)
FastInterpolations.gradient(itp3d, (0.5, 0.5, 0.5))   # -> (df/dx, df/dy, df/dz)
FastInterpolations.hessian(itp3d, (0.5, 0.5, 0.5))    # -> 3x3 symmetric matrix
```

## ForwardDiff Integration

ForwardDiff works seamlessly with both Tuple and Vector APIs:

```@example autodiff_nd
# Gradient (Vector API)
grad = ForwardDiff.gradient(itp, [0.5, 0.5])

# Hessian
H = ForwardDiff.hessian(itp, [0.5, 0.5])

# Partial derivatives (Tuple API)
df_dx = ForwardDiff.derivative(q -> itp((q, 0.5)), 0.5)
df_dy = ForwardDiff.derivative(q -> itp((0.5, q)), 0.5)
nothing  # hide
```

## Zygote Integration

Zygote requires `ChainRulesCore` for efficient differentiation:

```julia
using Zygote

grad = Zygote.gradient(itp, [0.5, 0.5])[1]
```

Without the ChainRulesCore extension, Zygote falls back to source transformation,
which is significantly slower and uses much more memory.

## Optimization with Optim.jl

### Standard Usage (ForwardDiff)

```julia
using Optim

# Direct optimization with AD
result = optimize(itp, [0.3, 0.3], LBFGS(); autodiff=:forward)
```

### Maximum Performance (Analytical Gradient)

For hot loops or performance-critical code, use the built-in analytical gradient:

```julia
# Simple: use FastInterpolations.gradient() directly
function g!(G, x)
    grad = FastInterpolations.gradient(itp, x)
    G .= grad
end

# Use with Optim.jl
result = optimize(x -> itp(x), g!, [0.3, 0.3], LBFGS())
```

Or for maximum control with the `deriv` keyword:

```julia
function g!(G, x)
    pt = (x[1], x[2])
    G[1] = itp(pt; deriv=DerivOp(1, 0))
    G[2] = itp(pt; deriv=DerivOp(0, 1))
end
```

Both approaches are much faster than ForwardDiff for gradient computation.

## Complex-valued Interpolants

For complex data, `ForwardDiff.gradient` requires scalar output:

```@example autodiff_nd
data_c = [sin(xi) + im*cos(yj) for xi in x, yj in y]
itp_c = cubic_interp((x, y), data_c)

# ForwardDiff.gradient(itp_c, [0.5, 0.5])  # fails (gradient expects Real output)

# Option 1: Use jacobian
jac = ForwardDiff.jacobian(
    v -> [real(itp_c(v)), imag(itp_c(v))],
    [0.5, 0.5]
)
```

```@example autodiff_nd
# Option 2: Separate real/imag gradients
grad_re = ForwardDiff.gradient(v -> real(itp_c(v)), [0.5, 0.5])
grad_im = ForwardDiff.gradient(v -> imag(itp_c(v)), [0.5, 0.5])
nothing  # hide
```

## 3D and Higher Dimensions

All features work for any dimension:

```@example autodiff_nd
# 3D interpolant (reusing itp3d from above)
itp3d([0.5, 0.5, 0.5])
ForwardDiff.gradient(itp3d, [0.5, 0.5, 0.5])  # Returns [1, 1, 1]
```

## Full Optimization Example

For a complete real-world example using `gradient!`/`hessian!` with Optim.jl (including
FDM vs AD vs analytical comparison on the Rosenbrock function), see the
[**Optimization with Optim.jl**](@ref optimization_guide) guide.

## Implementation Notes

For detailed implementation information including:
- Type signature changes for AD compatibility
- ChainRulesCore frule/rrule implementation
- Performance analysis details

See the design document: `docs/design/nd_cubic_autodiff_support.md`
