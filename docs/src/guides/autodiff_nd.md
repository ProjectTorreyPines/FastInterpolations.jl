# AD Support for N-Dimensional Interpolants

This page covers automatic differentiation for `CubicInterpolantND` (2D, 3D, and higher dimensions).

For 1D interpolants, see [Automatic Differentiation Support](autodiff_support.md).

## Quick Start

```julia
using FastInterpolations, ForwardDiff

# 2D interpolant
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 20)
data = [sin(xi) * cos(yj) for xi in x, yj in y]
itp = cubic_interp((x, y), data)

# Vector API for AD
itp([0.5, 0.5])                           # evaluation
ForwardDiff.gradient(itp, [0.5, 0.5])     # gradient
ForwardDiff.hessian(itp, [0.5, 0.5])      # hessian
```

## Supported APIs

### Tuple API (Original)

```julia
itp((0.5, 0.5))                                    # evaluation
ForwardDiff.derivative(q -> itp((q, 0.5)), 0.5)   # partial derivative
```

### Vector API (AD-friendly)

```julia
itp([0.5, 0.5])                           # evaluation
ForwardDiff.gradient(itp, [0.5, 0.5])     # full gradient
ForwardDiff.hessian(itp, [0.5, 0.5])      # hessian matrix
```

## Performance Comparison

| Method | Time (2D) | Memory | Notes |
|--------|-----------|--------|-------|
| `deriv` keyword | **47ns** | 80 B | Analytical, fastest |
| ForwardDiff | 414ns | 592 B | Dual number propagation |
| Zygote (w/ rrule) | 1.6μs | 6 KiB | Uses ChainRulesCore |
| Zygote (w/o rrule) | 289μs | 1.49 MiB | Source transformation |

!!! tip "ChainRulesCore Extension"
    When `ChainRulesCore` is loaded (automatically with Zygote), the extension provides
    **~160x speedup** for reverse-mode AD by using analytical derivatives internally.

## ForwardDiff Integration

ForwardDiff works seamlessly with both Tuple and Vector APIs:

```julia
using ForwardDiff

# Gradient (Vector API)
grad = ForwardDiff.gradient(itp, [0.5, 0.5])

# Hessian
H = ForwardDiff.hessian(itp, [0.5, 0.5])

# Partial derivatives (Tuple API)
∂f_∂x = ForwardDiff.derivative(q -> itp((q, 0.5)), 0.5)
∂f_∂y = ForwardDiff.derivative(q -> itp((0.5, q)), 0.5)
```

## Zygote Integration

Zygote requires `ChainRulesCore` for efficient differentiation:

```julia
using Zygote

grad = Zygote.gradient(v -> itp(v), [0.5, 0.5])[1]
```

Without the ChainRulesCore extension, Zygote falls back to source transformation,
which is ~160x slower and uses ~250x more memory.

## Optimization with Optim.jl

### Standard Usage (ForwardDiff)

```julia
using Optim

# Direct optimization with AD
result = optimize(itp, [0.3, 0.3], LBFGS(); autodiff=:forward)
```

### Maximum Performance (Analytical Gradient)

For hot loops or performance-critical code:

```julia
# Analytical gradient function
function g!(G, x)
    pt = (x[1], x[2])
    G[1] = itp(pt; deriv=(1, 0))
    G[2] = itp(pt; deriv=(0, 1))
end

# Use with Optim.jl
result = optimize(x -> itp((x[1], x[2])), g!, [0.3, 0.3], LBFGS())
```

This approach is ~10x faster than ForwardDiff for gradient computation.

## Complex-valued Interpolants

For complex data, `gradient` requires scalar output:

```julia
data_c = [sin(xi) + im*cos(yj) for xi in x, yj in y]
itp_c = cubic_interp((x, y), data_c)

# ❌ This fails (gradient expects Real output)
# ForwardDiff.gradient(itp_c, [0.5, 0.5])

# ✅ Option 1: Use jacobian
jac = ForwardDiff.jacobian(
    v -> [real(itp_c(v)), imag(itp_c(v))],
    [0.5, 0.5]
)

# ✅ Option 2: Separate real/imag gradients
grad_re = ForwardDiff.gradient(v -> real(itp_c(v)), [0.5, 0.5])
grad_im = ForwardDiff.gradient(v -> imag(itp_c(v)), [0.5, 0.5])
```

## 3D and Higher Dimensions

All features work for any dimension:

```julia
# 3D interpolant
z = range(0.0, 1.0, 20)
data3d = [xi + yj + zk for xi in x, yj in y, zk in z]
itp3d = cubic_interp((x, y, z), data3d)

# AD works the same way
itp3d([0.5, 0.5, 0.5])
ForwardDiff.gradient(itp3d, [0.5, 0.5, 0.5])  # Returns [1, 1, 1]
```

## Implementation Notes

For detailed implementation information including:
- Type signature changes for AD compatibility
- ChainRulesCore frule/rrule implementation
- Performance analysis details

See the design document: `docs/design/nd_cubic_autodiff_support.md`
